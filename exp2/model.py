from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, n_out))


class ExpressionEncoder(nn.Module):
    """Gene-aware token pooling; no cross-cell attention or batch statistics."""

    def __init__(self, hidden):
        super().__init__()
        self.value = mlp(1, hidden, hidden)
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        self.token_net = mlp(hidden, hidden, hidden)
        self.out = nn.Sequential(nn.LayerNorm(hidden), mlp(hidden, hidden, hidden))

    def forward(self, corrupted, mask, genes):
        # Values at masked positions are zeroed *before* ValueMLP as defense in depth.
        values = self.value(corrupted.masked_fill(mask, 0).unsqueeze(-1))
        tokens = genes.unsqueeze(0) + torch.where(mask[..., None], self.mask_token, values)
        return self.out(self.token_net(tokens).mean(1))


class JointModel(nn.Module):
    def __init__(self, n_genes, n_types, q_dim, hidden=64, distance_mode="fixed", distance_power=1.0,
                 sender_chunk_size=64, checkpoint_senders=False):
        super().__init__()
        self.n_types = n_types
        self.distance_mode, self.initial_distance_power = distance_mode, distance_power
        self.sender_chunk_size, self.checkpoint_senders = sender_chunk_size, checkpoint_senders
        if distance_mode not in ("fixed", "learned"):
            raise ValueError("distance_mode must be fixed or learned")
        if distance_mode == "learned":
            if distance_power <= 1e-4:
                raise ValueError("Learned distance power must start above 0.0001")
            value = distance_power - 1e-4
            self.raw_distance_power = nn.Parameter(torch.tensor(value + math.log(-math.expm1(-value))))
        self.genes = nn.Parameter(torch.randn(n_genes, hidden) * 0.02)
        self.internal_encoder = ExpressionEncoder(hidden)
        self.message_encoders = nn.ModuleList(ExpressionEncoder(hidden) for _ in range(n_types))
        self.internal_decoders = nn.ModuleList(mlp(2 * hidden, hidden, 1) for _ in range(n_types))
        self.external_decoders = nn.ModuleList(mlp(3 * hidden, hidden, 1) for _ in range(n_types))
        self.env_internal = mlp(hidden + n_types, hidden, q_dim)
        self.env_message = mlp(hidden, hidden, q_dim)

    def messages(self, corrupted, mask, types):
        result = corrupted.new_zeros((len(corrupted), self.genes.shape[1]))
        for t, encoder in enumerate(self.message_encoders):
            selected = torch.nonzero(types == t, as_tuple=True)[0]
            for chunk in selected.split(self.sender_chunk_size):
                if self.checkpoint_senders and self.training and torch.is_grad_enabled():
                    values = checkpoint(encoder, corrupted[chunk], mask[chunk], self.genes, use_reentrant=False)
                else:
                    values = encoder(corrupted[chunk], mask[chunk], self.genes)
                result[chunk] = values
        return result

    def distance_power(self):
        if self.distance_mode == "learned":
            return nn.functional.softplus(self.raw_distance_power) + 1e-4
        return self.genes.new_tensor(self.initial_distance_power)

    def distance_weights(self, base_weights, edge_dst, n_receivers):
        if self.distance_mode == "fixed" or not len(base_weights):
            return base_weights
        # base alpha ∝ (d+eps)^(-p_initial); raising it to p/p_initial and
        # renormalizing gives exactly alpha ∝ (d+eps)^(-p), with gradient to p.
        logits = base_weights.clamp_min(torch.finfo(base_weights.dtype).tiny).log()
        logits = logits * (self.distance_power() / self.initial_distance_power)
        maxima = logits.new_full((n_receivers,), -torch.inf)
        maxima.scatter_reduce_(0, edge_dst, logits, reduce="amax", include_self=True)
        weights = (logits - maxima[edge_dst]).exp()
        denominator = weights.new_zeros(n_receivers).index_add(0, edge_dst, weights)
        return weights / denominator[edge_dst]

    def aggregate(self, messages, edge_src, edge_dst, weights, n_receivers):
        weights = self.distance_weights(weights, edge_dst, n_receivers)
        result = messages.new_zeros((n_receivers, messages.shape[1]))
        return result.index_add(0, edge_dst, messages[edge_src] * weights[:, None])

    def decode(self, h, incoming, receiver_types):
        n, g = len(h), len(self.genes)
        gene = self.genes.unsqueeze(0).expand(n, -1, -1)
        state = h[:, None].expand(-1, g, -1)
        message = incoming[:, None].expand(-1, g, -1)
        internal = h.new_zeros((n, g))
        delta = h.new_zeros((n, g))
        for t in range(self.n_types):
            selected = receiver_types == t
            if selected.any():
                common = torch.cat((state[selected], gene[selected]), -1)
                internal[selected] = self.internal_decoders[t](common).squeeze(-1)
                actual = torch.cat((common, message[selected]), -1)
                reference = torch.cat((common, torch.zeros_like(message[selected])), -1)
                # Both evaluations remain in the SAME autograd graph. No extra reconstruction loss.
                delta[selected] = (self.external_decoders[t](actual) -
                                   self.external_decoders[t](reference)).squeeze(-1)
        return internal, delta

    def forward(self, corrupted, mask, types, receivers, edge_src, edge_dst, weights, cached_messages=None):
        h = self.internal_encoder(corrupted[receivers], mask[receivers], self.genes)
        if cached_messages is not None and (self.training or torch.is_grad_enabled()):
            raise ValueError("Message caching is only allowed in no-grad evaluation")
        messages = self.messages(corrupted, mask, types) if cached_messages is None else cached_messages
        if cached_messages is not None:
            # Cached receiver rows may have a different mask; replace them even though
            # independent validation batches prohibit them from being used as senders.
            messages = messages.clone()
            messages[receivers] = self.messages(corrupted[receivers], mask[receivers], types[receivers])
        incoming = self.aggregate(messages, edge_src, edge_dst, weights, len(receivers))
        internal, delta = self.decode(h, incoming, types[receivers])
        type_code = nn.functional.one_hot(types[receivers], self.n_types).to(h.dtype)
        return {
            "internal": internal, "delta": delta, "prediction": internal + delta,
            "h": h, "messages": messages, "incoming": incoming,
            "q_internal": self.env_internal(torch.cat((Reverse.apply(h), type_code), -1)),
            "q_message": self.env_message(incoming),
        }
