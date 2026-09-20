from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.nn import functional as F

from .data import (Dataset, load_data, split_receivers, make_graph, environment,
                   build_modules, mask_rows, altered_graph)
from .model import JointModel


@dataclass
class Config:
    data: str | None = None
    coordinates: str | None = None
    synthetic: bool = False
    type_key: str = "celltype2"
    perturbation_key: str = "perturbation"
    tissue_key: str | None = None
    control_label: str = "Control"
    assume_single_tissue: bool = False
    transform: str = "log1p"
    output: str = "results/run"
    device: str = "auto"
    seed: int = 42
    epochs: int = 20
    batch_size: int = 16
    hidden: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_fraction: float = 0.2
    neighbors: int = 16
    graph_mode: str = "knn"
    distance_mode: str = "fixed"
    sender_chunk_size: int = 64
    checkpoint_senders: bool = False
    radius: float | None = None
    distance_power: float = 1.0
    distance_epsilon: float = 0.001
    n_modules: int = 20
    random_ratio: float = 0.2
    high_min: float = 0.4
    high_max: float = 0.5
    module_ratio: float = 0.4
    sender_ratio: float = 0.2
    mixture: tuple = (0.5, 0.25, 0.25)
    validation_ratios: tuple = (0.2, 0.4, 0.5, 0.6)
    lambda_msg: float = 0.05
    lambda_adv: float = 0.01
    huber_delta: float = 1.0
    shuffle_repeats: int = 3
    min_relative_gain: float = 0.01
    delta_rms_threshold: float = 0.001
    absolute_tolerance: float = 0.0001
    cpu_threads: int = 2

    def validate(self):
        if not self.synthetic and not self.data:
            raise ValueError("Set --data or --synthetic")
        if self.epochs < 1 or self.batch_size < 1 or self.hidden < 2 or self.cpu_threads < 1:
            raise ValueError("epochs, batch_size, cpu_threads must be positive; hidden >= 2")
        ratios = [self.random_ratio, self.high_min, self.high_max, self.module_ratio,
                  self.sender_ratio, *self.validation_ratios]
        if any(not 0 < r <= 1 for r in ratios) or self.high_min > self.high_max:
            raise ValueError("Mask ratios must be in (0,1], with high_min <= high_max")
        if not 0 < self.validation_fraction < 1 or self.neighbors < 1 or self.n_modules < 1:
            raise ValueError("Invalid split/graph/module configuration")
        if len(self.mixture) != 3 or min(self.mixture) < 0 or not np.isclose(sum(self.mixture), 1):
            raise ValueError("mixture must contain three nonnegative probabilities summing to 1")
        if min(self.lambda_msg, self.lambda_adv, self.distance_power, self.weight_decay) < 0:
            raise ValueError("Loss weights, weight decay and distance power must be nonnegative")
        if min(self.huber_delta, self.distance_epsilon, self.learning_rate) <= 0:
            raise ValueError("Huber delta, distance epsilon and learning rate must be positive")
        if self.shuffle_repeats < 1 or not self.validation_ratios:
            raise ValueError("Validation requires ratios and at least one shuffle replicate")
        if self.radius is not None and self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.transform not in ("log1p", "raw"):
            raise ValueError("transform must be log1p or raw")
        if self.graph_mode not in ("knn", "all") or self.distance_mode not in ("fixed", "learned"):
            raise ValueError("graph_mode: knn/all; distance_mode: fixed/learned")
        if self.graph_mode == "all" and self.radius is not None:
            raise ValueError("all mode must not have a radius cutoff")
        if self.distance_mode == "learned" and self.distance_power <= 1e-4:
            raise ValueError("learned distance_power must initially exceed 0.0001")
        if self.sender_chunk_size < 1:
            raise ValueError("sender_chunk_size must be positive")


def synthetic_data(seed=42):
    """Execution fixture, not evidence of successful biological decomposition."""
    rng = np.random.default_rng(seed)
    n, g = 80, 24
    coordinates = rng.uniform(0, 10, (n, 2))
    types = rng.integers(0, 2, n)
    perturbation = np.where(np.arange(n) % 4 == 0, "KO", "Control")
    internal = rng.normal(size=(n, 3)) @ rng.normal(size=(3, g)) * 0.2
    distance = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=-1)
    signal = np.exp(-distance) @ (perturbation == "KO").astype(float)
    expression = np.maximum(0, 2 + internal + signal[:, None] * np.linspace(-0.3, 0.6, g))
    return Dataset(np.log1p(expression).astype(np.float32), coordinates, types,
                   ["A", "B"], np.full(n, "synthetic"), perturbation,
                   perturbation == "Control", [f"g{i}" for i in range(g)],
                   [f"cell{i}" for i in range(n)])


def get_data(cfg):
    if cfg.synthetic:
        data = synthetic_data(cfg.seed)
        if cfg.transform == "raw":
            data.x = np.expm1(data.x)
        return data
    return load_data(cfg.data, cfg.coordinates, cfg.type_key, cfg.perturbation_key,
                     cfg.tissue_key, cfg.control_label, cfg.assume_single_tissue, cfg.transform)


def setup(cfg):
    cfg.validate()
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device
    if device == "auto":
        device = "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(device)


def batch(data, graph_set, ids, modules, rng, mode, ratio, sender_ratio, device, blocked=(), sender_masks=None):
    # Each cell occurs once in this local union, including receiver/sender overlaps.
    rows = [[graph[i] for i in ids] for graph in graph_set]
    nodes = np.unique(np.concatenate([ids] + [src for group in rows for src, _ in group]))
    receiver_positions = np.searchsorted(nodes, ids)
    masks = (mask_rows(len(nodes), data.x.shape[1], rng, sender_ratio) if sender_masks is None
             else sender_masks[nodes].copy())
    masks[receiver_positions] = mask_rows(len(ids), data.x.shape[1], rng, ratio,
                                          modules if mode == "module" else None)
    if len(blocked):
        masks[np.isin(nodes, blocked)] = True
    # The corruption boundary is the only model-input code that reads original expression.
    corrupted = data.x[nodes].copy()
    corrupted[masks] = 0
    edges = []
    for group in rows:
        src = np.concatenate([src for src, _ in group])
        dst = np.repeat(np.arange(len(ids)), [len(src) for src, _ in group])
        weights = np.concatenate([weights for _, weights in group])
        edges.append((torch.as_tensor(np.searchsorted(nodes, src), device=device),
                      torch.as_tensor(dst, device=device), torch.as_tensor(weights, device=device)))
    return {
        "corrupted": torch.as_tensor(corrupted, device=device),
        "mask": torch.as_tensor(masks, device=device),
        "types": torch.as_tensor(data.types[nodes], device=device),
        "receivers": torch.as_tensor(receiver_positions, device=device),
        "edges": edges,
        "target_mask": torch.as_tensor(masks[receiver_positions], device=device),
        "nodes": nodes,
    }


def forward(model, b, cached_messages=None):
    return model(b["corrupted"], b["mask"], b["types"], b["receivers"], *b["edges"][0],
                 cached_messages=cached_messages)


def norm(parameters):
    grads = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(grads).sum().sqrt()) if grads else 0.0


def diagnostic_flags(metrics, moments, cfg):
    full = metrics["real_messages"]["huber"]
    internal = metrics["internal_only"]["huber"]
    shuffled = metrics["shuffled_messages"]["huber"]
    conditional = metrics["conditional_shuffled_messages"]["huber"]
    small_gain = lambda reference: reference - full <= max(cfg.absolute_tolerance,
                                                          cfg.min_relative_gain * abs(reference))
    return {
        "full_not_better_than_internal": bool(small_gain(internal)),
        "messages_not_better_than_shuffle": bool(small_gain(shuffled)),
        "sender_state_not_better_than_conditional_shuffle": bool(small_gain(conditional)),
        "external_near_zero": bool(moments["delta_rms_over_target_rms"] < cfg.delta_rms_threshold),
    }


def null_graphs(data, graph, cfg, receivers=None):
    graphs, names, changes = [graph], ["real_messages"], {}
    for kind, conditional, rewire in (("shuffled_messages", False, False),
                                       ("conditional_shuffled_messages", True, False),
                                       ("shuffled_neighborhoods", False, True)):
        for r in range(cfg.shuffle_repeats):
            altered, fraction = altered_graph(data, graph, cfg.seed + 500 + r,
                                               conditional=conditional, rewire=rewire, report_receivers=receivers)
            graphs.append(altered)
            names.append(kind)
            changes.setdefault(kind, []).append(fraction)
    return graphs, names, changes


def independent_batches(ids, graphs, batch_size):
    """No active receiver is a sender to another active receiver, even in null graphs.

    This lets validation vary receiver corruption while holding sender inputs fixed,
    without creating two different corrupted versions of the same cell.
    """
    current, senders = [], set()
    for i in ids:
        incoming = set(int(j) for graph in graphs for j in graph[i][0])
        if current and (len(current) >= batch_size or int(i) in senders or incoming.intersection(current)):
            yield np.asarray(current, dtype=np.int64)
            current, senders = [], set()
        current.append(int(i))
        senders.update(incoming)
    if current:
        yield np.asarray(current, dtype=np.int64)


@torch.no_grad()
def evaluate(model, data, graph, val, modules, q, cfg, device, export=False):
    model.eval()
    graphs, names, changes = null_graphs(data, graph, cfg, val)
    sender_masks = mask_rows(len(data.x), data.x.shape[1], np.random.default_rng(cfg.seed + 9000), cfg.sender_ratio)
    receiver_batches = list(independent_batches(val, graphs, cfg.batch_size))
    # Each sender's corruption is fixed throughout validation. Independent batches
    # ensure any receiver-specific corruption override is never used by an outgoing edge.
    cached = []
    for start in range(0, len(data.x), cfg.sender_chunk_size):
        end = start + cfg.sender_chunk_size
        masked = sender_masks[start:end]
        corrupted = data.x[start:end].copy()
        corrupted[masked] = 0
        cached.append(model.messages(torch.as_tensor(corrupted, device=device),
                                     torch.as_tensor(masked, device=device),
                                     torch.as_tensor(data.types[start:end], device=device)))
    cached_messages = torch.cat(cached)
    report, exports = {}, {}
    regimes = [(f"random_{r:g}", "random", r) for r in cfg.validation_ratios]
    regimes.append(("module", "module", cfg.module_ratio))
    for regime_index, (label, mode, ratio) in enumerate(regimes):
        # Reset each epoch: identical receiver masks and nulls for paired comparisons.
        rng = np.random.default_rng(cfg.seed + 10000 + regime_index)
        totals, counts, per_tissue = {}, {}, {}
        values = {k: [] for k in ("internal", "delta", "target")}
        q_errors = np.zeros(2)
        target_count, actual_masks = 0, []
        saved = {k: [] for k in ("internal", "delta", "prediction", "mask")}
        for ids in receiver_batches:
            b = batch(data, graphs, ids, modules, rng, mode, ratio, cfg.sender_ratio, device,
                      sender_masks=sender_masks)
            out = forward(model, b, cached_messages[torch.as_tensor(b["nodes"], device=device)])
            target = torch.as_tensor(data.x[ids], device=device)
            selected = b["target_mask"]
            target_count += int(selected.sum())
            actual_masks.extend(selected.float().mean(1).cpu().tolist())
            q_target = torch.as_tensor(q[ids], device=device)
            q_errors += [float(F.mse_loss(out["q_internal"], q_target)) * len(ids),
                         float(F.mse_loss(out["q_message"], q_target)) * len(ids)]
            for key in values:
                tensor = target if key == "target" else out[key]
                values[key].append(tensor[selected].cpu().numpy())
            predictions = [("internal_only", out["internal"])]
            for name, edges in zip(names, b["edges"]):
                if name == "real_messages":
                    pred = out["prediction"]
                else:
                    incoming = model.aggregate(out["messages"], *edges, len(ids))
                    _, delta = model.decode(out["h"], incoming, b["types"][b["receivers"]])
                    pred = out["internal"] + delta
                predictions.append((name, pred))
            for name, pred in predictions:
                difference = pred - target
                huber = F.huber_loss(pred, target, reduction="none", delta=cfg.huber_delta)
                totals.setdefault(name, np.zeros(3))
                totals[name] += [float(huber[selected].sum()), float(difference[selected].square().sum()),
                                 float(difference[selected].abs().sum())]
                counts[name] = counts.get(name, 0) + int(selected.sum())
                cell_loss = (huber * selected).sum(1).cpu().numpy()
                cell_count = selected.sum(1).cpu().numpy()
                for tissue in np.unique(data.tissue[ids]):
                    chosen = data.tissue[ids] == tissue
                    entry = per_tissue.setdefault(str(tissue), {}).setdefault(name, [0.0, 0])
                    entry[0] += float(cell_loss[chosen].sum())
                    entry[1] += int(cell_count[chosen].sum())
            if export:
                for key in ("internal", "delta", "prediction"):
                    saved[key].append(out[key].cpu().numpy())
                saved["mask"].append(selected.cpu().numpy())
        metrics = {name: dict(zip(("huber", "mse", "mae"), (sums / counts[name]).tolist()))
                   for name, sums in totals.items()}
        metrics["full_model"] = metrics["real_messages"].copy()
        v = {k: np.concatenate(parts) for k, parts in values.items()}
        moments = {
            "mean_abs_external_delta": float(np.abs(v["delta"]).mean()),
            "std_external_delta": float(v["delta"].std()),
            "variance_internal_hat": float(v["internal"].var()),
            "variance_external_delta": float(v["delta"].var()),
            "delta_rms_over_target_rms": float(np.sqrt(np.mean(v["delta"] ** 2)) /
                                                max(np.sqrt(np.mean(v["target"] ** 2)), 1e-8)),
        }
        tissue_metrics = {t: {name: total / count for name, (total, count) in rows.items()}
                          for t, rows in per_tissue.items()}
        gains = [row["shuffled_messages"] - row["real_messages"] for row in tissue_metrics.values()]
        interval = None
        if len(gains) >= 2:
            bootstrap = np.random.default_rng(cfg.seed).choice(gains, (2000, len(gains))).mean(1)
            interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        flags = diagnostic_flags(metrics, moments, cfg)
        report[label] = {
            "metrics": metrics, "moments_masked_positions": moments, "flags": flags,
            "status": "FAIL_DIAGNOSTIC" if any(flags.values()) else "PASS_DIAGNOSTICS_NOT_CAUSAL_PROOF",
            "n_masked_positions": target_count, "actual_mask_fraction": float(np.mean(actual_masks)),
            "environment_mse": dict(zip(("internal", "message"), (q_errors / len(val)).tolist())),
            "huber_by_tissue": tissue_metrics,
            "shuffle_gain_tissue_bootstrap_95ci": interval,
            "uncertainty_note": "Tissue bootstrap is descriptive; one tissue cannot support an independent-tissue CI.",
            "shuffle_changed_edge_fraction": changes,
            "sender_mask_policy": "fixed across epochs/regimes; no active receiver is another active receiver's sender",
            "distance_power": float(model.distance_power()),
            "neighborhood_null_note": ("Complete graph: same sender set, shuffled assignment to distance weights"
                                       if cfg.graph_mode == "all" else "Randomized sender set at fixed weight slots"),
        }
        if export:
            for key, parts in saved.items():
                exports[f"{label}_{key}"] = np.concatenate(parts)
    return report, exports


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def train(cfg):
    device = setup(cfg)
    output = Path(cfg.output)
    if (output / "checkpoint.pt").exists() or (output / "history.jsonl").exists():
        raise FileExistsError(f"Existing run in {output}; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    data = get_data(cfg)
    train_ids, val = split_receivers(data, cfg.validation_fraction, cfg.seed)
    graph = make_graph(data, cfg.neighbors, cfg.radius, cfg.distance_power, cfg.distance_epsilon, cfg.graph_mode)
    modules = build_modules(data.x, data.types, train_ids, cfg.n_modules)
    q, q_names = environment(data, graph, cfg.control_label)
    q_mean, q_scale = q[train_ids].mean(0), q[train_ids].std(0)
    q_scale[q_scale < 1e-6] = 1
    q = (q - q_mean) / q_scale
    model = JointModel(data.x.shape[1], len(data.type_names), q.shape[1], cfg.hidden,
                       cfg.distance_mode, cfg.distance_power, cfg.sender_chunk_size,
                       cfg.checkpoint_senders).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    counts = {
        "definition": f"obs[{cfg.perturbation_key!r}] == {cfg.control_label!r}; missing/unknown labels excluded",
        "total_cells": len(data.x), "genes": data.x.shape[1], "eligible_receivers": int(data.eligible.sum()),
        "train_receivers": len(train_ids), "validation_receivers": len(val),
        "known_perturbed_senders": int(((data.perturbation != cfg.control_label) &
                                        (data.perturbation != "__unknown__")).sum()),
        "unknown_label_senders": int((data.perturbation == "__unknown__").sum()),
        "all_sender_cells": len(data.x), "type_names": data.type_names,
        "tissues": np.unique(data.tissue).tolist(), "isolated_cells": sum(len(src) == 0 for src, _ in graph),
        "eligible_by_type": {t: int((data.eligible & (data.types == i)).sum())
                             for i, t in enumerate(data.type_names)},
        "eligible_by_tissue": {str(t): int((data.eligible & (data.tissue == t)).sum())
                               for t in np.unique(data.tissue)},
        "evaluation_scope": "within-tissue receiver holdout; not generalization",
        "training_sender_policy": "all expression of validation receivers masked during training",
        "device": str(device), "torch_version": torch.__version__,
        "graph_mode": cfg.graph_mode, "distance_mode": cfg.distance_mode,
        "environment_weight_policy": "fixed at initial distance_power; auxiliary targets do not drift",
    }
    json_write(output / "config.json", asdict(cfg))
    json_write(output / "data_report.json", counts)
    json_write(output / "modules.json", {str(m): [data.genes[j] for j in np.flatnonzero(modules == m)]
                                         for m in np.unique(modules)})
    np.savez_compressed(output / "split.npz", train=train_ids, validation=val,
                        eligible_receiver_mask=data.eligible, cell_ids=np.asarray(data.cell_ids))
    print(json.dumps(counts), flush=True)
    rng = np.random.default_rng(cfg.seed)
    best = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rng.permutation(train_ids)
        totals, n_positions, steps = np.zeros(4), 0, 0
        grad_norms = {name: 0.0 for name in ("internal_encoder", "message_encoders", "external_decoders")}
        reconstruction_grad_norms = {}
        power_gradient_norm = 0.0
        for start in range(0, len(order), cfg.batch_size):
            ids = order[start:start + cfg.batch_size]
            mode = rng.choice(["random", "high", "module"], p=cfg.mixture)
            ratio = (cfg.random_ratio if mode == "random" else
                     rng.uniform(cfg.high_min, cfg.high_max) if mode == "high" else cfg.module_ratio)
            b = batch(data, [graph], ids, modules, rng, mode, ratio, cfg.sender_ratio, device, blocked=val)
            target = torch.as_tensor(data.x[ids], device=device)
            q_target = torch.as_tensor(q[ids], device=device)
            optimizer.zero_grad(set_to_none=True)
            out = forward(model, b)
            selected = b["target_mask"]
            reconstruction = F.huber_loss(out["prediction"][selected], target[selected], delta=cfg.huber_delta)
            message_loss = F.mse_loss(out["q_message"], q_target)
            adversarial_loss = F.mse_loss(out["q_internal"], q_target)
            # GRL flips the encoder gradient exactly once; predictor minimizes positive MSE.
            loss = reconstruction + cfg.lambda_msg * message_loss + cfg.lambda_adv * adversarial_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            if steps == 0:
                # Separate from auxiliary gradients so message gradients cannot be attributed only to q.
                for name in grad_norms:
                    gradients = torch.autograd.grad(reconstruction, tuple(getattr(model, name).parameters()),
                                                    retain_graph=True, allow_unused=True)
                    squares = [g.detach().square().sum() for g in gradients if g is not None]
                    reconstruction_grad_norms[name] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
            loss.backward()
            if cfg.distance_mode == "learned":
                power_gradient_norm += norm((model.raw_distance_power,))
            for name in grad_norms:
                grad_norms[name] += norm(getattr(model, name).parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            count = int(selected.sum())
            totals += np.asarray([float(v.detach()) for v in (loss, reconstruction, message_loss, adversarial_loss)]) * count
            n_positions += count
            steps += 1
        validation, _ = evaluate(model, data, graph, val, modules, q, cfg, device)
        score = float(np.mean([v["metrics"]["full_model"]["huber"] for v in validation.values()]))
        record = {"epoch": epoch,
                  "train": dict(zip(("total", "reconstruction", "message_aux", "internal_env_aux"),
                                    (totals / n_positions).tolist())),
                  "gradient_norms_train_total_preclip_mean": {k: v / steps for k, v in grad_norms.items()},
                  "gradient_norms_reconstruction_only_first_train_batch": reconstruction_grad_norms,
                  "distance_power": float(model.distance_power().detach()),
                  "distance_raw_power_gradient_abs_train_mean": power_gradient_norm / steps,
                  "validation": validation, "selection_score_mean_huber": score}
        with (output / "history.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        json_write(output / "validation_latest.json", record)
        print(f"epoch={epoch} train_huber={record['train']['reconstruction']:.6f} val_huber={score:.6f} "
              f"flags={ {k: v['status'] for k, v in validation.items()} }", flush=True)
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch,
                        "genes": data.genes, "cell_ids": data.cell_ids, "type_names": data.type_names,
                        "train_ids": train_ids.tolist(), "val_ids": val.tolist(), "modules": modules.tolist(),
                        "q_mean": q_mean.tolist(), "q_scale": q_scale.tolist(), "q_names": q_names,
                        "validation": validation}, output / "checkpoint.pt")
    return output


def evaluate_checkpoint(checkpoint, output, device="auto", data_path=None, coordinates=None):
    # Checkpoints produced here contain only tensors and basic Python containers.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = Config(**saved["config"])
    cfg.device = device
    if data_path is not None:
        cfg.data = data_path
    if coordinates is not None:
        cfg.coordinates = coordinates
    device = setup(cfg)
    data = get_data(cfg)
    if data.genes != saved["genes"] or data.cell_ids != saved["cell_ids"] or data.type_names != saved["type_names"]:
        raise ValueError("Evaluation requires the same cell/gene order and cell-type vocabulary as training")
    graph = make_graph(data, cfg.neighbors, cfg.radius, cfg.distance_power, cfg.distance_epsilon, cfg.graph_mode)
    q, names = environment(data, graph, cfg.control_label)
    if names != saved["q_names"]:
        raise ValueError("Environment schema changed")
    q = ((q - np.asarray(saved["q_mean"])) / np.asarray(saved["q_scale"])).astype(np.float32)
    model = JointModel(data.x.shape[1], len(data.type_names), q.shape[1], cfg.hidden,
                       cfg.distance_mode, cfg.distance_power, cfg.sender_chunk_size,
                       cfg.checkpoint_senders).to(device)
    model.load_state_dict(saved["model"])
    val = np.asarray(saved["val_ids"])
    report, exports = evaluate(model, data, graph, val, np.asarray(saved["modules"]), q, cfg, device, export=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists() or (output / "predictions.npz").exists():
        raise FileExistsError("Evaluation output already exists; choose a new directory")
    json_write(output / "evaluation.json", {"checkpoint_epoch": saved["epoch"], "validation": report})
    np.savez_compressed(output / "predictions.npz", **exports,
                        receiver_ids=np.asarray(data.cell_ids)[val], genes=np.asarray(data.genes))
    print(f"Evaluation saved to {output}", flush=True)
