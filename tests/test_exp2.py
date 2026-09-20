import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from exp2.data import make_graph, build_modules, mask_rows, altered_graph, environment, split_receivers
from exp2.model import JointModel, Reverse
from exp2.runner import (Config, synthetic_data, batch, forward, train, evaluate_checkpoint,
                         diagnostic_flags, independent_batches)


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    data = synthetic_data(4)
    graph = make_graph(data, neighbors=3)
    train_ids, val = split_receivers(data, 0.2, 4)
    modules = build_modules(data.x, data.types, train_ids, 5)
    q, _ = environment(data, graph, "Control")
    model = JointModel(data.x.shape[1], len(data.type_names), q.shape[1], hidden=12)
    return data, graph, train_ids, val, modules, model


def test_joint_reconstruction_reaches_every_branch_without_auxiliary(fixture):
    data, graph, ids, val, modules, model = fixture
    b = batch(data, [graph], ids[:8], modules, np.random.default_rng(4), "random", 0.5, 0.2, "cpu", val)
    out = forward(model, b)
    selected = b["target_mask"]
    F.huber_loss(out["prediction"][selected], torch.tensor(data.x[ids[:8]])[selected]).backward()
    for component in (model.internal_encoder, model.internal_decoders, model.message_encoders,
                      model.external_decoders):
        assert sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None) > 0
    assert all(p.requires_grad for p in model.parameters())


def test_masked_values_cannot_change_any_prediction_or_message(fixture):
    data, graph, ids, val, modules, model = fixture
    b = batch(data, [graph], ids[:8], modules, np.random.default_rng(3), "module", 0.4, 0.2, "cpu", val)
    before = forward(model, b)
    b["corrupted"][b["mask"]] = 1e8
    after = forward(model, b)
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    blocked = np.isin(b["nodes"], val)
    assert b["mask"][blocked].all()


def test_external_zero_reference_and_signed_output(fixture):
    *_, model = fixture
    h, incoming = torch.randn(4, 12), torch.randn(4, 12)
    types = torch.tensor([0, 1, 0, 1])
    internal, delta = model.decode(h, torch.zeros_like(incoming), types)
    assert torch.count_nonzero(delta) == 0
    _, real_delta = model.decode(h, incoming, types)
    assert torch.count_nonzero(real_delta) > 0
    # A negative output is representable: negating final weights negates all deltas.
    for decoder in model.external_decoders:
        with torch.no_grad():
            decoder[-1].weight.neg_()
            decoder[-1].bias.neg_()
    _, flipped = model.decode(h, incoming, types)
    torch.testing.assert_close(flipped, -real_delta)


def test_grl_reverses_encoder_but_not_predictor_gradient():
    x = torch.tensor([2.0], requires_grad=True)
    weight = torch.tensor([3.0], requires_grad=True)
    (weight * Reverse.apply(x)).sum().backward()
    assert x.grad.item() == -3
    assert weight.grad.item() == 2


def test_graph_weights_no_self_no_cross_tissue_and_nulls(fixture):
    data, *_ = fixture
    data.tissue = np.where(np.arange(len(data.x)) < 40, "t1", "t2")
    data.coordinates[1] = data.coordinates[0]  # Duplicate coordinates still must not create self edges.
    graph = make_graph(data, neighbors=5, power=2)
    variants = [graph]
    for conditional, rewire in ((False, False), (True, False), (False, True)):
        alternate, fraction = altered_graph(data, graph, 9, conditional, rewire)
        assert 0 <= fraction <= 1
        variants.append(alternate)
        if conditional:
            for (old, w), (new, w2) in zip(graph, alternate):
                np.testing.assert_array_equal(data.types[old], data.types[new])
                np.testing.assert_array_equal(data.perturbation[old], data.perturbation[new])
                np.testing.assert_array_equal(w, w2)
    for variant in variants:
        for i, (src, weights) in enumerate(variant):
            assert i not in src
            assert np.all(data.tissue[src] == data.tissue[i])
            assert np.isfinite(weights).all()
            assert weights.sum() == pytest.approx(1)
    i = 4
    src, weights = graph[i]
    raw = 1 / (np.linalg.norm(data.coordinates[src] - data.coordinates[i], axis=1) + 0.001) ** 2
    np.testing.assert_allclose(weights, raw / raw.sum(), rtol=1e-6)


def test_modules_ignore_validation_values_and_mask_whole_modules(fixture):
    data, graph, ids, val, modules, _ = fixture
    changed = data.x.copy()
    changed[val] = np.random.default_rng(42).normal(0, 100, changed[val].shape)
    np.testing.assert_array_equal(modules, build_modules(changed, data.types, ids, 5))
    masks = mask_rows(10, len(modules), np.random.default_rng(0), 0.4, modules)
    for m in np.unique(modules):
        assert np.all(masks[:, modules == m] == masks[:, modules == m][:, :1])


def test_validation_receivers_never_supervised_in_training(fixture):
    data, graph, ids, val, modules, _ = fixture
    assert set(ids).isdisjoint(val)
    assert data.eligible[ids].all() and data.eligible[val].all()
    b = batch(data, [graph], ids, modules, np.random.default_rng(0), "random", 0.2, 0.2, "cpu", val)
    for local, global_id in enumerate(b["nodes"]):
        if global_id in val:
            assert b["mask"][local].all()
            assert b["corrupted"][local].count_nonzero() == 0
    assert len(np.unique(b["nodes"])) == len(b["nodes"])


def test_isolated_receivers_have_no_external_output(fixture):
    data, _, ids, _, modules, model = fixture
    graph = make_graph(data, neighbors=3, radius=1e-12)
    b = batch(data, [graph], ids[:3], modules, np.random.default_rng(0), "random", 0.5, 0.2, "cpu")
    out = forward(model, b)
    assert out["incoming"].count_nonzero() == 0
    assert out["delta"].count_nonzero() == 0


def test_collapse_flags_do_not_accept_zero_delta():
    metrics = {name: {"huber": 1.0} for name in ("real_messages", "internal_only", "shuffled_messages",
                                                "conditional_shuffled_messages")}
    assert all(diagnostic_flags(metrics, {"delta_rms_over_target_rms": 0.0}, Config()).values())


def test_receiver_ratio_changes_leave_validation_sender_inputs_fixed(fixture):
    data, graph, _, val, modules, model = fixture
    alternate, _ = altered_graph(data, graph, 7)
    graphs = [graph, alternate]
    fixed_sender_masks = mask_rows(len(data.x), data.x.shape[1], np.random.default_rng(1), 0.2)
    visited = []
    for ids in independent_batches(val, graphs, 8):
        visited.extend(ids.tolist())
        low = batch(data, graphs, ids, modules, np.random.default_rng(2), "random", 0.2, 0.2,
                    "cpu", sender_masks=fixed_sender_masks)
        high = batch(data, graphs, ids, modules, np.random.default_rng(2), "random", 0.6, 0.2,
                     "cpu", sender_masks=fixed_sender_masks)
        for src, _, _ in low["edges"]:
            assert not set(low["nodes"][src]).intersection(ids)
            torch.testing.assert_close(low["corrupted"][src], high["corrupted"][src], atol=0, rtol=0)
            torch.testing.assert_close(low["mask"][src], high["mask"][src], atol=0, rtol=0)
    assert visited == val.tolist()


def test_checkpoint_round_trip_and_cpu_training(tmp_path):
    import json
    cfg = Config(synthetic=True, epochs=1, batch_size=16, hidden=8, neighbors=2,
                 shuffle_repeats=1, validation_ratios=(0.2,), n_modules=4,
                 output=str(tmp_path / "train"), device="cpu", lambda_adv=0, lambda_msg=0)
    train(cfg)
    evaluate_checkpoint(tmp_path / "train/checkpoint.pt", tmp_path / "eval", "cpu")
    saved = torch.load(tmp_path / "train/checkpoint.pt", weights_only=True)
    report = json.loads((tmp_path / "eval/evaluation.json").read_text())
    for regime in saved["validation"]:
        for variant in saved["validation"][regime]["metrics"]:
            assert report["validation"][regime]["metrics"][variant] == pytest.approx(
                saved["validation"][regime]["metrics"][variant], rel=1e-6)
    arrays = np.load(tmp_path / "eval/predictions.npz")
    np.testing.assert_allclose(arrays["random_0.2_prediction"],
                               arrays["random_0.2_internal"] + arrays["random_0.2_delta"])
    with pytest.raises(FileExistsError):
        train(cfg)


def test_all_graph_uses_every_other_cell_in_same_tissue(fixture):
    data, *_ = fixture
    data.tissue = np.where(np.arange(len(data.x)) < 40, "t1", "t2")
    graph = make_graph(data, neighbors=1, mode="all", power=1.5)
    assert not isinstance(graph, list)
    for i, (src, weights) in enumerate(graph):
        expected = np.flatnonzero((data.tissue == data.tissue[i]) & (np.arange(len(data.x)) != i))
        np.testing.assert_array_equal(src, expected)
        assert len(src) == 39
        raw = (np.linalg.norm(data.coordinates[src] - data.coordinates[i], axis=1) + 0.001) ** -1.5
        np.testing.assert_allclose(weights, raw / raw.sum(), rtol=1e-6)
    with pytest.raises(ValueError):
        make_graph(data, mode="all", radius=2)


def test_learned_distance_matches_inverse_power_and_has_gradient():
    model = JointModel(3, 1, 2, hidden=4, distance_mode="learned", distance_power=1)
    distances = torch.tensor([1., 2., 4., 1., 3.])
    dst = torch.tensor([0, 0, 0, 1, 1])
    base = 1 / (distances + 0.001)
    base = base / torch.zeros(2).index_add(0, dst, base)[dst]
    initial = model.distance_weights(base, dst, 2)
    torch.testing.assert_close(initial, base)
    weights = model.distance_weights(base, dst, 2)
    (weights * distances).sum().backward()
    assert model.raw_distance_power.grad.abs() > 0
    with torch.no_grad():
        model.raw_distance_power.add_(1)
    p = model.distance_power()
    expected = (distances + 0.001).pow(-p)
    expected = expected / torch.zeros(2).index_add(0, dst, expected)[dst]
    torch.testing.assert_close(model.distance_weights(base, dst, 2), expected)
    assert p > 0
    assert weights[0] > weights[1] > weights[2]


def test_checkpoint_sender_chunks_preserve_joint_gradients(fixture):
    data, _, ids, val, modules, model = fixture
    graph = make_graph(data, mode="all")
    checkpointed = copy.deepcopy(model)
    checkpointed.checkpoint_senders = True
    checkpointed.sender_chunk_size = 7
    b = batch(data, [graph], ids[:3], modules, np.random.default_rng(2), "random", 0.5, 0.2, "cpu", val)
    out = forward(model, b)
    other = forward(checkpointed, b)
    torch.testing.assert_close(out["prediction"], other["prediction"], rtol=1e-5, atol=1e-7)
    out["prediction"].square().mean().backward()
    other["prediction"].square().mean().backward()
    for (name, p), (_, q) in zip(model.named_parameters(), checkpointed.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-7, msg=name)


def test_cached_evaluation_matches_uncached_corrupted_inputs(fixture):
    data, _, ids, _, modules, model = fixture
    graph = make_graph(data, mode="all")
    fixed = mask_rows(len(data.x), data.x.shape[1], np.random.default_rng(1), 0.2)
    b = batch(data, [graph], ids[:1], modules, np.random.default_rng(2), "random", 0.6, 0.2,
              "cpu", sender_masks=fixed)
    model.eval()
    x = torch.tensor(data.x).masked_fill(torch.tensor(fixed), 0)
    with torch.no_grad():
        cached = model.messages(x, torch.tensor(fixed), torch.tensor(data.types))
        actual = forward(model, b)
        optimized = forward(model, b, cached[b["nodes"]])
    for key in actual:
        torch.testing.assert_close(actual[key], optimized[key])


def test_all_learned_checkpoint_round_trip(tmp_path):
    import json
    cfg = Config(synthetic=True, epochs=1, batch_size=16, hidden=8, graph_mode="all",
                 distance_mode="learned", checkpoint_senders=True, sender_chunk_size=16,
                 shuffle_repeats=1, validation_ratios=(0.2,), n_modules=4,
                 output=str(tmp_path / "train"), device="cpu", lambda_adv=0, lambda_msg=0)
    train(cfg)
    saved = torch.load(tmp_path / "train/checkpoint.pt", weights_only=True)
    assert "raw_distance_power" in saved["model"]
    latest = json.loads((tmp_path / "train/validation_latest.json").read_text())
    assert latest["distance_raw_power_gradient_abs_train_mean"] > 0
    evaluate_checkpoint(tmp_path / "train/checkpoint.pt", tmp_path / "eval", "cpu")
    result = json.loads((tmp_path / "eval/evaluation.json").read_text())["validation"]
    for regime in saved["validation"]:
        assert result[regime]["distance_power"] == saved["validation"][regime]["distance_power"]
        assert result[regime]["metrics"]["real_messages"] == pytest.approx(saved["validation"][regime]["metrics"]["real_messages"])
