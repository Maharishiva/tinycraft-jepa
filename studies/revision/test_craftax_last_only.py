"""Bounded CPU correctness checks; no real training data or GPU is used.

 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m studies.revision.test_craftax_last_only

The small real-model fixture retains 17 actions, nine encoded frames, eight
predictor outputs, BatchNorm, the factored predictor and the original optimizer.
Widths and the update count are reduced only inside this test fixture.
"""
import copy
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import train as T
from studies.revision import craftax_last_only as C
from studies.revision import last_only_objective as LO
from studies.revision import target_gradient as G


def reference():
    return json.loads((C.ROOT / "artifacts/revision/craftax_goal_v2.json").read_text())


def expect_value_error(call, message):
    try:
        call()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected rejection: {message}")


def test_production_config_and_immutable_sources():
    ref = reference()
    for seed in (0, 1, 2):
        cfg = C.config_from_reference(ref, seed)
        assert asdict(cfg) == {**ref["model_metadata"]["cfg"], "seed": seed}
        assert cfg.tokens * cfg.token_dim == 512 and cfg.n_actions == 17
        assert cfg.history == 8 and cfg.batch == 128 and cfg.steps == 8500
    for field, value in (("batch", 64), ("steps", 8499), ("sigreg_weight", 0.1),
                         ("history", 4), ("n_actions", 4), ("token_dim", 128)):
        changed = copy.deepcopy(ref)
        changed["model_metadata"]["cfg"][field] = value
        expect_value_error(lambda: C.config_from_reference(changed, 0), "frozen Craftax recipe")
    expect_value_error(lambda: C.config_from_reference(ref, 3), "predeclared model seeds")
    assert C.verify_sources(ref)["train.py"] == C.BASE_TRAIN_SHA256
    assert C.sha256(G.__file__) == "4dc9280ea077e954096bc7968f3ab19031f82369124b4de62acf5e5e6a14d488"


def test_only_final_residual_both_gradients_and_no_eight_multiplier():
    pred = jnp.arange(2 * 8 * 5, dtype=jnp.float32).reshape(2, 8, 5) / 31
    target = -pred + 0.2
    loss, (grad_pred, grad_target) = jax.value_and_grad(LO.prediction_mse, argnums=(0, 1))(
        pred, target, "last_only")
    expected = np.asarray(2 * (pred[:, -1] - target[:, -1]) / (pred.shape[0] * pred.shape[2]))
    np.testing.assert_allclose(loss, jnp.mean((pred[:, -1] - target[:, -1]) ** 2), rtol=1e-7)
    np.testing.assert_array_equal(grad_pred[:, :-1], np.zeros((2, 7, 5)))
    np.testing.assert_array_equal(grad_target[:, :-1], np.zeros((2, 7, 5)))
    np.testing.assert_allclose(grad_pred[:, -1], expected, rtol=1e-7)
    np.testing.assert_allclose(grad_target[:, -1], -expected, rtol=1e-7)
    assert np.any(np.asarray(grad_pred[:, -1])) and np.any(np.asarray(grad_target[:, -1]))
    # Earlier residuals can grow arbitrarily without affecting the selected loss.
    changed = pred.at[:, :-1].add(1e5)
    np.testing.assert_array_equal(LO.prediction_mse(changed, target, "last_only"), loss)


def test_sigreg_keeps_all_nine_frames_and_both_encoder_paths():
    emb = jax.random.normal(jax.random.PRNGKey(21), (3, 9, 5)) * 0.4
    key = jax.random.PRNGKey(22)
    weight = 0.09

    def regularizer(z):
        return weight * T.sigreg(z.transpose(1, 0, 2), key, 5, 8)

    def loss(z, full_prefix=False):
        pred = 0.8 * z[:, :-1] + 0.1
        mse = (G.prediction_mse(pred, z[:, 1:], "joint") if full_prefix
               else LO.prediction_mse(pred, z[:, 1:], "last_only"))
        return mse + regularizer(z)

    grad_reg = jax.grad(regularizer)(emb)
    grad_last = jax.grad(loss)(emb)
    residual = 0.8 * emb[:, -2] + 0.1 - emb[:, -1]
    expected = jnp.zeros_like(emb).at[:, -2].set(0.8 * 2 * residual / residual.size)
    expected = expected.at[:, -1].set(-2 * residual / residual.size)
    np.testing.assert_allclose(grad_last - grad_reg, expected, atol=1e-7, rtol=1e-6)
    assert np.all(np.linalg.norm(np.asarray(grad_reg), axis=-1) > 0)
    np.testing.assert_array_equal(grad_last[:, :-2], grad_reg[:, :-2])
    assert np.any(np.asarray(expected[:, -2])) and np.any(np.asarray(expected[:, -1]))
    # The original and selected objectives add the identical all-frame term.
    for full_prefix in (False, True):
        pred = 0.8 * emb[:, :-1] + 0.1
        mse = (G.prediction_mse(pred, emb[:, 1:], "joint") if full_prefix
               else LO.prediction_mse(pred, emb[:, 1:], "last_only"))
        np.testing.assert_allclose(loss(emb, full_prefix) - mse, regularizer(emb), rtol=1e-6)


def test_recorded_sampling_is_the_original_formula():
    keys = jax.random.split(jax.random.PRNGKey(9), 7)
    recorded = C.sample_ids(keys, batch=5, num_windows=23)
    original = np.stack([np.asarray(jax.random.randint(jax.random.split(key)[0], (5,), 0, 23))
                         for key in keys])
    np.testing.assert_array_equal(recorded, original)


def test_immutable_update_has_full_forward_and_unchanged_regularizer():
    cfg = replace(C.config_from_reference(reference(), 1), batch=3, sigreg_knots=5, sigreg_proj=8)

    class ToyModel(nnx.Module):
        def __init__(self):
            self.enc = nnx.Param(jnp.asarray([0.5, 0.7, 0.9, 1.1]))
            self.pred = nnx.Param(jnp.asarray([0.8, 0.6, 0.4, 0.2]))

        def encode(self, window, train):
            assert train and window.shape[1] == 9
            return window * self.enc[...]

        def act_embed(self, actions):
            assert actions.shape[1] == 8
            return jax.nn.one_hot(actions, 17)[..., :4]

        def predict(self, emb, act_emb, train):
            assert train and emb.shape[1] == act_emb.shape[1] == 8
            return emb * self.pred[...] + act_emb / 20

    frames = jax.random.normal(jax.random.PRNGKey(51), (12, 2, 4))
    actions = jnp.arange(22, dtype=jnp.int32).reshape(11, 2) % 17
    table = jnp.asarray(T.valid_starts(np.zeros((11, 2), np.int64), 9))
    keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 7), 1)
    k_idx, k_sig = jax.random.split(keys[0])
    ids = jax.random.randint(k_idx, (cfg.batch,), 0, table.shape[0])
    start, env = table[ids, 0], table[ids, 1]
    index = start[:, None] + jnp.arange(9)[None]
    window = frames[index, env[:, None]]
    window_actions = actions[index[:, :-1], env[:, None]]

    def manual_loss(model, stop_target=False, final_only_reg=False):
        emb = model.encode(window, True)
        pred = model.predict(emb[:, :8], model.act_embed(window_actions), True)
        target = jax.lax.stop_gradient(emb[:, -1]) if stop_target else emb[:, -1]
        mse = jnp.mean((pred[:, -1] - target) ** 2)
        reg_emb = emb[:, -1:] if final_only_reg else emb
        reg = T.sigreg(reg_emb.transpose(1, 0, 2), k_sig, cfg.sigreg_knots, cfg.sigreg_proj)
        return mse + cfg.sigreg_weight * reg, (mse, reg)

    expected, actual = ToyModel(), ToyModel()
    (_, expected_losses), grad = nnx.value_and_grad(manual_loss, has_aux=True)(expected)
    _, stopped_grad = nnx.value_and_grad(manual_loss, has_aux=True)(ToyModel(), True, False)
    _, shortened_grad = nnx.value_and_grad(manual_loss, has_aux=True)(ToyModel(), False, True)
    assert not np.allclose(np.asarray(grad.enc.value), np.asarray(stopped_grad.enc.value))
    assert not np.allclose(np.asarray(grad.enc.value), np.asarray(shortened_grad.enc.value))
    nnx.Optimizer(expected, optax.sgd(0.03), wrt=nnx.Param).update(expected, grad)
    opt = nnx.Optimizer(actual, optax.sgd(0.03), wrt=nnx.Param)
    losses = LO.make_run(cfg, "last_only")(actual, opt, frames, actions, table, keys)
    np.testing.assert_allclose(np.asarray(losses[0]), np.asarray(expected_losses), atol=1e-6, rtol=1e-6)
    for wanted, got in zip(jax.tree.leaves(T._weights(expected)), jax.tree.leaves(T._weights(actual))):
        np.testing.assert_allclose(wanted, got, atol=1e-6, rtol=1e-6)


def test_common_driver_full_prefix_matches_original_with_seventeen_actions():
    # Three updates include two nonzero learning-rate steps. All nine frames
    # and all eight predictor positions remain present in this small fixture.
    cfg = replace(C.config_from_reference(reference(), 1), steps=3, chunk=1, batch=2,
                  emb_dim=16, enc_depth=1, enc_heads=2, enc_res=1, token_dim=4,
                  token_hidden=32, pred_width=16, pred_depth=1, pred_heads=2,
                  sigreg_knots=5, sigreg_proj=8, data_budget=24)
    rng = np.random.default_rng(31)
    with tempfile.TemporaryDirectory(prefix="craftax-last-only-test-") as temporary:
        root = Path(temporary)
        data = root / "train.npz"
        dones = np.zeros((12, 2), bool)
        dones[1, 1] = True
        np.savez(data, frames=rng.integers(0, 256, (13, 2, T.H, T.W, 3), dtype=np.uint8),
                 actions=np.arange(24, dtype=np.int16).reshape(12, 2) % 17, dones=dones,
                 # Pickled sentinels would fail allow_pickle=False if loaded.
                 rewards=np.asarray({"do_not_load": True}, dtype=object),
                 meta=np.asarray({"do_not_load": True}, dtype=object))
        provenance = {"data_sha256": C.sha256(data), "sources": C.verify_sources(reference())}
        original, _ = T.train(data, root / "original/model.npz", cfg)
        joint, _ = C._fit(data, root / "joint", cfg, run=G.make_run(cfg, "joint"),
                          provenance=provenance, prediction_loss_mode="full_prefix_joint_test_only")
        last, _ = C._fit(data, root / "last_only", cfg, run=LO.make_run(cfg, "last_only"),
                         provenance=provenance)
        with np.load(root / "original/losses.npz") as a, np.load(root / "joint/losses.npz") as b:
            np.testing.assert_array_equal(a["history"], b["history"])
        assert C.weights_sha256(original) == C.weights_sha256(joint), "common driver differs from T.train"
        assert C.weights_sha256(joint) != C.weights_sha256(last), "last-only intervention is inactive"
        manifests = {name: json.loads((root / name / "manifest.json").read_text())
                     for name in ("joint", "last_only")}
        for field in ("cfg", "initial_weights_sha256", "data_sha256", "data_array_sha256",
                      "window_table_sha256", "update_keys_sha256", "sampled_window_ids_sha256"):
            assert manifests["joint"][field] == manifests["last_only"][field], field
        assert manifests["joint"]["initial_weights_sha256"] == C.weights_sha256(T.LeWM(cfg, T.rngs(cfg.seed)))
        for name, manifest in manifests.items():
            assert manifest["status"] == "complete" and manifest["completed_steps"] == 3
            assert manifest["encoder_frame_count_per_window"] == 9
            assert manifest["predictor_output_count_per_window"] == 8
            assert manifest["sigreg_frame_count_per_window"] == 9
            assert manifest["historical_pairing"] == C.PAIRING
            bundle, meta = T.load_model(root / name / "model.npz")
            assert bundle.decoder is None and bundle.heads is None
            assert meta["cfg"] == asdict(cfg) and meta["completed_steps"] == cfg.steps
            assert meta["source_sha256"] == C.sha256(C.__file__)
            assert meta["base_train_source_sha256"] == C.BASE_TRAIN_SHA256
            assert meta["objective_source_sha256"] == C.OBJECTIVE_SHA256
            assert C.weights_sha256(bundle.model) == manifest["final_weights_sha256"]
            assert C.sha256(root / name / "model.npz") == manifest["final_checkpoint_sha256"]
            with np.load(root / name / "sampling.npz") as audit:
                np.testing.assert_array_equal(audit["window_table"], T.valid_starts(dones.astype(np.int64), 9))
                for array, field in (("window_table", "window_table_sha256"), ("update_keys", "update_keys_sha256"),
                                     ("sampled_window_ids", "sampled_window_ids_sha256")):
                    assert C.array_sha256(audit[array]) == manifest[field]
        with np.load(root / "joint/losses.npz") as a, np.load(root / "last_only/losses.npz") as b:
            # The first lr=0 update preserves equal encoders and regularizers.
            np.testing.assert_array_equal(a["history"][:2, 1], b["history"][:2, 1])
        expect_value_error(lambda: C._fit(data, root / "last_only", cfg,
                           run=LO.make_run(cfg, "last_only"), provenance=provenance), "refusing to overwrite")
        expect_value_error(lambda: C.load_training_arrays(data, replace(cfg, n_actions=4)), "native action ID")


if __name__ == "__main__":
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("these correctness checks must run with JAX_PLATFORMS=cpu")
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_"))
    for name, test in tests:
        test()
        print(f"PASS {name}", flush=True)
    print(f"PASS all {len(tests)} Craftax last-only checks", flush=True)
