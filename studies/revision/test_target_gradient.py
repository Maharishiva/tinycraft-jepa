"""CPU checks for the target-gradient driver; no real experiment is launched.

Run from the repository root:
 JAX_PLATFORMS=cpu python -m studies.revision.test_target_gradient
"""
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import train as T
from studies.revision import target_gradient as G


def test_production_config_matches_original_flags():
    # Exact flags in studies/stoch/train_arms.sh (fixed-step tile_factored arm).
    original = T.Config(seed=2, steps=8500, lr=1e-3, n_actions=4,
                        encoder="tile", emb_dim=128, enc_depth=6, enc_heads=8,
                        tokens=16, token_dim=32, token_hidden=512,
                        predictor="factored", pred_width=256, pred_depth=6,
                        pred_heads=8, history=8)
    assert asdict(G.production_config(2)) == asdict(original)
    assert G.CHECKPOINT_STEPS == (0, 500, 1000, 2000, 4000, 4250, 8500)


def test_stop_target_changes_only_target_derivative():
    pred = jnp.asarray([[0.5, -0.25, 0.75], [0.8, 0.1, -0.6]])
    target = jnp.asarray([[-0.2, 0.4, 0.3], [0.1, 0.7, -0.1]])
    outputs = {}
    for mode in G.MODES:
        outputs[mode] = jax.value_and_grad(G.prediction_mse, argnums=(0, 1))(pred, target, mode)
    np.testing.assert_array_equal(outputs["joint"][0], outputs["stop_target"][0])
    np.testing.assert_array_equal(outputs["joint"][1][0], outputs["stop_target"][1][0])
    np.testing.assert_array_equal(outputs["stop_target"][1][1], np.zeros_like(target))
    np.testing.assert_allclose(outputs["joint"][1][1], 2 * (target - pred) / pred.size, rtol=1e-7)
    assert np.any(np.asarray(outputs["joint"][1][1]) != 0)


def test_sigreg_gradient_reaches_target_embedding():
    emb = jax.random.normal(jax.random.PRNGKey(21), (3, 4, 5)) * 0.4
    key = jax.random.PRNGKey(22)
    weight = 0.09

    def regularizer(z):
        return weight * T.sigreg(z.transpose(1, 0, 2), key, 5, 8)

    def loss(z, mode):
        # Last frame has no input-side path; its stopped-MSE gradient is zero.
        pred = 0.8 * z[:, :-1] + 0.1
        return G.prediction_mse(pred, z[:, 1:], mode) + regularizer(z)

    grad_joint = jax.grad(loss)(emb, "joint")
    grad_stop = jax.grad(loss)(emb, "stop_target")
    grad_reg = jax.grad(regularizer)(emb)
    np.testing.assert_array_equal(grad_stop[:, -1], grad_reg[:, -1])
    assert np.all(np.linalg.norm(np.asarray(grad_reg), axis=-1) > 0)
    assert not np.array_equal(np.asarray(grad_joint[:, -1]), np.asarray(grad_stop[:, -1]))


def test_recorded_samples_equal_training_formula():
    keys = jax.random.split(jax.random.PRNGKey(9), 7)
    recorded = G.sample_ids(keys, batch=5, num_windows=23)
    original = np.stack([np.asarray(jax.random.randint(jax.random.split(key)[0], (5,), 0, 23))
                         for key in keys])
    np.testing.assert_array_equal(recorded, original)


def test_data_identity_rejects_wrong_world():
    meta = {"env": "redcube", "config": "gray_frame_still", "split": "train", "num_actions": 4,
            "spec": {"noise": "gray", "per_episode": False, "wind": False}}
    G.validate_data_identity(meta, "gray_frame_still", G.production_config(0))
    try:
        G.validate_data_identity(meta, "gray_frame_wind", G.production_config(0))
    except ValueError as error:
        assert "metadata does not identify" in str(error)
    else:
        raise AssertionError("wrong world label passed identity validation")


def test_joint_matches_original_and_pair_has_identical_provenance():
    # A small instance exercises the actual tile/factored model, BatchNorm,
    # clipping, AdamW, cosine schedule, sampling, and two compiled NNX scans.
    # Two steps are necessary: original schedule gives the first update lr=0.
    cfg = replace(G.production_config(1), steps=2, chunk=1, batch=2,
                  emb_dim=16, enc_depth=1, enc_heads=2, enc_res=1,
                  token_hidden=32, pred_width=16, pred_depth=1, pred_heads=2,
                  history=2, sigreg_knots=5, sigreg_proj=8)
    rng = np.random.default_rng(31)
    with tempfile.TemporaryDirectory(prefix="target-gradient-test-") as temporary:
        root = Path(temporary)
        data = root / "train.npz"
        np.savez(data,
                 meta=json.dumps({"env": "redcube", "config": "gray_frame_still", "split": "train",
                                  "num_actions": 4, "seed": 31,
                                  "spec": {"noise": "gray", "per_episode": False, "wind": False}}),
                 frames=rng.integers(0, 256, (6, 2, T.H, T.W, 3), dtype=np.uint8),
                 actions=rng.integers(0, cfg.n_actions, (5, 2), dtype=np.int16),
                 dones=np.zeros((5, 2), bool))
        original, _ = T.train(data, root / "original/model.npz", cfg)
        joint, _ = G.train(data, root / "joint", cfg, "joint", "gray_frame_still", (0, 1, 2))
        stop, _ = G.train(data, root / "stop_target", cfg, "stop_target", "gray_frame_still", (0, 1, 2))
        np.testing.assert_array_equal(np.load(root / "original/losses.npz")["history"],
                                      np.load(root / "joint/losses.npz")["history"])
        assert G.weights_sha256(original) == G.weights_sha256(joint), "joint training differs from T.train"
        joint_manifest = json.loads((root / "joint/manifest.json").read_text())
        stop_manifest = json.loads((root / "stop_target/manifest.json").read_text())
        for field in ("initial_weights_sha256", "data_sha256", "data_array_sha256", "cfg",
                      "window_table_sha256", "update_keys_sha256", "sampled_window_ids_sha256"):
            assert joint_manifest[field] == stop_manifest[field], field
        init = T.LeWM(cfg, T.rngs(cfg.seed))
        assert G.weights_sha256(init) == joint_manifest["initial_weights_sha256"]
        for mode, manifest in (("joint", joint_manifest), ("stop_target", stop_manifest)):
            assert manifest["status"] == "complete"
            assert [item["step"] for item in manifest["checkpoints"]] == [0, 1, 2]
            initial_checkpoint, initial_meta = T.load_model(root / mode / "model_step000000.npz")
            assert initial_meta["completed_steps"] == 0
            assert G.weights_sha256(initial_checkpoint.model) == G.weights_sha256(init)
            final_checkpoint, final_meta = T.load_model(root / mode / "model.npz")
            assert final_meta["completed_steps"] == 2
            assert final_meta["cfg"]["steps"] == 2
            assert G.weights_sha256(final_checkpoint.model) == manifest["final_weights_sha256"]
            assert manifest["final_checkpoint_sha256"] == G.sha256(root / mode / "model.npz")
        # The first lr=0 step leaves parameters unchanged; the second nonzero
        # step proves the intervention is active, with no change in initial loss.
        joint_loss = np.load(root / "joint/losses.npz")["history"]
        stop_loss = np.load(root / "stop_target/losses.npz")["history"]
        np.testing.assert_array_equal(joint_loss, stop_loss)
        assert G.weights_sha256(joint) != G.weights_sha256(stop), "stop-target did not alter the update"
        try:
            G.train(data, root / "joint", cfg, "joint", "gray_frame_still", (0, 1, 2))
        except ValueError as error:
            assert "refusing to overwrite" in str(error)
        else:
            raise AssertionError("a completed fit was silently overwritten")


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_"))
    for name, test in tests:
        test()
        print(f"PASS {name}", flush=True)
    print(f"PASS all {len(tests)} target-gradient checks", flush=True)
