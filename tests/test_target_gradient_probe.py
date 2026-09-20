"""CPU-only mathematical and data-split checks for the registered readouts."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies.revision import target_gradient_probe as P


def test_ridge_matches_augmented_least_squares_with_unpenalized_intercept():
    rng = np.random.default_rng(731)
    x = rng.normal(size=(240, 12)) + np.arange(12) * 3
    y = x @ rng.normal(size=(12, 2)) + [71, -32] + rng.normal(size=(240, 2)) * 0.2
    result = P.fit_ridge(x, y)
    penalty = P.RIDGE_FACTOR * x.var(0).mean()
    design = np.column_stack((x, np.ones(len(x))))
    regularizer = np.column_stack((np.sqrt(len(x) * penalty) * np.eye(x.shape[1]), np.zeros(x.shape[1])))
    augmented = np.concatenate((design, regularizer), axis=0)
    target = np.concatenate((y, np.zeros((x.shape[1], 2))), axis=0)
    coefficients = np.linalg.lstsq(augmented, target, rcond=None)[0]
    actual = P.predict_ridge(result, x)
    np.testing.assert_allclose(actual, design @ coefficients, atol=1e-9, rtol=1e-9)


def test_ridge_global_scale_and_translation_do_not_change_predictions():
    rng = np.random.default_rng(62)
    x = rng.normal(size=(200, 8))
    xt = rng.normal(size=(31, 8))
    y = x[:, :2] * [5, 11] + [17, -4]
    expected = P.predict_ridge(P.fit_ridge(x, y), xt)
    # This is a mathematical covariance-scale check, not a production latent edit.
    actual = P.predict_ridge(P.fit_ridge(x * 0.003 + 23, y), xt * 0.003 + 23)
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)


def test_constant_features_return_train_mean_without_numerical_failure():
    x = np.zeros((50, 7))
    y = np.arange(100).reshape(50, 2)
    result = P.fit_ridge(x, y)
    expected = np.broadcast_to(y.mean(0), (3, 2))
    np.testing.assert_allclose(P.predict_ridge(result, np.zeros((3, 7))), expected)
    assert result["effective_rank"] == result["mean_feature_variance"] == result["ridge_penalty"] == 0


def test_metrics_distinguish_coordinate_mae_and_summed_l1():
    target = np.asarray([[0, 0], [10, 10]], dtype=float)
    prediction = target + [3, 4]
    actual = P.metrics(target, prediction)
    assert actual["mae_px"] == 3.5
    assert actual["l1_px"] == 7
    assert actual["euclidean_rmse_px"] == 5
    assert abs(actual["rmse_px"] - np.sqrt(12.5)) < 1e-12
    assert abs(actual["r2"] - 0.5) < 1e-12


def test_samples_are_unique_reproducible_and_address_every_archived_frame():
    shape = (23, 11)
    first, pairs = P.sample_indices(shape, 100, P.SAMPLE_SEED)
    second, _ = P.sample_indices(shape, 100, P.SAMPLE_SEED)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.ravel_multi_index(pairs.T, shape))
    assert len(np.unique(first)) == len(first)
    assert P.array_sha256(pairs) == P.array_sha256(pairs.copy())
    all_flat, _ = P.sample_indices(shape, np.prod(shape), P.SAMPLE_SEED)
    np.testing.assert_array_equal(np.sort(all_flat), np.arange(np.prod(shape)))


def test_archive_validation_rejects_shared_collection_seed_and_changed_world():
    common = {"env": "redcube", "config": "gray_frame_still", "spec": {"wind": False},
              "policy": "uniform random", "num_actions": 4}
    train = dict(common, split="train", seed=0)
    valid = dict(common, split="eval", seed=1000)
    P.validate_archives(train, valid)
    for invalid in (dict(valid, seed=0), dict(valid, seed=None), dict(valid, spec={"wind": True}), dict(valid, split="train")):
        try:
            P.validate_archives(train, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted incompatible archive metadata: {invalid}")


if __name__ == "__main__":
    for name, function in sorted(globals().copy().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
