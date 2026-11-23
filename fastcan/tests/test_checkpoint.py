"""Test resumable checkpoints of LazyFastCan."""

from functools import partial

import numpy as np
import pytest

from fastcan import Checkpoint, CheckpointError, LazyFastCan
from fastcan.narx import gen_poly_features, make_poly_ids

# Module-level state used by the interruptible generators below. A generator
# must be defined at module level to be fingerprintable by the checkpoint.
_RUN_STATE = {
    "fail_after": None,
    "calls": 0,
    "id_offset": 0,
    "batch_limit": None,
}


class _JobInterrupted(Exception):
    pass


def _make_data():
    rng = np.random.default_rng(42)
    n_samples, n_features = 200, 6
    X = rng.normal(size=(n_samples, n_features))
    ids = make_poly_ids(n_features, degree=2)
    y = X @ rng.normal(size=(n_features, 3)) + 0.01 * rng.normal(size=(n_samples, 3))
    return X, y, ids


def interruptible_gen(X, skip_indices, ids, batch_size):
    """Module-level generator that aborts after a global number of batches."""
    for id_batch, feat_batch in gen_poly_features(
        X, ids=ids, skip_indices=skip_indices, batch_size=batch_size
    ):
        _RUN_STATE["calls"] += 1
        if (
            _RUN_STATE["batch_limit"] is not None
            and _RUN_STATE["calls"] > _RUN_STATE["batch_limit"]
        ):
            # Simulate a candidate definition that yields fewer batches.
            return
        if (
            _RUN_STATE["fail_after"] is not None
            and _RUN_STATE["calls"] == _RUN_STATE["fail_after"]
        ):
            raise _JobInterrupted("simulated job interruption")
        yield id_batch, feat_batch


def offset_id_gen(X, skip_indices, ids, batch_size):
    """Like :func:`interruptible_gen` but with a mutable offset on each id.

    The offset is read at call time, so the id stream can change without the
    generator's definition fingerprint changing.
    """
    for id_batch, feat_batch in interruptible_gen(
        X, skip_indices, ids=ids, batch_size=batch_size
    ):
        yield id_batch + _RUN_STATE["id_offset"], feat_batch


def _reset(fail_after=None, id_offset=0, batch_limit=None):
    _RUN_STATE["fail_after"] = fail_after
    _RUN_STATE["calls"] = 0
    _RUN_STATE["id_offset"] = id_offset
    _RUN_STATE["batch_limit"] = batch_limit


def _fit_until_interrupted(estimator, X, y, checkpoint):
    with pytest.raises(_JobInterrupted):
        estimator.fit(X, y)
    assert checkpoint.exists()
    return checkpoint.load()


def test_checkpoint_run_matches_uninterrupted(tmp_path):
    """A checkpointed run and a plain run give identical results."""
    X, y, ids = _make_data()
    gen = partial(interruptible_gen, ids=ids, batch_size=4)
    baseline = LazyFastCan(6, feature_generator=gen).fit(X, y)

    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    selector = LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint)
    selector.fit(X, y)

    assert np.array_equal(selector.indices_, baseline.indices_)
    np.testing.assert_allclose(selector.scores_, baseline.scores_)
    # A completed fit cleans up its checkpoint file.
    assert not checkpoint.exists()


@pytest.mark.parametrize("fail_after", [2, 4, 8, 15])
def test_resume_after_interruption_is_consistent(tmp_path, fail_after):
    """Resuming after an interruption reproduces the uninterrupted output."""
    X, y, ids = _make_data()
    n_select = 7
    gen = partial(interruptible_gen, ids=ids, batch_size=3)
    baseline = LazyFastCan(n_select, feature_generator=gen).fit(X, y)

    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    _reset(fail_after=fail_after)
    _fit_until_interrupted(
        LazyFastCan(n_select, feature_generator=gen, checkpoint=checkpoint),
        X,
        y,
        checkpoint,
    )

    _reset()
    resumed = LazyFastCan(n_select, feature_generator=gen, checkpoint=checkpoint).fit(
        X, y
    )

    assert np.array_equal(resumed.indices_, baseline.indices_)
    np.testing.assert_allclose(resumed.scores_, baseline.scores_)
    assert resumed.n_features_ == baseline.n_features_
    assert not checkpoint.exists()


def test_resume_at_round_boundary(tmp_path):
    """An interruption between two rounds is also recoverable."""
    X, y, ids = _make_data()
    n_select = 5
    gen = partial(interruptible_gen, ids=ids, batch_size=100)
    baseline = LazyFastCan(n_select, feature_generator=gen).fit(X, y)

    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    # One round consumes a single batch. The first round completes and its
    # round-boundary checkpoint (round=1, batch=0) is written; the second
    # round then aborts before processing any batch.
    _reset(fail_after=2)
    state = _fit_until_interrupted(
        LazyFastCan(n_select, feature_generator=gen, checkpoint=checkpoint),
        X,
        y,
        checkpoint,
    )
    assert state["round"] == 1
    assert state["batch"] == 0

    _reset()
    resumed = LazyFastCan(n_select, feature_generator=gen, checkpoint=checkpoint).fit(
        X, y
    )
    assert np.array_equal(resumed.indices_, baseline.indices_)
    np.testing.assert_allclose(resumed.scores_, baseline.scores_)


def test_repeated_resumptions(tmp_path):
    """Several interruptions in a row still converge to the same result."""
    X, y, ids = _make_data()
    n_select = 6
    gen = partial(interruptible_gen, ids=ids, batch_size=2)
    baseline = LazyFastCan(n_select, feature_generator=gen).fit(X, y)

    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    for fail_after in (2, 5, 9):
        _reset(fail_after=fail_after)
        with pytest.raises(_JobInterrupted):
            LazyFastCan(n_select, feature_generator=gen, checkpoint=checkpoint).fit(
                X, y
            )
        assert checkpoint.exists()

    _reset()
    resumed = LazyFastCan(n_select, feature_generator=gen, checkpoint=checkpoint).fit(
        X, y
    )
    assert np.array_equal(resumed.indices_, baseline.indices_)
    np.testing.assert_allclose(resumed.scores_, baseline.scores_)
    assert not checkpoint.exists()


def test_resume_refuses_changed_data(tmp_path):
    X, y, ids = _make_data()
    gen = partial(interruptible_gen, ids=ids, batch_size=3)
    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    _reset(fail_after=3)
    _fit_until_interrupted(
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint), X, y, checkpoint
    )

    X_changed = X.copy()
    X_changed[0, 0] += 1.0
    _reset()
    with pytest.raises(CheckpointError, match="X or the target matrix y"):
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint).fit(X_changed, y)

    y_changed = y.copy()
    y_changed[1, 0] += 1.0
    with pytest.raises(CheckpointError, match="X or the target matrix y"):
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint).fit(X, y_changed)


def test_resume_refuses_changed_generator_definition(tmp_path):
    X, y, ids = _make_data()
    gen = partial(interruptible_gen, ids=ids, batch_size=3)
    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    _reset(fail_after=3)
    _fit_until_interrupted(
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint), X, y, checkpoint
    )

    _reset()
    changed_gen = partial(interruptible_gen, ids=ids, batch_size=5)
    with pytest.raises(CheckpointError, match="candidate definition has changed"):
        LazyFastCan(6, feature_generator=changed_gen, checkpoint=checkpoint).fit(X, y)


def test_resume_refuses_changed_candidate_ids(tmp_path):
    """Same generator definition, different yielded ids -> refused."""
    X, y, ids = _make_data()
    gen = partial(offset_id_gen, ids=ids, batch_size=1)
    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    _reset(fail_after=2, id_offset=0)
    _fit_until_interrupted(
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint), X, y, checkpoint
    )

    _reset(id_offset=100)
    with pytest.raises(CheckpointError, match="candidate ids"):
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint).fit(X, y)


def test_resume_refuses_truncated_candidate_stream(tmp_path):
    """Fewer candidate batches than recorded -> refused."""
    X, y, ids = _make_data()
    gen = partial(interruptible_gen, ids=ids, batch_size=3)
    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    _reset(fail_after=4)
    state = _fit_until_interrupted(
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint), X, y, checkpoint
    )
    assert state["batch"] == 3

    # The generator now ends after only two batches, fewer than the three
    # recorded in the checkpoint.
    _reset(batch_limit=2)
    with pytest.raises(CheckpointError, match="fewer candidate batches"):
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint).fit(X, y)


def test_resume_refuses_changed_parameters(tmp_path):
    X, y, ids = _make_data()
    gen = partial(interruptible_gen, ids=ids, batch_size=3)
    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    _reset(fail_after=3)
    _fit_until_interrupted(
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint), X, y, checkpoint
    )

    _reset()
    with pytest.raises(CheckpointError, match="estimator parameters have changed"):
        LazyFastCan(4, feature_generator=gen, checkpoint=checkpoint).fit(X, y)
    with pytest.raises(CheckpointError, match="estimator parameters have changed"):
        LazyFastCan(6, feature_generator=gen, tol=1e-3, checkpoint=checkpoint).fit(X, y)
    with pytest.raises(CheckpointError, match="estimator parameters have changed"):
        LazyFastCan(
            6,
            feature_generator=gen,
            indices_include=[0],
            checkpoint=checkpoint,
        ).fit(X, y)


def test_resume_refuses_corrupted_checkpoint(tmp_path):
    X, y, ids = _make_data()
    gen = partial(interruptible_gen, ids=ids, batch_size=3)
    path = tmp_path / "run.ckpt"
    checkpoint = Checkpoint(path)
    _reset(fail_after=3)
    _fit_until_interrupted(
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint), X, y, checkpoint
    )

    path.write_bytes(b"not a pickle")
    _reset()
    with pytest.raises(CheckpointError, match="Cannot read checkpoint"):
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint).fit(X, y)


def test_resume_with_include(tmp_path):
    """Checkpoints work with prerequisite features."""
    X, y, ids = _make_data()
    gen = partial(interruptible_gen, ids=ids, batch_size=3)
    baseline = LazyFastCan(6, feature_generator=gen, indices_include=[0, 2]).fit(X, y)

    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    _reset(fail_after=3)
    _fit_until_interrupted(
        LazyFastCan(
            6,
            feature_generator=gen,
            indices_include=[0, 2],
            checkpoint=checkpoint,
        ),
        X,
        y,
        checkpoint,
    )
    _reset()
    resumed = LazyFastCan(
        6, feature_generator=gen, indices_include=[0, 2], checkpoint=checkpoint
    ).fit(X, y)
    assert np.array_equal(resumed.indices_, baseline.indices_)
    np.testing.assert_allclose(resumed.scores_, baseline.scores_)


def test_checkpoint_requires_fingerprintable_generator(tmp_path):
    """Local generators cannot be fingerprinted and fail with a clear error."""

    def local_gen(X, skip_indices):
        yield from gen_poly_features(X, ids=_make_data()[2], skip_indices=skip_indices)

    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    X, y, _ = _make_data()
    with pytest.raises(CheckpointError, match="Cannot fingerprint"):
        LazyFastCan(3, feature_generator=local_gen, checkpoint=checkpoint).fit(X, y)
    # No checkpoint file should be left behind.
    assert not checkpoint.exists()


def test_interruption_before_first_batch_starts_fresh(tmp_path):
    """If nothing was processed yet, a rerun simply starts from scratch."""
    X, y, ids = _make_data()
    gen = partial(interruptible_gen, ids=ids, batch_size=3)
    baseline = LazyFastCan(6, feature_generator=gen).fit(X, y)

    checkpoint = Checkpoint(tmp_path / "run.ckpt")
    _reset(fail_after=1)
    with pytest.raises(_JobInterrupted):
        LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint).fit(X, y)
    assert not checkpoint.exists()

    _reset()
    resumed = LazyFastCan(6, feature_generator=gen, checkpoint=checkpoint).fit(X, y)
    assert np.array_equal(resumed.indices_, baseline.indices_)
