"""
Lazy FastCan selector.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

import hashlib
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils._array_api import (
    get_namespace_and_device,
    move_to,
    supported_float_dtypes,
)
from sklearn.utils._param_validation import Interval

from ._checkpoint import (
    Checkpoint,
    CheckpointError,
    _array_fingerprint,
    _generator_fingerprint,
    _ids_fingerprint,
)
from ._fastcan import _check_indices_params, _check_X_y


def _classical_gram_schmidt(x, W, xp):
    """Classical Gram-Schmidt orthogonalization.
    Parameters
    ----------
    x: (n_samples, batch_size) Features for orthogonalization.
    W: (n_samples, n_selected) Selected orthonormal features,
        where n_selected can be zero.

    Returns
    -------
    q: (n_samples, batch_size) Orthonormal features.
    """
    q = x - W @ (W.T @ x)
    norm = xp.linalg.vector_norm(q, axis=0)
    norm_safe = xp.where(norm == 0, xp.ones_like(norm), norm)
    return q / norm_safe


def _skip2valid(skip_indices, n_features, xp, device):
    """Convert skip_indices to valid_indices for feature generators."""
    skip_indices = _check_indices_params(skip_indices, n_features, xp=xp, device=device)
    all_indices = xp.arange(n_features, dtype=xp.int64, device=device)
    skip_mask = xp.isin(all_indices, skip_indices)
    valid_indices = all_indices[~skip_mask]
    return valid_indices


def _default_feature_generator(X, skip_indices, batch_size=16):
    """Default feature generator that yields batches of columns from X."""
    xp, _, device_ = get_namespace_and_device(X)
    n_features = X.shape[1]
    valid_indices = _skip2valid(skip_indices, n_features, xp=xp, device=device_)

    n_valid = valid_indices.shape[0]
    for i in range(0, n_valid, batch_size):
        stop_idx = min(i + batch_size, n_valid)
        batch_idx = valid_indices[i:stop_idx]
        yield batch_idx, xp.take(X, batch_idx, axis=1)


def _check_generated_features(ids, feats, n_samples, sample_mask, xp):
    """Check the validity of generated features."""
    if (not hasattr(ids, "ndim")) or (ids.ndim != 1):
        raise ValueError(
            f"Generated feature index {ids} is not a one-dimensional array."
        )
    if (not hasattr(feats, "ndim")) or (feats.ndim != 2):
        raise ValueError(
            f"Generated feature with index {ids} is not two-dimensional array."
        )
    if not xp.isdtype(ids.dtype, xp.int64):
        raise TypeError(f"Generated feature index {ids} is not an array of integers.")
    if xp.any(ids < 0):
        raise ValueError(f"Generated feature index {ids} contains negative values.")
    if feats.shape[0] != n_samples:
        raise ValueError(
            f"Generated feature with index {ids} has length {feats.shape[0]}, "
            f"but expected length is {n_samples}."
        )
    feature_masked = feats[sample_mask]
    if not xp.all(xp.isfinite(feature_masked)):
        raise ValueError(
            f"Generated feature with index {ids} contains non-finite values."
        )
    return feature_masked


class LazyFastCan(BaseEstimator):
    """Lazy version of FastCan selector.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    n_features_to_select : int, default=1
        The parameter is the absolute number of features to select.

    feature_generator : callable or None, default=None
        A generator that takes the input data `X` and an array of `skip_indices`
        and yields `(index, feature)` tuples,
        where `index` should be an 1D array of integers from 0 to n_features-1.
        The feature should be a 2D array of shape (n_samples, batch_size).
        The `skip_indices` indicates which features should be skipped.
        If the generator requires additional parameters,
        you can use `functools.partial` to wrap the generator with those parameters.

    sample_mask : ndarray of shape (n_samples,), default=None
        Bool mask for valid samples.

    indices_include : array-like of shape (n_inclusions,), default=None
        The indices of the prerequisite features.

    tol : float, default=1e-6
        Tolerance for linear dependence check.
        The classical Gram-Schmidt fails when abs(x.T @ W) > `tol`.

    checkpoint : :class:`~fastcan.Checkpoint` or None, default=None
        Resumable round state for interrupting and restarting a long run.
        When a :class:`~fastcan.Checkpoint` is given, the progress
        is persisted after every processed candidate batch. If the fit is
        interrupted, calling :meth:`fit` again with the same checkpoint resumes
        the run from the beginning of the unfinished batch and yields exactly
        the same ``indices_`` and ``scores_`` as an uninterrupted run.
        Reusing a checkpoint is explicitly refused with a
        :class:`~fastcan.CheckpointError` when the data ``X``/``y``,
        the candidate definition (the feature generator or its arguments) or
        the estimator parameters have changed. The checkpoint file is removed
        when the fit finishes successfully.

    Attributes
    ----------
    indices_ : ndarray of shape (n_features_to_select,), dtype=int
        The indices of the selected features. The order of the indices
        is corresponding to the feature selection process.

    scores_ : ndarray of shape (n_features_to_select,), dtype=float
        The h-correlation scores of selected features. The order of
        the scores is corresponding to the feature selection process.

    indices_include_ : ndarray of shape (n_inclusions,), dtype=int
        The indices of the prerequisite features.


    Examples
    --------
    >>> from functools import partial
    >>> from fastcan import LazyFastCan
    >>> from fastcan.narx import gen_time_shift_features, make_time_shift_ids
    >>> fg = partial(gen_time_shift_features, ids=make_time_shift_ids(2, 2))
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> y = [[0, 0], [0, 1], [1, 1], [0, 2]]
    >>> LazyFastCan(2, feature_generator=fg).fit(X, y).indices_
    array([2, 1])
    """

    _parameter_constraints: dict = {
        "n_features_to_select": [
            Interval(Integral, 1, None, closed="left"),
        ],
        "feature_generator": [callable, None],
        "sample_mask": ["array-like", None],
        "indices_include": [None, "array-like"],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "checkpoint": [Checkpoint, None],
    }

    def __init__(
        self,
        n_features_to_select=1,
        feature_generator=None,
        sample_mask=None,
        indices_include=None,
        tol=1e-6,
        checkpoint=None,
    ):
        self.n_features_to_select = n_features_to_select
        self.feature_generator = feature_generator
        self.sample_mask = sample_mask
        self.indices_include = indices_include
        self.tol = tol
        self.checkpoint = checkpoint

    def fit(self, X, y):
        self._validate_params()
        xp, is_array_api, device_ = get_namespace_and_device(X)
        if is_array_api:
            y = move_to(y, xp=xp, device=device_)
        X, y = _check_X_y(self, X, y, dtype=supported_float_dtypes(xp, device_), xp=xp)
        feature_generator = self.feature_generator or _default_feature_generator
        n_samples = X.shape[0]
        if self.sample_mask is None:
            sample_mask = xp.ones(n_samples, dtype=xp.bool, device=device_)
        else:
            sample_mask = check_array(self.sample_mask, dtype=xp.bool, ensure_2d=False)
            if sample_mask.shape[0] != n_samples:
                raise ValueError(
                    f"The length of sample_mask {sample_mask.shape[0]} does not match "
                    f"the number of samples {n_samples}."
                )
        # Not checking n_features related error, as n_features is unknown in lazy mode
        self.indices_include_ = _check_indices_params(
            self.indices_include, xp=xp, device=device_
        )
        indices_include = xp.asarray(self.indices_include_, copy=True)

        checkpoint = self.checkpoint
        fingerprints = None
        resume_state = None
        if checkpoint is not None:
            fingerprints = self._make_fingerprints(
                X, y, sample_mask, feature_generator, xp
            )
            if checkpoint.exists():
                resume_state = self._load_checkpoint(checkpoint, fingerprints)

        if resume_state is None:
            y_masked = y[sample_mask]
            y_transformed, _ = xp.linalg.qr(
                y_masked - xp.mean(y_masked, axis=0), mode="reduced"
            )
            n_samples_masked = y_masked.shape[0]
            W = xp.zeros(
                (n_samples_masked, self.n_features_to_select),
                dtype=X.dtype,
                device=device_,
            )
            indices = xp.zeros(
                self.n_features_to_select, dtype=xp.int64, device=device_
            )
            scores = xp.zeros(self.n_features_to_select, dtype=X.dtype, device=device_)
            max_feat_idx = xp.asarray(-1, device=device_)
            start_round = 0
        else:
            y_transformed = xp.asarray(
                resume_state["y_transformed"], dtype=X.dtype, device=device_
            )
            W = xp.asarray(resume_state["W"], dtype=X.dtype, device=device_)
            indices = xp.asarray(
                resume_state["indices"], dtype=xp.int64, device=device_
            )
            scores = xp.asarray(resume_state["scores"], dtype=X.dtype, device=device_)
            max_feat_idx = xp.asarray(
                resume_state["max_feat_idx"], dtype=xp.int64, device=device_
            )
            indices_include = xp.asarray(
                resume_state["indices_include"], dtype=xp.int64, device=device_
            )
            start_round = resume_state["round"]

        for i in range(start_round, self.n_features_to_select):
            best_score = -xp.inf
            best_index = None
            best_feat = None
            # Running hash of the candidate-id stream of the current round.
            ids_hasher = hashlib.sha256()
            # Per-batch running hashes, enabling batch-by-batch verification
            # of the replayed candidate stream after a resume.
            ids_hashes = []
            n_batches_seen = 0
            batch_cursor = 0
            if resume_state is not None and i == start_round:
                # The checkpoint stores the best candidate found over the
                # batches processed before the interruption. Replay the whole
                # candidate stream: the already processed batches are only
                # used to verify the id stream and to rebuild this incumbent,
                # so the selection result stays identical to an uninterrupted
                # run regardless of where the interruption happened.
                batch_cursor = resume_state["batch"]
                if batch_cursor > 0 and resume_state["best_index"] != -1:
                    # All processed batches were linearly dependent and no
                    # incumbent exists yet: keep best_index/best_feat as None.
                    best_score = float(resume_state["best_score"])
                    best_index = int(resume_state["best_index"])
                    best_feat = xp.asarray(
                        resume_state["best_feat"], dtype=X.dtype, device=device_
                    )
            for id_batch, feat_batch in feature_generator(X, skip_indices=indices[:i]):
                ids_hasher.update(_ids_fingerprint(id_batch))
                batch_ids_hash = ids_hasher.copy().digest()
                n_batches_seen += 1
                is_replayed = n_batches_seen <= batch_cursor
                # Verify the replayed stream batch by batch against the
                # per-batch hashes recorded before the interruption, so a
                # changed candidate id is detected even if the change lies
                # inside the last processed batch.
                if is_replayed and (
                    batch_ids_hash != resume_state["ids_hashes"][n_batches_seen - 1]
                ):
                    raise CheckpointError(
                        f"Refusing to reuse checkpoint "
                        f"{checkpoint.path!r}: the candidate ids "
                        "yielded by the feature generator differ from "
                        "those recorded in the checkpoint. The "
                        "candidate definition has effectively "
                        "changed. Delete the checkpoint file to start "
                        "a fresh run."
                    )
                # Sanity check on the final replayed batch: the running hash
                # must also match the aggregate hash stored in the checkpoint.
                if (
                    is_replayed
                    and n_batches_seen == batch_cursor
                    and ids_hasher.copy().hexdigest() != resume_state["ids_hash"]
                ):
                    raise CheckpointError(
                        f"Refusing to reuse checkpoint "
                        f"{checkpoint.path!r}: corrupted candidate-id "
                        "stream record."
                    )
                ids_hashes.append(batch_ids_hash)
                feat_batch = _check_generated_features(
                    id_batch, feat_batch, n_samples, sample_mask, xp
                )
                # Replayed batches also contribute to the largest candidate id
                # seen so far, otherwise n_features_ could be underestimated
                # after a resume.
                max_feat_idx = xp.maximum(max_feat_idx, xp.max(id_batch))
                feat_centered = feat_batch - xp.mean(feat_batch, axis=0)
                feat_orth = _classical_gram_schmidt(feat_centered, W[:, :i], xp)
                include_support = xp.isin(indices_include, id_batch)
                if xp.any(include_support):
                    include_index = xp.nonzero(include_support)[0][
                        0
                    ]  # Get the 1st index
                    best_index = indices_include[include_index].item()
                    batch_index = xp.nonzero(id_batch == best_index)[0][0]
                    best_feat = feat_orth[:, batch_index]
                    r_include = best_feat @ y_transformed
                    best_score = xp.sum(r_include**2)
                    indices_include[include_index] = -1  # Mark as used
                    if checkpoint is not None and not is_replayed:
                        self._save_checkpoint(
                            checkpoint,
                            fingerprints,
                            round=i,
                            batch=n_batches_seen,
                            ids_hash=ids_hasher.copy().hexdigest(),
                            ids_hashes=ids_hashes,
                            y_transformed=y_transformed,
                            W=W,
                            indices=indices,
                            scores=scores,
                            max_feat_idx=max_feat_idx,
                            indices_include=indices_include,
                            best_score=best_score,
                            best_index=best_index,
                            best_feat=best_feat,
                        )
                    break
                # Linear dependence check
                if i > 0:
                    g = feat_orth.T @ W[:, :i]
                    g_max = xp.max(xp.abs(g), axis=1)
                    valid_mask = g_max <= self.tol
                    if not xp.any(valid_mask):
                        if checkpoint is not None and not is_replayed:
                            self._save_checkpoint(
                                checkpoint,
                                fingerprints,
                                round=i,
                                batch=n_batches_seen,
                                ids_hash=ids_hasher.copy().hexdigest(),
                                ids_hashes=ids_hashes,
                                y_transformed=y_transformed,
                                W=W,
                                indices=indices,
                                scores=scores,
                                max_feat_idx=max_feat_idx,
                                indices_include=indices_include,
                                best_score=best_score,
                                best_index=best_index,
                                best_feat=best_feat,
                            )
                        continue
                    valid_indices = xp.nonzero(valid_mask)[0]
                    feat_valid = xp.take(feat_orth, valid_indices, axis=1)
                else:
                    valid_indices = xp.arange(feat_orth.shape[1], device=device_)
                    feat_valid = feat_orth

                r = feat_valid.T @ y_transformed
                score_batch = xp.sum(r * r, axis=1)

                idx = xp.argmax(score_batch)
                score = float(score_batch[idx])

                if score > best_score:
                    best_score = score
                    valid_idx_in_batch = int(valid_indices[int(idx)])
                    best_index = id_batch[valid_idx_in_batch]
                    best_feat = feat_orth[:, valid_idx_in_batch]
                if checkpoint is not None and not is_replayed:
                    self._save_checkpoint(
                        checkpoint,
                        fingerprints,
                        round=i,
                        batch=n_batches_seen,
                        ids_hash=ids_hasher.copy().hexdigest(),
                        ids_hashes=ids_hashes,
                        y_transformed=y_transformed,
                        W=W,
                        indices=indices,
                        scores=scores,
                        max_feat_idx=max_feat_idx,
                        indices_include=indices_include,
                        best_score=best_score,
                        best_index=best_index,
                        best_feat=best_feat,
                    )
            if n_batches_seen < batch_cursor:
                # The generator ended before replaying all batches recorded
                # in the checkpoint: the candidate stream has been truncated.
                raise CheckpointError(
                    f"Refusing to reuse checkpoint {checkpoint.path!r}: the "
                    "feature generator yields fewer candidate batches than "
                    "recorded in the checkpoint. The candidate definition has "
                    "effectively changed. Delete the checkpoint file to start "
                    "a fresh run."
                )
            if best_index is None:
                raise RuntimeError(
                    "No more features can be selected when adding "
                    f"the {i + 1}th feature."
                )
            if best_score == 0:
                raise RuntimeError(
                    f"No improvement can be achieved when adding the {i + 1}th feature."
                )
            indices[i] = best_index
            scores[i] = best_score
            W[:, i] = best_feat
            # The round is complete: persist the new round state so that an
            # interruption between two rounds is recoverable as well.
            if checkpoint is not None:
                self._save_checkpoint(
                    checkpoint,
                    fingerprints,
                    round=i + 1,
                    batch=0,
                    ids_hash=hashlib.sha256(b"").hexdigest(),
                    ids_hashes=[],
                    y_transformed=y_transformed,
                    W=W,
                    indices=indices,
                    scores=scores,
                    max_feat_idx=max_feat_idx,
                    indices_include=indices_include,
                    best_score=-np.inf,
                    best_index=None,
                    best_feat=None,
                )
            # A resumed run must not reuse the in-round cursor of a finished
            # round.
            resume_state = None
        self.indices_ = indices
        self.scores_ = scores
        self.n_features_ = max_feat_idx + 1
        if not xp.all(indices_include == -1):
            raise RuntimeError(
                "Not all prerequisite features can be selected. "
                f"Remaining indices: {indices_include[indices_include != -1]}"
            )
        if checkpoint is not None:
            checkpoint.clear()
        return self

    def _make_fingerprints(self, X, y, sample_mask, feature_generator, xp):
        """Build the fingerprints that bind a checkpoint to one specific run."""
        data_hasher = hashlib.sha256()
        data_hasher.update(_array_fingerprint(X).encode("utf-8"))
        data_hasher.update(_array_fingerprint(y).encode("utf-8"))
        data_hasher.update(_array_fingerprint(sample_mask).encode("utf-8"))

        params_hasher = hashlib.sha256()
        params_hasher.update(repr(self.n_features_to_select).encode("utf-8"))
        params_hasher.update(repr(float(self.tol)).encode("utf-8"))
        params_hasher.update(_array_fingerprint(self.indices_include_).encode("utf-8"))

        return {
            "data": data_hasher.hexdigest(),
            "generator": _generator_fingerprint(feature_generator),
            "params": params_hasher.hexdigest(),
        }

    @staticmethod
    def _load_checkpoint(checkpoint, fingerprints):
        """Load and validate a checkpoint for the current run.

        Returns ``None`` when the checkpoint path does not exist.
        """
        state = checkpoint.load()
        checkpoint.verify(
            state,
            data_fingerprint=fingerprints["data"],
            generator_fingerprint=fingerprints["generator"],
            params_fingerprint=fingerprints["params"],
        )
        return state

    @staticmethod
    def _save_checkpoint(
        checkpoint,
        fingerprints,
        *,
        round,
        batch,
        ids_hash,
        ids_hashes,
        y_transformed,
        W,
        indices,
        scores,
        max_feat_idx,
        indices_include,
        best_score,
        best_index,
        best_feat,
    ):
        """Persist the current round state.

        Only the state of fully processed work is stored. An interruption
        while writing leaves the previous valid checkpoint in place thanks to
        the atomic replace inside :meth:`Checkpoint.save`.
        """
        from ._checkpoint import _CHECKPOINT_VERSION

        checkpoint.save(
            {
                "version": _CHECKPOINT_VERSION,
                "fingerprints": dict(fingerprints),
                "round": round,
                "batch": batch,
                "ids_hash": ids_hash,
                "ids_hashes": list(ids_hashes),
                "y_transformed": _to_cpu_numpy(y_transformed),
                "W": _to_cpu_numpy(W),
                "indices": _to_cpu_numpy(indices),
                "scores": _to_cpu_numpy(scores),
                "max_feat_idx": int(_to_cpu_numpy(max_feat_idx)),
                "indices_include": _to_cpu_numpy(indices_include),
                "best_score": float(best_score),
                "best_index": -1 if best_index is None else int(best_index),
                "best_feat": None if best_feat is None else _to_cpu_numpy(best_feat),
            }
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.array_api_support = True
        return tags


def _to_cpu_numpy(array):
    """Move an array-API array to a host-side numpy array for pickling."""
    if hasattr(array, "cpu"):
        array = array.cpu()
    return np.asarray(array)
