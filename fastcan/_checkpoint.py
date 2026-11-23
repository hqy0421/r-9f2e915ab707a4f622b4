"""
Resumable round state (checkpoint) for :class:`~fastcan.LazyFastCan`.

A checkpoint makes a LazyFastCan run resumable across interruptions. The
selection proceeds in *rounds* (one round per selected feature) and each round
consumes the candidate generator batch by batch. A checkpoint is written after
every fully processed batch, so an interrupted job loses at most the work of
one batch.

A checkpoint can only be reused by a run whose inputs are identical to those
of the run that created it. The checkpoint stores fingerprints of

* the feature matrix ``X`` and the target matrix ``y``,
* the candidate definition, i.e. the feature generator together with the
  arguments bound to it,
* the estimator parameters (``n_features_to_select``, ``indices_include``,
  ``sample_mask`` and ``tol``), and
* the stream of candidate ids actually yielded by the generator.

If any of these changes, resumption is explicitly refused with a
:class:`CheckpointError`. The candidate-id stream check additionally detects
generators whose definition looks identical but whose yielded ids have changed
(e.g. a generator depending on an external file or a random state).
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

import hashlib
import os
import pickle
import tempfile
from functools import partial

import numpy as np

# Bumped whenever the on-disk layout of the checkpoint changes.
_CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be loaded or safely reused."""


def _to_numpy(obj):
    """Best-effort conversion of an array-API object to a numpy array."""
    if hasattr(obj, "cpu"):
        # Torch and other device-backed arrays.
        obj = obj.cpu()
    return np.asarray(obj)


def _array_fingerprint(obj):
    """Content fingerprint of an array-like object.

    The fingerprint covers shape, dtype and the raw bytes, so any change to
    the data (including a single element) changes the fingerprint.
    """
    arr = _to_numpy(obj)
    hasher = hashlib.sha256()
    hasher.update(b"ndarray")
    hasher.update(repr(arr.shape).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _fingerprint_value(obj):
    """Fingerprint an arbitrary value used in a generator definition."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return ("builtin", repr(obj))
    if isinstance(obj, partial):
        return (
            "partial",
            _fingerprint_value(obj.func),
            [_fingerprint_value(a) for a in obj.args],
            sorted((k, _fingerprint_value(v)) for k, v in obj.keywords.items()),
        )
    if hasattr(obj, "__array__"):
        return ("ndarray", _array_fingerprint(obj))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, [_fingerprint_value(v) for v in obj])
    if isinstance(obj, dict):
        return (
            "dict",
            sorted((repr(k), _fingerprint_value(v)) for k, v in obj.items()),
        )
    if callable(obj):
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        if module is None or qualname is None or "<" in qualname:
            raise CheckpointError(
                "Cannot fingerprint the feature generator "
                f"{obj!r}: it is a locally defined or anonymous callable. "
                "Define the generator at module level (or wrap it with "
                "functools.partial) so that a resumable run can verify "
                "that the candidate definition is unchanged."
            )
        return ("callable", module, qualname)
    raise CheckpointError(
        f"Cannot fingerprint the value {obj!r} of type {type(obj).__name__} "
        "used by the feature generator. Only built-in types, numpy arrays, "
        "module-level callables, functools.partial objects and containers "
        "thereof are supported in resumable runs."
    )


def _generator_fingerprint(feature_generator):
    """Fingerprint the candidate definition (generator + bound arguments)."""
    hasher = hashlib.sha256()
    hasher.update(pickle.dumps(_fingerprint_value(feature_generator), protocol=5))
    return hasher.hexdigest()


def _ids_fingerprint(ids):
    """Fingerprint one batch of candidate ids."""
    ids_np = _to_numpy(ids).astype(np.int64, copy=False)
    hasher = hashlib.sha256()
    hasher.update(repr(ids_np.shape).encode("utf-8"))
    hasher.update(ids_np.tobytes())
    return hasher.digest()


class Checkpoint:
    """Persistent round state for resumable :class:`~fastcan.LazyFastCan` fits.

    Parameters
    ----------
    path : str or os.PathLike
        Path of the checkpoint file. The file is created on the first
        :meth:`save` and updated atomically after every processed candidate
        batch. It is removed automatically when the fit finishes successfully.

    Attributes
    ----------
    path : str
        The checkpoint path.

    round_ : int or None
        The zero-based round (selection step) stored in the checkpoint, i.e.
        the number of already completed rounds. ``None`` if no checkpoint has
        been written yet.

    batch_ : int or None
        The number of candidate batches already processed within the current
        round. ``None`` if no checkpoint has been written yet.

    Examples
    --------
    >>> from pathlib import Path
    >>> from functools import partial
    >>> from fastcan import LazyFastCan, Checkpoint
    >>> from fastcan.narx import gen_poly_features, make_poly_ids
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> y = [[0], [1], [1], [2]]
    >>> ids = make_poly_ids(2, degree=2)
    >>> gen = partial(gen_poly_features, ids=ids, batch_size=2)
    >>> checkpoint = Checkpoint(Path("lazyfastcan.ckpt"))
    >>> selector = LazyFastCan(2, feature_generator=gen, checkpoint=checkpoint)
    >>> selector.fit(X, y)  # doctest: +SKIP
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self.round_ = None
        self.batch_ = None

    def __repr__(self):
        return (
            f"Checkpoint(path={self.path!r}, round_={self.round_!r}, "
            f"batch_={self.batch_!r})"
        )

    def exists(self):
        """Return whether a checkpoint file exists at :attr:`path`."""
        return os.path.exists(self.path)

    def load(self):
        """Load the checkpoint state.

        Returns
        -------
        state : dict
            The stored state dictionary.

        Raises
        ------
        CheckpointError
            If the file is missing, unreadable or uses an incompatible
            checkpoint format.
        """
        try:
            with open(self.path, "rb") as f:
                state = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError) as exc:
            raise CheckpointError(
                f"Cannot read checkpoint {self.path!r}: {exc}. Delete the "
                "file to start a fresh run."
            ) from exc
        if not isinstance(state, dict) or state.get("version") != _CHECKPOINT_VERSION:
            raise CheckpointError(
                f"Checkpoint {self.path!r} is corrupted or was written by an "
                "incompatible version of fastcan. Delete the file to start a "
                "fresh run."
            )
        self.round_ = state["round"]
        self.batch_ = state["batch"]
        return state

    def save(self, state):
        """Atomically write the checkpoint state to :attr:`path`."""
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".lazyfastcan-ckpt-", dir=directory, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(state, f, protocol=5)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        self.round_ = state["round"]
        self.batch_ = state["batch"]

    def clear(self):
        """Remove the checkpoint file if it exists."""
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        self.round_ = None
        self.batch_ = None

    def verify(
        self,
        state,
        *,
        data_fingerprint,
        generator_fingerprint,
        params_fingerprint,
    ):
        """Verify that a loaded checkpoint matches the current run.

        Parameters
        ----------
        state : dict
            State returned by :meth:`load`.
        data_fingerprint : str
            Fingerprint of the current ``X`` and ``y``.
        generator_fingerprint : str
            Fingerprint of the current candidate definition.
        params_fingerprint : str
            Fingerprint of the current estimator parameters.

        Raises
        ------
        CheckpointError
            If any fingerprint differs, meaning the data, candidate definition
            or estimator parameters have changed since the checkpoint was
            written.
        """
        checks = (
            (
                "data",
                data_fingerprint,
                "the feature matrix X or the target matrix y has changed",
            ),
            (
                "generator",
                generator_fingerprint,
                "the feature generator or its candidate definition has changed",
            ),
            ("params", params_fingerprint, "the estimator parameters have changed"),
        )
        for key, current, reason in checks:
            if state["fingerprints"][key] != current:
                raise CheckpointError(
                    f"Refusing to reuse checkpoint {self.path!r}: {reason}. "
                    "Delete the checkpoint file (or pass checkpoint=None) to "
                    "start a fresh run."
                )
