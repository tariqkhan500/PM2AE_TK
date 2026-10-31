"""Summarize predicted-only point clouds by mean XYZ in occupied voxels."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment


def voxel_average(xyz, voxel_size) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 XYZ means and uint32 counts using cloud-minimum voxels.

    Each voxel is a cube of ``voxel_size`` in the source coordinate units.
    Input points must already be decoder predictions in global coordinates.
    """
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError('xyz must be a nonempty N x 3 array')
    if not np.isfinite(points).all():
        raise ValueError('xyz must contain only finite coordinates')
    if isinstance(voxel_size, bool) or not isinstance(voxel_size, Real) or not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('voxel_size must be a positive finite real number')

    origin = points.min(axis=0)
    offsets = points - origin
    scaled = offsets / float(voxel_size)
    if not np.isfinite(scaled).all() or np.any(scaled >= float(2**63)):
        raise ValueError('voxel_size is too small for the cloud coordinate range')
    keys = np.floor(scaled).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > np.iinfo(np.uint32).max):
        raise ValueError('A voxel contains too many predictions for uint32 counts')
    means = np.column_stack([
        origin[axis] + np.bincount(inverse, weights=offsets[:, axis], minlength=len(counts)) / counts
        for axis in range(3)
    ])
    if not np.isfinite(means).all():
        raise ValueError('Voxel means must be finite')
    return means.astype(np.float64, copy=False), counts.astype(np.uint32, copy=False)


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise XYZ costs, rejecting a finite-input range that overflows."""
    difference = a[:, None, :] - b[None, :, :]
    costs = np.sum(difference * difference, axis=2)
    if not np.isfinite(costs).all():
        raise ValueError('Predicted coordinates are too large for correspondence distances')
    return costs


def _nn_average(sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    anchors = sets[0]
    sums = anchors.copy()
    counts = np.ones(len(anchors), dtype=np.uint64)
    for points in sets[1:]:
        nearest = np.argmin(_squared_distances(points, anchors), axis=1)
        np.add.at(sums, nearest, points)
        np.add.at(counts, nearest, 1)
    return sums / counts[:, None], counts


def _ot_average(sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    template = sets[0].copy()
    previous_assignments = None
    for _ in range(5):
        updated = np.zeros_like(template)
        assignments = []
        for points in sets:
            rows, columns = linear_sum_assignment(_squared_distances(template, points))
            assignment = np.empty(len(template), dtype=np.intp)
            assignment[rows] = columns
            updated += points[assignment] / len(sets)
            assignments.append(assignment)
        template = updated
        if previous_assignments is not None and all(
            np.array_equal(current, previous)
            for current, previous in zip(assignments, previous_assignments)
        ):
            break
        previous_assignments = assignments
    return template, np.full(len(template), len(sets), dtype=np.uint64)


def patch_correspondence_average(
    pass_predictions, masked_ids, patch_count: int, method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Match unordered predictions within each patch and average all sets.

    Inputs are decoder predictions [iterations, passes, masked, K, 3] and
    matching patch IDs [iterations, passes, masked]. Every iteration must
    cover every patch. NN assigns points to fixed anchors from the first set;
    OT uses equal-mass one-to-one assignments against an evolving template.
    """
    if method not in ('nn', 'ot'):
        raise ValueError("method must be 'nn' or 'ot'")
    if isinstance(patch_count, bool) or not isinstance(patch_count, Integral) or patch_count < 1:
        raise ValueError('patch_count must be a positive integer')
    predictions = np.asarray(pass_predictions, dtype=np.float64)
    ids = np.asarray(masked_ids)
    if (
        predictions.ndim != 5 or predictions.shape[-1] != 3
        or any(size < 1 for size in predictions.shape[:4])
    ):
        raise ValueError('pass_predictions must have shape [I,R,M,K,3] with positive dimensions')
    if ids.shape != predictions.shape[:3] or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('masked_ids must be an integer array of shape [I,R,M]')
    if np.any(ids < 0) or np.any(ids >= patch_count):
        raise ValueError('masked_ids must refer to occupied patches')
    if not np.isfinite(predictions).all():
        raise ValueError('pass_predictions must contain only finite coordinates')
    for iteration_ids in ids:
        if np.unique(iteration_ids).size != patch_count:
            raise ValueError('Every iteration must predict every occupied patch')

    k = predictions.shape[-2]
    sets_by_patch = [[] for _ in range(patch_count)]
    for patch_id, points in zip(ids.flat, predictions.reshape(-1, k, 3)):
        sets_by_patch[int(patch_id)].append(points)

    averaged = np.empty((patch_count, k, 3), dtype=np.float64)
    support = np.empty((patch_count, k), dtype=np.uint32)
    for patch_id, sets in enumerate(sets_by_patch):
        mean, counts = _nn_average(sets) if method == 'nn' else _ot_average(sets)
        if not np.isfinite(mean).all() or np.any(counts > np.iinfo(np.uint32).max):
            raise ValueError('Representative points or support counts exceed export range')
        averaged[patch_id] = mean
        support[patch_id] = counts
    return averaged, support


def average_prediction_sets(pass_predictions, masked_ids, patch_count, method):
    """Average ragged Point-M2AE passes, returning only covered patch IDs.

    Patches contain cloud-relative XYZ, with patch center offsets already added.
    All prediction sets participate, including repeats within an iteration.
    """
    if method not in ('nn', 'ot'):
        raise ValueError("method must be 'nn' or 'ot'")
    if len(pass_predictions) != len(masked_ids) or not len(pass_predictions):
        raise ValueError('Each prediction pass needs matching patch IDs')
    sets_by_patch = {}
    k = None
    for predictions, ids in zip(pass_predictions, masked_ids):
        predictions, ids = np.asarray(predictions, dtype=np.float64), np.asarray(ids)
        if (predictions.ndim != 3 or predictions.shape[-1] != 3 or predictions.shape[1] < 1
                or ids.shape != (len(predictions),) or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids >= patch_count)
                or len(np.unique(ids)) != len(ids) or not np.isfinite(predictions).all()):
            raise ValueError('Expected finite [M,K,3] predictions and unique middle-patch IDs')
        if k is not None and predictions.shape[1] != k:
            raise ValueError('All predicted patches must have the same number of points')
        k = predictions.shape[1]
        for patch_id, points in zip(ids, predictions):
            sets_by_patch.setdefault(int(patch_id), []).append(points)
    covered_ids = np.array(sorted(sets_by_patch), dtype=np.uint32)
    if not len(covered_ids):
        raise ValueError('No masked predictions to average')
    means, support = [], []
    for patch_id in covered_ids:
        sets = sets_by_patch[int(patch_id)]
        mean, counts = _nn_average(sets) if method == 'nn' else _ot_average(sets)
        if not np.isfinite(mean).all() or np.any(counts > np.iinfo(np.uint32).max):
            raise ValueError('Representative values exceed the export range')
        means.append(mean)
        support.append(counts.astype(np.uint32))
    return np.stack(means), np.stack(support), covered_ids
