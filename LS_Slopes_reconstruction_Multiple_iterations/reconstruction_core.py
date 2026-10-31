"""Ordinary Point-M2AE inference and first-prediction selection per patch."""
from __future__ import annotations

import numpy as np

from point2mae_reconstruction import predict_masked_patches, restore_coordinates
from representative_cloud import average_prediction_sets, voxel_average


def validate_run_counts(iterations, passes_per_iteration, seed):
    for name, value, maximum in (
        ('iterations', iterations, np.iinfo(np.uint32).max),
        ('passes_per_iteration', passes_per_iteration, np.iinfo(np.uint16).max),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f'{name} must be a positive integer no greater than {maximum}')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')


def reconstruct_iterations(model, points, *, iterations=1, passes_per_iteration=8, seed=42):
    """Run the reference notebook's forward path with independent iteration seeds.

    IDs refer to stable middle-level hierarchy centers, not occupied grid cells.
    Each pass returns a variable number of masked K-point patches. Patches already
    include their model-coordinate center offsets. Coverage is measured, never
    forced; the upstream implementation always exposes middle-level center zero.
    """
    import torch

    validate_run_counts(iterations, passes_per_iteration, seed)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8 or not np.isfinite(points).all():
        raise ValueError('Expected at least eight finite XYZ input points')
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('Point-M2AE inference requires a CUDA model')
    tensor = torch.as_tensor(np.ascontiguousarray(points), device=device).unsqueeze(0)
    prior_mode, numpy_state = model.training, np.random.get_state()
    records, coverage = [], []
    reference_centers = None
    iteration_seeds = np.array([(seed + i) % 2**32 for i in range(iterations)], dtype=np.uint32)
    model.eval()
    try:
        with torch.cuda.device(device), torch.random.fork_rng(devices=[device.index]), torch.inference_mode():
            for iteration, iteration_seed in enumerate(iteration_seeds, 1):
                np.random.seed(int(iteration_seed))
                torch.random.default_generator.manual_seed(int(iteration_seed))
                torch.cuda.manual_seed(int(iteration_seed))
                counts = None
                for pass_id in range(1, passes_per_iteration + 1):
                    prediction = predict_masked_patches(model, tensor)
                    centers = prediction['centers'].cpu().numpy()
                    if reference_centers is None:
                        reference_centers = centers.copy()
                    elif not np.array_equal(reference_centers, centers):
                        raise RuntimeError('Grouping centers changed; patch IDs cannot be combined safely')
                    if counts is None:
                        counts = np.zeros(len(centers), dtype=np.uint32)
                    ids = prediction['patch_ids'].cpu().numpy()
                    counts[ids] += 1
                    records.append({
                        'iteration_id': iteration, 'pass_id': pass_id,
                        'patch_ids': ids.copy(), 'patches': prediction['patches'].cpu().numpy().copy(),
                        'loss': prediction['loss'],
                    })
                coverage.append(counts)
                print(f'  Iteration {iteration}/{iterations}: {(counts > 0).sum()}/{len(counts)} '
                      f'middle patches ({(counts > 0).mean():.1%}), {passes_per_iteration} passes', flush=True)
    finally:
        np.random.set_state(numpy_state)
        model.train(prior_mode)
    return {'passes': records, 'centers_model': reference_centers,
            'coverage_counts': np.stack(coverage), 'iteration_seeds': iteration_seeds}


def retain_first_predictions(result):
    """Keep each patch's earliest K-point set across all passes and iterations.

    Keep original mask IDs and coverage as inference diagnostics. Empty retained
    passes are valid. Distinct patches may still have spatially overlapping points.
    """
    covered, records = set(), []
    for record in result['passes']:
        ids = record['patch_ids']
        keep = np.array([int(patch_id) not in covered for patch_id in ids], dtype=bool)
        covered.update(map(int, ids))
        records.append({**record, 'masked_ids': record.get('masked_ids', ids).copy(),
                        'patch_ids': ids[keep].copy(), 'patches': record['patches'][keep].copy()})
    return {**result, 'passes': records}


def collect_predictions(result, centroid, scale, representative_method, voxel_size):
    """Restore retained predictions once, preserving their earliest provenance."""
    records = [record for record in result['passes'] if len(record['patch_ids'])]
    if not records:
        raise ValueError('No masked predictions were produced')
    xyz = np.concatenate([restore_coordinates(record['patches'].reshape(-1, 3), centroid, scale)
                          for record in records])
    patch_ids = np.concatenate([np.repeat(record['patch_ids'], record['patches'].shape[1])
                                for record in records]).astype(np.uint32)
    clouds = {'all_predicted_xyz': xyz, 'patch_ids': patch_ids}
    for name, dtype in (('iteration', np.uint32), ('pass', np.uint16)):
        clouds[name + '_ids'] = np.concatenate([
            np.full(record['patches'].shape[0] * record['patches'].shape[1], record[name + '_id'], dtype=dtype)
            for record in records])
    if representative_method == 'voxel':
        representative, counts = voxel_average(xyz, voxel_size)
        representative_ids = np.empty(0, dtype=np.uint32)
    else:
        patches, counts, ids = average_prediction_sets(
            [record['patches'] for record in records], [record['patch_ids'] for record in records],
            len(result['centers_model']), representative_method)
        representative = restore_coordinates(patches.reshape(-1, 3), centroid, scale)
        representative_ids = np.repeat(ids, patches.shape[1]).astype(np.uint32)
        counts = counts.reshape(-1)
    clouds.update(representative_xyz=representative, representative_counts=counts,
                  representative_patch_ids=representative_ids)
    if int(counts.sum(dtype=np.uint64)) != len(xyz):
        raise RuntimeError('Representative support counts do not account for every prediction')
    return clouds
