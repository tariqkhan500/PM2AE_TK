"""Batch slope-unit reconstruction with the Point2MAE notebook's CUDA model."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import uuid

import laspy
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from point2mae_reconstruction import load_checkpoint_model, prepare_cloud
from point_cloud_io import write_point_clouds
from reconstruction_core import collect_predictions, reconstruct_iterations, retain_first_predictions, validate_run_counts

DEFAULT_INPUT = PROJECT_ROOT / 'Output/full_20261006_001322_611477/LS_head_scarp_Slope_units'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'training_runs/pointm2ae_xy_20261030T190853Z_d323ddc4/best.pt'
DEFAULT_MODEL_ROOT = PROJECT_ROOT / 'Point-M2AE-main'
SUMMARY_FIELDS = [
    'source_file', 'SU_id', 'input_points', 'patch_count', 'covered_patches',
    'iterations', 'passes_per_iteration', 'total_passes', 'coverage_fraction',
    'coverage_by_iteration', 'mean_masked_chamfer_model_units', 'representative_method',
    'all_predicted_points', 'representative_points', 'combined_file',
    'all_las_file', 'all_ply_file', 'representative_las_file', 'representative_ply_file', 'seconds',
]


def _write_manifest(path, manifest):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _save_unit_outputs(output_root, relative, header, result, clouds, centroid, scale,
                       su_id, method, voxel_size):
    """Stage all artifacts; publish exclusively and roll back a failed unit."""
    paths = {
        'combined_file': Path('combined') / relative.parent / (relative.stem + '_reconstructed.npz'),
        'all_las_file': Path('point_clouds') / relative.parent / (relative.stem + '_first_predictions.las'),
        'representative_las_file': Path('point_clouds') / relative.parent / (relative.stem + '_representative.las'),
    }
    paths['all_ply_file'] = paths['all_las_file'].with_suffix('.ply')
    paths['representative_ply_file'] = paths['representative_las_file'].with_suffix('.ply')
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix='.pending_', dir=output_root) as directory:
            staged = Path(directory)
            combined = staged / paths['combined_file']
            combined.parent.mkdir(parents=True, exist_ok=True)
            arrays = {
                **clouds, 'centers_model': result['centers_model'],
                'cloud_centroid': centroid, 'cloud_scale': np.float64(scale),
                'coverage_counts': result['coverage_counts'], 'iteration_seeds': result['iteration_seeds'],
                'loss_per_pass': np.array([r['loss'] for r in result['passes']]),
                'masked_patch_counts': np.array([len(r['masked_ids']) for r in result['passes']]),
                'retained_patch_counts': np.array([len(r['patch_ids']) for r in result['passes']]),
                'representative_method': np.array(method), 'prediction_selection': np.array('first_per_patch'),
                'source_file': np.array(relative.as_posix()), 'SU_id': np.uint32(su_id),
            }
            if voxel_size is not None:
                arrays['voxel_size'] = np.float64(voxel_size)
            np.savez_compressed(combined, **arrays)
            artifacts = [paths['combined_file']]
            for record in result['passes']:
                iteration, pass_id = record['iteration_id'], record['pass_id']
                pass_relative = Path('passes') / relative.parent / (
                    f'{relative.stem}_iteration_{iteration:02d}_pass_{pass_id:02d}.npz')
                destination = staged / pass_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                mask = np.zeros(len(result['centers_model']), dtype=bool)
                mask[record['masked_ids']] = True
                np.savez_compressed(
                    destination, predicted_patches=record['patches'], masked_ids=record['masked_ids'],
                    retained_patch_ids=record['patch_ids'],
                    mae_mask=mask, centers_model=result['centers_model'], cloud_centroid=centroid,
                    cloud_scale=np.float64(scale), iteration_id=np.uint32(iteration),
                    reconstruction_pass=np.uint16(pass_id), seed=result['iteration_seeds'][iteration-1],
                    loss=np.float64(record['loss']),
                    coordinate_space=np.array('model coordinates with patch centers already added'),
                )
                artifacts.append(pass_relative)
            write_point_clouds(clouds['all_predicted_xyz'], header,
                               staged / paths['all_las_file'], staged / paths['all_ply_file'], su_id,
                               clouds['patch_ids'], clouds['pass_ids'], iteration_ids=clouds['iteration_ids'])
            options = ({'voxel_counts': clouds['representative_counts']} if method == 'voxel' else
                       {'patch_ids': clouds['representative_patch_ids'], 'support_counts': clouds['representative_counts']})
            write_point_clouds(clouds['representative_xyz'], header,
                               staged / paths['representative_las_file'], staged / paths['representative_ply_file'],
                               su_id, **options)
            artifacts.extend(paths[key] for key in ('all_las_file', 'all_ply_file',
                                                    'representative_las_file', 'representative_ply_file'))
            for artifact in artifacts:
                destination = output_root / artifact
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(staged / artifact, destination)
                published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise
    return {key: value.as_posix() for key, value in paths.items()}


def run_pipeline(input_dir, checkpoint_path, output_dir, *, iterations=1, passes_per_iteration=8,
                 seed=42, cuda_device=0, model_root=DEFAULT_MODEL_ROOT, limit=1,
                 source_glob='*.las', representative_method='nn', voxel_size=None,
                 mask_ratio=None):
    """Infer every selected LAS/LAZ keeping first patch predictions and measuring actual coverage."""
    source_root, checkpoint_path, output_root = [Path(p).expanduser().resolve()
                                                for p in (input_dir, checkpoint_path, output_dir)]
    if output_root.exists():
        raise FileExistsError(f'Choose a new output directory: {output_root}')
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    validate_run_counts(iterations, passes_per_iteration, seed)
    if type(cuda_device) is not int or cuda_device < 0:
        raise ValueError('cuda_device must be a nonnegative integer')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer or None')
    if not isinstance(representative_method, str) or representative_method.strip().lower() not in ('voxel','nn','ot'):
        raise ValueError('representative_method must be voxel, nn, or ot')
    method = representative_method.strip().lower()
    if method != 'voxel' and voxel_size is not None:
        raise ValueError('voxel_size applies only to voxel averaging')
    if method == 'voxel':
        voxel_size = .5 if voxel_size is None else voxel_size
        if isinstance(voxel_size, bool) or not isinstance(voxel_size, (int,float)) or not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError('voxel_size must be finite and positive in source coordinate units')
    files = sorted((p for p in source_root.rglob('*') if p.is_file() and p.suffix.lower() in ('.las','.laz')
                    and p.relative_to(source_root).match(source_glob)),
                   key=lambda p: p.relative_to(source_root).as_posix())
    files = files if limit is None else files[:limit]
    if not files:
        raise ValueError(f'No LAS/LAZ files match {source_glob!r} in {source_root}')
    model, run_config, checkpoint_info = load_checkpoint_model(
        checkpoint_path, model_root, cuda_device, mask_ratio=mask_ratio)
    manifest = {
        'format_version': 6, 'status': 'running', 'model': 'Point-M2AE',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'input_dir': str(source_root), 'output_dir': str(output_root),
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': _sha256(checkpoint_path),
        'checkpoint_info': checkpoint_info, 'run_config': run_config,
        'iterations': iterations, 'passes_per_iteration': passes_per_iteration,
        'seed': seed, 'cuda_device': cuda_device,
        'representative_method': method, 'voxel_size': voxel_size, 'prediction_selection': 'first_per_patch',
        'merge_method': 'first masked decoder prediction per middle-level patch across all passes and iterations',
        'iteration_definition': 'fixed number of ordinary random masking passes; not a complete coverage sweep',
        'coverage_definition': 'fraction of middle-level hierarchy patches predicted; not raw-point or grid coverage',
        'coverage_limitation': 'upstream masking always exposes middle-level center zero; full coverage is not guaranteed',
        'output_coordinates': 'source coordinates, restored using saved normalization',
        'original_points_in_exports': False, 'training_performed': False,
        'selection': {'source_glob': source_glob, 'limit': limit},
        'counts': {'selected': len(files), 'reconstructed': 0, 'failed': 0}, 'units': [], 'failures': [],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    _write_manifest(output_root / 'manifest.json', manifest)
    start = time.perf_counter()
    print(f'Point-M2AE: {len(files)} units, {iterations} iterations x {passes_per_iteration} passes; '
          f'first prediction per patch; {method} representative', flush=True)
    try:
        with (output_root / 'summary.csv').open('w', newline='', encoding='utf-8') as summary_stream, \
             (output_root / 'failures.csv').open('w', newline='', encoding='utf-8') as failure_stream:
            summary = csv.DictWriter(summary_stream, fieldnames=SUMMARY_FIELDS)
            failures = csv.DictWriter(failure_stream, fieldnames=['source_file','reason'])
            summary.writeheader()
            failures.writeheader()
            for source in files:
                relative = source.relative_to(source_root)
                unit_start = time.perf_counter()
                print(f'Reconstructing {relative}', flush=True)
                try:
                    match = re.fullmatch(r'SU_(\d+)', source.stem, re.IGNORECASE)
                    if match is None:
                        raise ValueError('Expected SU_<numeric_id>.las or .laz filename')
                    su_id = int(match.group(1))
                    source_las = laspy.read(source)
                    original = np.asarray(source_las.xyz, dtype=np.float64)
                    points, centroid, scale = prepare_cloud(original, run_config['normalization'])
                    # Stable across file selection/order; constrained to NumPy's seed range.
                    unit_seed = int.from_bytes(hashlib.sha256(f'{seed}:{relative.as_posix()}'.encode()).digest()[:4], 'little')
                    result = reconstruct_iterations(model, points, iterations=iterations,
                                                    passes_per_iteration=passes_per_iteration, seed=unit_seed)
                    result = retain_first_predictions(result)
                    clouds = collect_predictions(result, centroid, scale, method, voxel_size)
                    paths = _save_unit_outputs(output_root, relative, source_las.header, result, clouds,
                                               centroid, scale, su_id, method, voxel_size)
                    coverage = result['coverage_counts'] > 0
                    record = {
                        'source_file': relative.as_posix(), 'SU_id': su_id, 'input_points': len(original),
                        'patch_count': coverage.shape[1], 'covered_patches': int(coverage.any(axis=0).sum()),
                        'iterations': iterations, 'passes_per_iteration': passes_per_iteration,
                        'total_passes': len(result['passes']), 'coverage_fraction': float(coverage.any(axis=0).mean()),
                        'coverage_by_iteration': coverage.mean(axis=1).tolist(),
                        'mean_masked_chamfer_model_units': float(np.mean([r['loss'] for r in result['passes']])),
                        'representative_method': method, 'all_predicted_points': len(clouds['all_predicted_xyz']),
                        'representative_points': len(clouds['representative_xyz']), **paths,
                        'seconds': round(time.perf_counter() - unit_start, 3),
                    }
                    summary.writerow({**record, 'coverage_by_iteration': json.dumps(record['coverage_by_iteration'])})
                    manifest['units'].append(record)
                    manifest['counts']['reconstructed'] += 1
                except MemoryError:
                    raise
                except Exception as exc:
                    failure = {'source_file': relative.as_posix(), 'reason': f'{type(exc).__name__}: {exc}'}
                    failures.writerow(failure)
                    manifest['failures'].append(failure)
                    manifest['counts']['failed'] += 1
                    print(f'FAILED {relative}: {failure["reason"]}', flush=True)
                summary_stream.flush()
                failure_stream.flush()
                _write_manifest(output_root / 'manifest.json', manifest)
        manifest['status'] = 'complete' if not manifest['counts']['failed'] else 'completed_with_errors'
    except BaseException as exc:
        manifest['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['elapsed_seconds'] = round(time.perf_counter() - start, 3)
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        _write_manifest(output_root / 'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--model-root', type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--passes-per-iteration', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda-device', type=int, default=0)
    parser.add_argument('--limit', type=int, default=1, help='Default: 1; use --all-units for every match')
    parser.add_argument('--all-units', action='store_true')
    parser.add_argument('--source-glob', default='*.las')
    parser.add_argument('--representative-method', choices=('voxel','nn','ot'), default='nn')
    parser.add_argument('--voxel-size', type=float, help='Source units; voxel default: 0.5')
    args = vars(parser.parse_args(argv))
    if args.pop('all_units'):
        args['limit'] = None
    if args['output_dir'] is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        args['output_dir'] = HERE / 'results' / f'pointm2ae_{stamp}_{uuid.uuid4().hex[:8]}'
    args['checkpoint_path'] = args.pop('checkpoint')
    try:
        manifest = run_pipeline(**args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print('Results:', manifest['output_dir'])
    return 0 if not manifest['counts']['failed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
