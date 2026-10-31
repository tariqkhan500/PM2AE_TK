"""Run every notebook cell with a real checkpoint; write only temporary outputs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

FOLDER = Path(__file__).resolve().parents[1]
ROOT = FOLDER.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FOLDER))
prefix = Path(sys.prefix)
os.environ.setdefault('CUDA_HOME', str(prefix))
os.environ.setdefault('CPATH', str(prefix / 'targets/x86_64-linux/include'))
os.environ.setdefault('CC', str(prefix / 'bin/gcc'))
os.environ.setdefault('CXX', str(prefix / 'bin/g++'))
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ.setdefault('MAX_JOBS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['PATH'] = str(prefix / 'bin') + os.pathsep + os.environ['PATH']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if args.input is None:
        with (checkpoint.parent / 'test.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        source = Path(min(rows, key=lambda row: abs(int(row['point_count']) - 2048))['path'])
        # Training manifests may contain the workspace's former absolute path.
        if not source.is_file() and 'Output' in source.parts:
            source = ROOT.joinpath(*source.parts[source.parts.index('Output'):])
    else:
        source = args.input.resolve()
    source_bytes = source.read_bytes()
    with tempfile.TemporaryDirectory(prefix='pointm2ae-slope-notebook-') as temporary:
        os.environ.setdefault('MPLCONFIGDIR', temporary)
        import numpy as np
        import torch
        import laspy
        import reconstruct_slopes as pipeline
        from reconstruction_core import reconstruct_iterations, retain_first_predictions
        from point2mae_reconstruction import prepare_cloud, reconstruct_cloud

        notebook = json.loads((FOLDER / 'multiple_iteration_reconstruction.ipynb').read_text())
        loaded = {}
        real_load = pipeline.load_checkpoint_model
        def capture_load(*a, **kw):
            result = real_load(*a, **kw)
            loaded['model'], loaded['config'], _ = result
            loaded['bn'] = {k: v.clone() for k,v in result[0].state_dict().items()
                            if k.endswith(('running_mean','running_var','num_batches_tracked'))}
            return result
        namespace = {}
        with patch.object(pipeline, 'load_checkpoint_model', side_effect=capture_load):
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code': continue
                print('Executing notebook cell:', cell['id'], flush=True)
                exec(compile(''.join(cell['source']), cell['id'], 'exec'), namespace)
                if cell['id'] == 'settings':
                    namespace.update(INPUT_DIR=source.parent, CHECKPOINT=checkpoint,
                                     OUTPUT_DIR=Path(temporary)/'result', SOURCE_GLOB=source.name,
                                     ITERATIONS=2, PASSES_PER_ITERATION=8, REPRESENTATIVE_METHOD='nn', LIMIT=1)
        manifest = namespace['manifest']
        assert manifest['counts'] == {'selected':1,'reconstructed':1,'failed':0}, manifest['failures']
        output = Path(manifest['output_dir'])
        unit = manifest['units'][0]
        patches, ids, pieces = [], [], []
        for path in sorted((output/'passes').rglob('*.npz')):
            with np.load(path, allow_pickle=False) as saved:
                patches.append(saved['predicted_patches'].copy())
                ids.append(saved['retained_patch_ids'].copy())
                pieces.append(saved['predicted_patches'].astype(np.float64).reshape(-1,3)
                              * saved['cloud_scale'] + saved['cloud_centroid'])
        assert len(pieces) == 16
        retained_ids = np.concatenate(ids)
        assert len(retained_ids) == len(np.unique(retained_ids))
        with np.load(output/unit['combined_file'], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['all_predicted_xyz'], np.concatenate(pieces))
            assert saved['coverage_counts'].shape[0] == 2
            assert (saved['coverage_counts'][:,0] == 0).all()
            assert saved['representative_counts'].sum() == len(saved['all_predicted_xyz'])
            las = laspy.read(output/unit['all_las_file'])
            assert (np.abs(las.xyz-saved['all_predicted_xyz']) <= las.header.scales/2+1e-8).all()
        original = laspy.read(source)
        points, _, _ = prepare_cloud(original.xyz, loaded['config']['normalization'])
        seed = int.from_bytes(hashlib.sha256(f'42:{source.name}'.encode()).digest()[:4], 'little')
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state().clone()
        repeated = reconstruct_iterations(loaded['model'], points, iterations=2, passes_per_iteration=8, seed=seed)
        retained = retain_first_predictions(repeated)
        for record, expected, expected_ids in zip(retained['passes'], patches, ids):
            np.testing.assert_array_equal(record['patches'], expected)
            np.testing.assert_array_equal(record['patch_ids'], expected_ids)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        assert np.random.get_state()[2:] == numpy_state[2:]
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_state, rtol=0, atol=0)
        # Deduplicating our first iteration must reproduce the reference notebook.
        first, covered = [], set()
        for record in repeated['passes'][:8]:
            keep = np.array([int(i) not in covered for i in record['patch_ids']])
            if keep.any(): first.append(record['patches'][keep].reshape(-1,3))
            covered.update(map(int, record['patch_ids']))
        reference, _ = reconstruct_cloud(loaded['model'], points, passes=8, seed=seed)
        np.testing.assert_array_equal(np.concatenate(first), reference)
        for k,v in loaded['bn'].items(): torch.testing.assert_close(loaded['model'].state_dict()[k], v, rtol=0, atol=0)
        for module in (loaded['model'].group_dividers[1], loaded['model'].h_encoder, loaded['model'].rec_head):
            assert not module._forward_hooks
        assert source.read_bytes() == source_bytes
        print(json.dumps({'source':str(source), 'iterations':2, 'passes_per_iteration':8,
                          'coverage':unit['coverage_fraction'], 'coverage_by_iteration':unit['coverage_by_iteration'],
                          'predicted_points':unit['all_predicted_points'],
                          'representative_points':unit['representative_points'],
                          'reference_match':True, 'repeatable':True}, indent=2))
        print('CUDA notebook verification passed; temporary outputs removed on exit.')


if __name__ == '__main__': main()
