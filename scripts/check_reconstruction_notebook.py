"""Execute reconstruction notebook against a trained checkpoint; outputs are temporary."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
prefix = Path(sys.prefix)
os.environ.setdefault('CUDA_HOME', str(prefix))
os.environ.setdefault('CPATH', str(prefix / 'targets/x86_64-linux/include'))
os.environ.setdefault('CC', '/usr/bin/gcc')
os.environ.setdefault('CXX', '/usr/bin/g++')
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
        with (checkpoint.parent / 'test.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        # A held-out moderate-size real cloud keeps this a short inference check.
        source = Path(min(rows, key=lambda row: abs(int(row['point_count']) - 2048))['path'])
    else:
        source = args.input.resolve()
    original_source_stat = source.stat()
    notebook = json.loads((ROOT / 'Point2MAE_reconstruction.ipynb').read_text())
    cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}
    with tempfile.TemporaryDirectory(prefix='pointm2ae-reconstruction-check-') as temporary:
        os.environ.setdefault('MPLCONFIGDIR', temporary)
        output = Path(temporary) / 'prediction.las'
        settings = cells['settings'].replace("MODEL_PATH = ''", f'MODEL_PATH = {str(checkpoint)!r}')
        settings = settings.replace("INPUT_LAS = ''", f'INPUT_LAS = {str(source)!r}')
        settings = settings.replace("OUTPUT_LAS = ''", f'OUTPUT_LAS = {str(output)!r}')
        ns = {}
        exec(compile(settings, 'settings', 'exec'), ns)
        for name in ('load', 'reconstruct', 'comparison', 'export'):
            print('Checking notebook cell:', name, flush=True)
            exec(compile(cells[name], name, 'exec'), ns)

        import numpy as np
        import torch
        import laspy
        from point2mae_reconstruction import predict_masked_patches, reconstruct_cloud

        model = ns['model']
        bn_before = {k: v.clone() for k, v in model.state_dict().items()
                     if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        original_numpy_state = np.random.get_state()
        repeated, report = reconstruct_cloud(model, ns['model_points'], passes=ns['NUM_PASSES'], seed=ns['SEED'])
        np.testing.assert_array_equal(repeated, ns['model_reconstruction'])
        assert report == ns['report']
        np.testing.assert_array_equal(np.random.get_state()[1], original_numpy_state[1])
        assert np.random.get_state()[2:] == original_numpy_state[2:]
        for name, value in bn_before.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

        loss_inputs = {}
        def capture_loss_inputs(module, inputs):
            loss_inputs['prediction'], loss_inputs['target'] = inputs
        hook = model.rec_loss.register_forward_pre_hook(capture_loss_inputs)
        try:
            with torch.inference_mode():
                tensor = torch.from_numpy(ns['model_points']).unsqueeze(0).to(next(model.parameters()).device)
                one = predict_masked_patches(model, tensor)
                relative = one['patches'] - one['centers'][one['patch_ids']].unsqueeze(1)
                torch.testing.assert_close(relative, loss_inputs['prediction'], atol=2e-5, rtol=1e-5)
                actual_loss = model.rec_loss(relative, loss_inputs['target']).item()
                assert abs(actual_loss - one['loss']) < max(1e-5, abs(one['loss']) * 1e-4)
        finally:
            hook.remove()
        for module in (model.group_dividers[1], model.h_encoder, model.rec_head):
            assert not module._forward_hooks, 'Temporary capture hook leaked'
        output_cloud = laspy.read(output)
        # LAS integer quantization allows half a source-scale step per coordinate.
        error = np.abs(output_cloud.xyz - ns['reconstructed_xyz'])
        assert (error <= ns['source_cloud'].header.scales / 2 + 1e-8).all()
        assert output.with_suffix('.png').stat().st_size > 0
        metadata = json.loads(output.with_suffix('.json').read_text())
        assert metadata['predicted_points'] == len(output_cloud.points)
        assert source.stat().st_size == original_source_stat.st_size
        assert source.stat().st_mtime_ns == original_source_stat.st_mtime_ns
        print('PASS: trained checkpoint, deterministic patches, patch translation, loss, '
              'BatchNorm state, plot, LAS coordinates and report.', flush=True)


if __name__ == '__main__':
    main()
