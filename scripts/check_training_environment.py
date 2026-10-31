"""Run notebook preparation and GPU smoke checks, without starting training."""
import json
import os
from pathlib import Path
import sys
import tempfile


root = Path(__file__).resolve().parents[1]
prefix = Path(sys.prefix)
os.environ.setdefault('CUDA_HOME', str(prefix))
os.environ.setdefault('CPATH', str(prefix / 'targets/x86_64-linux/include'))
os.environ.setdefault('CC', '/usr/bin/gcc')
os.environ.setdefault('CXX', '/usr/bin/g++')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ.setdefault('MAX_JOBS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['PATH'] = str(prefix / 'bin') + os.pathsep + os.environ['PATH']

notebook = json.loads((root / 'Point2MAE_training.ipynb').read_text())
cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
namespace = {}
exec(compile(cells['settings'].replace('INTERACTIVE = True', 'INTERACTIVE = False'),
             'settings', 'exec'), namespace)
with tempfile.TemporaryDirectory(prefix='pointm2ae-check-') as temporary:
    namespace['RUN_DIR'] = Path(temporary) / 'run'
    for name in ('manifest-functions', 'full-cloud-helpers', 'prepare-manifests',
                 'environment-check', 'dataset', 'xy-model'):
        print(f'Checking notebook cell: {name}', flush=True)
        exec(compile(cells[name], name, 'exec'), namespace)
    import torch
    from knn_cuda import KNN
    reference = torch.randn(1, 32, 3, device='cuda')
    query = torch.randn(1, 7, 3, device='cuda')
    distance, indices = KNN(k=4, transpose_mode=True)(reference, query)
    expected, expected_indices = torch.cdist(query, reference).topk(4, largest=False)
    torch.testing.assert_close(distance, expected, rtol=1e-4, atol=1e-5)
    assert torch.equal(indices, expected_indices)
    torch.cuda.synchronize()
    print('PASS: notebook GPU smoke checks and KNN reference comparison.', flush=True)
