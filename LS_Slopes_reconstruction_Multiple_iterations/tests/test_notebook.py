"""Execute every notebook code cell with a deterministic GPU-boundary fixture."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pointm2ae-slope-test-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

import test_pipeline
FOLDER = test_pipeline.FOLDER


class NotebookTests(unittest.TestCase):
    setUp = test_pipeline.PipelineTests.setUp

    def test_notebook_runs_every_cell_for_all_representative_methods(self):
        notebook = nbformat.read(FOLDER / 'multiple_iteration_reconstruction.ipynb', as_version=4)
        nbformat.validate(notebook)
        self.addCleanup(plt.close, 'all')
        for method in ('voxel', 'nn', 'ot'):
            namespace = {'__name__':'__notebook_test__'}
            with patch.object(Path, 'cwd', return_value=FOLDER), patch.object(plt, 'show'), redirect_stdout(io.StringIO()):
                for cell in notebook.cells:
                    if cell.cell_type != 'code': continue
                    exec(compile(cell.source, cell.id, 'exec'), namespace)
                    if cell.id == 'settings':
                        namespace.update(INPUT_DIR=self.source_root, CHECKPOINT=self.checkpoint,
                                         OUTPUT_DIR=self.root / method, ITERATIONS=2, PASSES_PER_ITERATION=2,
                                         REPRESENTATIVE_METHOD=method, VOXEL_SIZE=None, LIMIT=1)
            self.assertEqual(namespace['manifest']['counts']['reconstructed'], 1)
            self.assertEqual(float(namespace['rows'][0]['coverage_fraction']), .75)
            self.assertEqual(len(namespace['fig'].axes), 6)
            self.assertTrue(all(ax.collections for ax in namespace['fig'].axes))
            self.assertEqual(namespace['all_xyz'].shape, (6, 3))
            self.assertEqual(self.source.read_bytes(), self.before)
            plt.close(namespace['fig'])


if __name__ == '__main__': unittest.main()
