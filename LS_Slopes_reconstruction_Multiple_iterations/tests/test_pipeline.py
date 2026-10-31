"""LAS exports and notebook orchestration with the CUDA forward boundary stubbed."""
import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import laspy
import numpy as np
from pyproj import CRS

from test_core import prediction_fixture
FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOLDER))
sys.path.insert(0, str(FOLDER.parent))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = importlib.import_module('reconstruct_slopes')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_root = self.root / 'input'
        self.source = self.source_root / 'UTU' / 'SU_123.las'
        self.source.parent.mkdir(parents=True)
        header = laspy.LasHeader(point_format=6, version='1.4')
        header.scales = [.001]*3
        header.add_crs(CRS.from_epsg(6348))
        cloud = laspy.LasData(header)
        cloud.x = np.arange(8) + 1000.
        cloud.y = np.arange(8) + 2000.
        cloud.z = np.arange(8) + 100.
        cloud.classification = np.full(8, 5, dtype=np.uint8)
        cloud.write(self.source)
        self.checkpoint = self.root / 'best.pt'
        self.checkpoint.write_bytes(b'fixture checkpoint; loading is a GPU boundary')
        self.before = self.source.read_bytes()
        config = {'normalization': 'subtract cloud XYZ centroid only; no rescaling',
                  'model': {'group_divisors': [2,2,2], 'mask_ratio': .8}}
        self.addCleanup(patch.stopall)
        patch.object(self.pipeline, 'load_checkpoint_model', return_value=(object(), config, {'epoch': 1})).start()
        patch.object(self.pipeline, 'reconstruct_iterations', side_effect=lambda *a, **kw: prediction_fixture()).start()

    def run_batch(self, output, **kwargs):
        return self.pipeline.run_pipeline(self.source_root, self.checkpoint, output,
                                          iterations=2, passes_per_iteration=2, **kwargs)

    def test_all_methods_export_first_predictions_and_honest_coverage(self):
        for method in ('voxel', 'nn', 'ot'):
            with self.subTest(method=method):
                output = self.root / method
                manifest = self.run_batch(output, representative_method=method)
                self.assertEqual(manifest['counts'], {'selected':1, 'reconstructed':1, 'failed':0})
                unit = manifest['units'][0]
                self.assertEqual(unit['coverage_fraction'], .75)
                self.assertEqual(unit['coverage_by_iteration'], [.5,.75])
                self.assertEqual(unit['total_passes'], 4)
                self.assertEqual(unit['all_predicted_points'], 6)
                cloud = laspy.read(output / unit['all_las_file'])
                self.assertEqual(len(cloud.points), 6)
                self.assertTrue((cloud.classification == 0).all())
                self.assertEqual(cloud.header.parse_crs(), CRS.from_epsg(6348))
                np.testing.assert_array_equal(cloud.patch_id, [1,1,2,2,3,3])
                np.testing.assert_allclose(cloud.x, np.array([1,3,10,12,20,22])+1003.5)
                with np.load(output / unit['combined_file'], allow_pickle=False) as archive:
                    self.assertEqual(archive['representative_counts'].sum(), 6)
                    self.assertEqual(archive['coverage_counts'][0,0], 0)
                    np.testing.assert_allclose(archive['all_predicted_xyz'], cloud.xyz, atol=.00051)
                passes = sorted((output / 'passes').rglob('*.npz'))
                self.assertEqual(len(passes), 4)
                retained = []
                for pass_path in passes:
                    with np.load(pass_path, allow_pickle=False) as archive:
                        retained.extend(archive['retained_patch_ids'].tolist())
                        self.assertEqual(len(archive['predicted_patches']), len(archive['retained_patch_ids']))
                self.assertEqual(retained, [1,2,3])
                self.assertEqual(manifest['prediction_selection'], 'first_per_patch')
                self.assertIn('_first_predictions.las', unit['all_las_file'])
                self.assertEqual(self.source.read_bytes(), self.before)
                with self.assertRaises(FileExistsError): self.run_batch(output)

    def test_bad_cloud_is_reported_and_export_failure_rolls_back(self):
        bad = self.source.parent / 'SU_999.las'
        bad.write_bytes(b'bad LAS')
        result = self.run_batch(self.root / 'bad', limit=None)
        self.assertEqual(result['counts'], {'selected':2, 'reconstructed':1, 'failed':1})
        self.assertEqual(result['status'], 'completed_with_errors')
        with patch.object(self.pipeline, 'write_point_clouds', side_effect=OverflowError('test overflow')):
            result = self.run_batch(self.root / 'overflow', limit=1)
        self.assertEqual(result['counts']['reconstructed'], 0)
        self.assertFalse(list((self.root / 'overflow').rglob('*.las')))
        self.assertFalse(list((self.root / 'overflow').rglob('*.npz')))

    def test_invalid_settings_fail_before_output(self):
        for index, kwargs in enumerate(({'representative_method':'bad'}, {'voxel_size':0},
                                       {'representative_method':'ot','voxel_size':1},
                                       {'limit':0}, {'seed':2**32}, {'cuda_device':-1})):
            output = self.root / f'invalid{index}'
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.run_batch(output, **kwargs)
            self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()
