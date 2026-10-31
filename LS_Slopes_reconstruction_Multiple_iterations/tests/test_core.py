"""Predicted-only aggregation, including ragged masks and partial coverage."""
from pathlib import Path
import sys
import unittest
import numpy as np

FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOLDER))
sys.path.insert(0, str(FOLDER.parent))
import reconstruction_core as core


def prediction_fixture():
    # Four possible patches; zero is never predicted. Different mask counts.
    return {
        'centers_model': np.zeros((4, 3)),
        'coverage_counts': np.array([[0, 2, 1, 0], [0, 1, 1, 1]], dtype=np.uint32),
        'iteration_seeds': np.array([42, 43], dtype=np.uint32),
        'passes': [
            {'iteration_id': 1, 'pass_id': 1, 'patch_ids': np.array([1, 2]),
             'patches': np.array([[[1., 0, 0], [3, 0, 0]], [[10, 0, 0], [12, 0, 0]]]), 'loss': 1.},
            {'iteration_id': 1, 'pass_id': 2, 'patch_ids': np.array([1]),
             'patches': np.array([[[5., 0, 0], [3, 0, 0]]]), 'loss': 2.},
            {'iteration_id': 2, 'pass_id': 1, 'patch_ids': np.array([2]),
             'patches': np.array([[[14., 0, 0], [12, 0, 0]]]), 'loss': 3.},
            {'iteration_id': 2, 'pass_id': 2, 'patch_ids': np.array([1, 3]),
             'patches': np.array([[[5., 0, 0], [7, 0, 0]], [[20, 0, 0], [22, 0, 0]]]), 'loss': 4.},
        ],
    }


class AggregationTests(unittest.TestCase):
    def test_keeps_first_patch_prediction_across_passes_and_iterations(self):
        result = core.retain_first_predictions(prediction_fixture())
        clouds = core.collect_predictions(result, np.array([100., 200., 300.]), 2., 'voxel', 100.)
        np.testing.assert_array_equal(clouds['all_predicted_xyz'][:, 0],
                                      [102, 106, 120, 124, 140, 144])
        np.testing.assert_array_equal(clouds['all_predicted_xyz'][:, 1:], np.tile([200, 300], (6, 1)))
        np.testing.assert_array_equal(clouds['patch_ids'], [1, 1, 2, 2, 3, 3])
        np.testing.assert_array_equal(clouds['iteration_ids'], [1]*4 + [2]*2)
        np.testing.assert_array_equal(clouds['pass_ids'], [1]*4 + [2]*2)
        self.assertEqual(clouds['representative_counts'].sum(), 6)
        np.testing.assert_allclose(clouds['representative_xyz'], [[122.66666666666667, 200, 300]])
        self.assertEqual([len(r['patch_ids']) for r in result['passes']], [2,0,0,1])
        self.assertEqual([len(r['masked_ids']) for r in result['passes']], [2,1,1,2])
        np.testing.assert_array_equal(result['coverage_counts'], [[0,2,1,0],[0,1,1,1]])

    def test_nn_and_ot_use_only_retained_first_predictions(self):
        for method in ('nn', 'ot'):
            with self.subTest(method=method):
                clouds = core.collect_predictions(core.retain_first_predictions(prediction_fixture()), np.zeros(3), 1., method, None)
                self.assertEqual(clouds['representative_xyz'].shape, (6, 3))
                np.testing.assert_array_equal(clouds['representative_patch_ids'], [1,1,2,2,3,3])
                self.assertEqual(clouds['representative_counts'].sum(), 6)
                np.testing.assert_array_equal(clouds['representative_xyz'][-2:, 0], [20,22])
                if method == 'ot':
                    np.testing.assert_allclose(clouds['representative_xyz'][:,0], [1,3,10,12,20,22])
                    np.testing.assert_array_equal(clouds['representative_counts'], [1,1,1,1,1,1])


if __name__ == '__main__': unittest.main()
