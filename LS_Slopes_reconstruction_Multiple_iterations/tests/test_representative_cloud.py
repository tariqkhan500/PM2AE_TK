"""Analytical checks for averaging generated points in source coordinates."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from representative_cloud import patch_correspondence_average, voxel_average


class VoxelAverageTests(unittest.TestCase):
    def test_averages_each_occupied_voxel_and_counts_predictions(self):
        xyz = np.array([
            [10.1, 20.1, 30.1],
            [10.3, 20.2, 30.2],
            [11.1, 21.1, 31.1],
            [11.3, 21.2, 31.2],
        ], dtype=np.float64)
        original = xyz.copy()
        means, counts = voxel_average(xyz, 0.5)
        np.testing.assert_allclose(means, [[10.2, 20.15, 30.15], [11.2, 21.15, 31.15]])
        np.testing.assert_array_equal(counts, [2, 2])
        self.assertEqual(means.dtype, np.dtype('float64'))
        self.assertEqual(counts.dtype, np.dtype('uint32'))
        np.testing.assert_array_equal(xyz, original)

    def test_voxels_are_anchored_at_the_minimum_xyz(self):
        means, counts = voxel_average(np.array([[0.7, 4.0, 9.0], [1.1, 4.0, 9.0]]), 1.0)
        np.testing.assert_allclose(means, [[0.9, 4.0, 9.0]])
        np.testing.assert_array_equal(counts, [2])

    def test_rejects_empty_malformed_and_nonfinite_clouds(self):
        cases = [np.empty((0, 3)), np.empty((3, 0)), np.ones((3, 2)),
                 np.array([[0.0, np.nan, 0.0]]), np.array([[0.0, np.inf, 0.0]])]
        for xyz in cases:
            with self.subTest(xyz=xyz):
                with self.assertRaises(ValueError):
                    voxel_average(xyz, 1.0)

    def test_rejects_invalid_voxel_sizes(self):
        for size in (0, -1, np.nan, np.inf, -np.inf, True, '1.0'):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    voxel_average(np.ones((1, 3)), size)

    def test_rejects_index_overflow_from_tiny_voxels(self):
        with self.assertRaises(ValueError):
            voxel_average(np.array([[0.0, 0.0, 0.0], [1e20, 0.0, 0.0]]), 1e-20)


class PatchCorrespondenceTests(unittest.TestCase):
    def test_unordered_slots_are_matched_before_averaging(self):
        predictions = np.array([
            [[[[0., 0., 0.], [10., 0., 0.]]]],
            [[[[11., 0., 0.], [1., 0., 0.]]]],
        ])
        masked_ids = np.zeros((2, 1, 1), dtype=np.int64)
        for method in ('nn', 'ot'):
            with self.subTest(method=method):
                local, counts = patch_correspondence_average(predictions, masked_ids, 1, method)
                np.testing.assert_allclose(local, [[[0.5, 0., 0.], [10.5, 0., 0.]]])
                np.testing.assert_array_equal(counts, [[2, 2]])
                self.assertEqual(local.dtype, np.dtype('float64'))
                self.assertEqual(counts.dtype, np.dtype('uint32'))

    def test_nn_many_to_one_counts_every_overlap_prediction_once(self):
        predictions = np.array([[
            [[[0., 0., 0.], [10., 0., 0.]]],
            [[[1., 0., 0.], [2., 0., 0.]]],
        ]])
        masked_ids = np.zeros((1, 2, 1), dtype=np.int64)
        original = predictions.copy()
        local, counts = patch_correspondence_average(predictions, masked_ids, 1, 'nn')
        np.testing.assert_allclose(local, [[[1., 0., 0.], [10., 0., 0.]]])
        np.testing.assert_array_equal(counts, [[3, 1]])
        self.assertEqual(int(counts.sum()), 4)
        np.testing.assert_array_equal(predictions, original)

    def test_ot_uses_one_to_one_assignments_for_each_overlap_set(self):
        predictions = np.array([[
            [[[0., 0., 0.], [10., 0., 0.]]],
            [[[1., 0., 0.], [2., 0., 0.]]],
        ]])
        masked_ids = np.zeros((1, 2, 1), dtype=np.int64)
        local, counts = patch_correspondence_average(predictions, masked_ids, 1, 'ot')
        np.testing.assert_allclose(local, [[[0.5, 0., 0.], [6., 0., 0.]]])
        np.testing.assert_array_equal(counts, [[2, 2]])

    def test_ot_reassigns_against_evolving_template(self):
        predictions = np.array([
            [[[[0., 0., 0.], [2., 0., 0.]]]],
            [[[[0., 10., 0.], [2., -10., 0.]]]],
            [[[[0., 0., 0.], [2., 1., 0.]]]],
        ])
        masked_ids = np.zeros((3, 1, 1), dtype=np.int64)
        local, counts = patch_correspondence_average(predictions, masked_ids, 1, 'ot')
        np.testing.assert_allclose(local, [[[2 / 3, 11 / 3, 0.], [4 / 3, -10 / 3, 0.]]])
        np.testing.assert_array_equal(counts, [[3, 3]])

    def test_output_follows_patch_id_order_across_passes(self):
        predictions = np.array([[
            [
                [[10., 0., 0.], [20., 0., 0.]],
                [[0., 0., 0.], [2., 0., 0.]],
            ],
            [
                [[1., 0., 0.], [3., 0., 0.]],
                [[11., 0., 0.], [21., 0., 0.]],
            ],
        ]])
        masked_ids = np.array([[[1, 0], [0, 1]]])
        local, counts = patch_correspondence_average(predictions, masked_ids, 2, 'nn')
        np.testing.assert_allclose(local, [
            [[0.5, 0., 0.], [2.5, 0., 0.]],
            [[10.5, 0., 0.], [20.5, 0., 0.]],
        ])
        np.testing.assert_array_equal(counts, [[2, 2], [2, 2]])

    def test_rejects_invalid_method_and_patch_count(self):
        predictions = np.zeros((1, 1, 1, 2, 3))
        masked_ids = np.zeros((1, 1, 1), dtype=np.int64)
        for method in ('voxel', 'NN', '', None):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    patch_correspondence_average(predictions, masked_ids, 1, method)
        for patch_count in (0, -1, True, 1.5):
            with self.subTest(patch_count=patch_count):
                with self.assertRaises(ValueError):
                    patch_correspondence_average(predictions, masked_ids, patch_count, 'nn')

    def test_rejects_malformed_ids_predictions_and_missing_coverage(self):
        predictions = np.zeros((1, 1, 1, 2, 3))
        ids = np.zeros((1, 1, 1), dtype=np.int64)
        cases = [
            (np.zeros((1, 1, 2, 3)), ids, 1),
            (np.zeros((1, 1, 1, 2, 2)), ids, 1),
            (np.zeros((0, 1, 1, 2, 3)), np.zeros((0, 1, 1), dtype=np.int64), 1),
            (predictions, np.zeros((1, 1, 2), dtype=np.int64), 1),
            (predictions, np.zeros((1, 1, 1), dtype=np.float64), 1),
            (predictions, np.array([[[-1]]]), 1),
            (predictions, np.array([[[1]]]), 1),
            (predictions, ids, 2),
            (np.full_like(predictions, np.nan), ids, 1),
        ]
        for pred, masked, count in cases:
            with self.subTest(shape=pred.shape, ids=masked, patch_count=count):
                with self.assertRaises(ValueError):
                    patch_correspondence_average(pred, masked, count, 'nn')

    def test_each_iteration_must_cover_every_patch(self):
        predictions = np.zeros((2, 1, 1, 2, 3))
        ids = np.array([[[0]], [[1]]])
        with self.assertRaises(ValueError):
            patch_correspondence_average(predictions, ids, 2, 'ot')


if __name__ == '__main__':
    unittest.main()
