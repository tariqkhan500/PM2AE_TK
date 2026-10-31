"""Check the exported artifacts read correctly in point-cloud software."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import laspy
import numpy as np
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from point_cloud_io import write_point_clouds


class PointCloudIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.header = laspy.LasHeader(point_format=6, version="1.4")
        self.header.scales = np.array([0.01, 0.01, 0.001])
        self.header.offsets = np.array([703000.0, 2020000.0, 100.0])
        self.crs = CRS.from_user_input("EPSG:6348+6641")
        self.header.add_crs(self.crs)
        self.header.add_extra_dim(laspy.ExtraBytesParams(name="landslide", type=np.uint8))
        self.xyz = np.array([
            [703816.593457, 2020393.184592, 98.254391],
            [703842.272319, 2020418.044938, 125.894839],
            [703830.123456, 2020402.234567, 112.456789],
        ], dtype=np.float64)
        self.patch_ids = np.array([4, 4, 9], dtype=np.int64)
        self.pass_ids = np.array([0, 0, 1], dtype=np.int64)
        self.las_path = self.directory / "LAM_1" / "SU_27077_reconstructed.las"
        self.ply_path = self.directory / "LAM_1" / "SU_27077_reconstructed.ply"

    def write(self, **overrides):
        arguments = dict(xyz=self.xyz, source_header=self.header,
                         las_path=self.las_path, ply_path=self.ply_path,
                         su_id=27077, patch_ids=self.patch_ids, pass_ids=self.pass_ids)
        arguments.update(overrides)
        return write_point_clouds(**arguments)

    def test_las_round_trip_retains_world_coordinates_crs_and_provenance(self):
        paths = self.write()
        self.assertEqual(paths, {"las": str(self.las_path), "ply": str(self.ply_path)})
        exported = laspy.read(self.las_path)
        restored = np.column_stack((exported.x, exported.y, exported.z))
        self.assertTrue(np.all(np.abs(restored - self.xyz) <= self.header.scales / 2 + 1e-9))
        self.assertEqual(str(exported.header.version), "1.4")
        self.assertEqual(exported.header.point_format.id, 6)
        np.testing.assert_array_equal(exported.header.scales, self.header.scales)
        np.testing.assert_array_equal(exported.header.offsets, self.header.offsets)
        self.assertTrue(exported.header.parse_crs().equals(self.crs))
        np.testing.assert_array_equal(exported["SU_id"], [27077, 27077, 27077])
        np.testing.assert_array_equal(exported["patch_id"], [4, 4, 9])
        np.testing.assert_array_equal(exported["reconstruction_pass"], [0, 0, 1])
        self.assertEqual(exported["SU_id"].dtype, np.dtype("uint32"))
        self.assertEqual(exported["patch_id"].dtype, np.dtype("uint32"))
        self.assertEqual(exported["reconstruction_pass"].dtype, np.dtype("uint16"))

    def test_raw_cloud_records_iteration_ids_without_voxel_count(self):
        self.write(pass_ids=np.array([1, 1, 2]), iteration_ids=np.array([1, 1, 3]))
        exported = laspy.read(self.las_path)
        np.testing.assert_array_equal(exported["patch_id"], [4, 4, 9])
        np.testing.assert_array_equal(exported["reconstruction_pass"], [1, 1, 2])
        np.testing.assert_array_equal(exported["iteration_id"], [1, 1, 3])
        self.assertEqual(exported["iteration_id"].dtype, np.dtype("uint32"))
        self.assertNotIn("voxel_count", exported.point_format.dimension_names)

    def test_representative_cloud_records_counts_without_false_patch_or_pass_ids(self):
        self.write(patch_ids=None, pass_ids=None, voxel_counts=np.array([2, 7, 1]))
        exported = laspy.read(self.las_path)
        self.assertTrue(exported.header.parse_crs().equals(self.crs))
        self.assertEqual(set(exported.point_format.extra_dimension_names), {"SU_id", "voxel_count"})
        np.testing.assert_array_equal(exported["voxel_count"], [2, 7, 1])
        self.assertEqual(exported["voxel_count"].dtype, np.dtype("uint32"))
        np.testing.assert_array_equal(exported["SU_id"], [27077, 27077, 27077])
        np.testing.assert_allclose(np.loadtxt(self.ply_path, skiprows=7), self.xyz, rtol=0, atol=0.5e-6)

    def test_matched_representative_preserves_patch_ids_and_support_counts(self):
        self.write(pass_ids=None, support_counts=np.array([4, 3, 5], dtype=np.uint32))
        exported = laspy.read(self.las_path)
        self.assertEqual(set(exported.point_format.extra_dimension_names),
                         {'SU_id', 'patch_id', 'support_count'})
        np.testing.assert_array_equal(exported['SU_id'], [27077, 27077, 27077])
        np.testing.assert_array_equal(exported['patch_id'], [4, 4, 9])
        np.testing.assert_array_equal(exported['support_count'], [4, 3, 5])
        self.assertEqual(exported['support_count'].dtype, np.dtype('uint32'))
        self.assertTrue(exported.header.parse_crs().equals(self.crs))
        np.testing.assert_array_equal(exported.header.scales, self.header.scales)
        np.testing.assert_array_equal(exported.header.offsets, self.header.offsets)
        restored = np.column_stack((exported.x, exported.y, exported.z))
        self.assertTrue(np.all(np.abs(restored - self.xyz) <= self.header.scales / 2 + 1e-9))
        np.testing.assert_array_equal(np.loadtxt(self.ply_path, skiprows=7), self.xyz)

    def test_matched_representative_rejects_conflicting_or_invalid_support_counts(self):
        cases = [
            dict(support_counts=np.array([1, 1, 1])),
            dict(pass_ids=None, iteration_ids=np.array([1, 1, 1]), support_counts=np.array([1, 1, 1])),
            dict(pass_ids=None, voxel_counts=np.array([1, 1, 1]), support_counts=np.array([1, 1, 1])),
            dict(patch_ids=None, pass_ids=None, support_counts=np.array([1, 1, 1])),
            dict(pass_ids=None, support_counts=np.array([1, 2])),
            dict(pass_ids=None, support_counts=np.array([1, 0, 2])),
            dict(pass_ids=None, support_counts=np.array([1, -1, 2])),
            dict(pass_ids=None, support_counts=np.array([1, 2**32, 2], dtype=np.int64)),
            dict(pass_ids=None, support_counts=np.array([1., 2., 3.])),
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.write(**overrides)
                self.assertFalse(self.las_path.exists())
                self.assertFalse(self.ply_path.exists())

    def test_metadata_modes_reject_incomplete_or_conflicting_provenance(self):
        cases = [dict(patch_ids=None), dict(pass_ids=None),
                 dict(patch_ids=None, pass_ids=None),
                 dict(iteration_ids=np.array([1, 2])),
                 dict(iteration_ids=np.array([1, 0, 2])),
                 dict(patch_ids=None, pass_ids=None, iteration_ids=np.array([1, 1, 1])),
                 dict(voxel_counts=np.array([1, 2, 3])),
                 dict(patch_ids=None, pass_ids=None, voxel_counts=np.array([1, 0, 3])),
                 dict(patch_ids=None, pass_ids=None, voxel_counts=np.array([1, 2**32, 3])),
                 dict(patch_ids=None, pass_ids=None, voxel_counts=np.array([1, 2])),
                 dict(patch_ids=None, pass_ids=None, voxel_counts=np.array([1.0, 2.0, 3.0]))]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.write(**overrides)
                self.assertFalse(self.las_path.exists())
                self.assertFalse(self.ply_path.exists())

    def test_generated_points_do_not_inherit_landslide_labels_or_mutate_source_header(self):
        original_dimensions = tuple(self.header.point_format.dimension_names)
        original_crs = self.header.parse_crs()
        self.write()
        exported = laspy.read(self.las_path)
        self.assertNotIn("landslide", exported.point_format.dimension_names)
        self.assertEqual(tuple(self.header.point_format.dimension_names), original_dimensions)
        self.assertTrue(self.header.parse_crs().equals(original_crs))

    def test_ply_ascii_round_trip_preserves_double_xyz(self):
        self.write()
        lines = self.ply_path.read_text(encoding="ascii").splitlines()
        self.assertEqual(lines[:7], ["ply", "format ascii 1.0", "element vertex 3",
                                    "property double x", "property double y",
                                    "property double z", "end_header"])
        parsed = np.loadtxt(self.ply_path, skiprows=7, dtype=np.float64)
        np.testing.assert_array_equal(parsed, self.xyz)

    def test_ply_preserves_distinct_submicrometer_coordinates(self):
        self.header.scales = np.full(3, 1e-7)
        self.header.offsets = np.array([1.0, 2.0, 3.0])
        precise_xyz = np.array([
            [1.0000001, 2.0000001, 3.0000001],
            [1.0000002, 2.0000002, 3.0000002],
            [1.0000003, 2.0000003, 3.0000003],
        ], dtype=np.float64)
        self.write(xyz=precise_xyz)
        parsed = np.loadtxt(self.ply_path, skiprows=7, dtype=np.float64)
        np.testing.assert_array_equal(parsed, precise_xyz)
        self.assertEqual(len(np.unique(parsed[:, 0])), 3)

    def test_existing_either_output_is_untouched_and_does_not_create_counterpart(self):
        for existing, other in ((self.las_path, self.ply_path), (self.ply_path, self.las_path)):
            with self.subTest(existing=existing.suffix):
                existing.parent.mkdir(parents=True, exist_ok=True)
                existing.write_bytes(b"existing artifact")
                with self.assertRaises(FileExistsError):
                    self.write()
                self.assertEqual(existing.read_bytes(), b"existing artifact")
                self.assertFalse(other.exists())
                existing.unlink()

    def test_invalid_coordinates_or_lengths_create_no_artifacts(self):
        cases = [dict(xyz=np.empty((0, 3))), dict(xyz=np.ones((3, 2))),
                 dict(xyz=np.array([[np.nan, 0., 0.]] * 3)),
                 dict(xyz=np.array([[0., np.inf, 0.]] * 3)),
                 dict(patch_ids=np.array([1, 2])), dict(pass_ids=np.array([0, 1])),
                 dict(patch_ids=np.array([[1], [2], [3]]))]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.write(**overrides)
                self.assertFalse(self.las_path.exists())
                self.assertFalse(self.ply_path.exists())

    def test_ids_cannot_be_silently_truncated_or_wrapped(self):
        cases = [dict(su_id=-1), dict(su_id=2**32), dict(su_id=1.5), dict(su_id=True),
                 dict(patch_ids=np.array([0, -1, 2])),
                 dict(patch_ids=np.array([0, 2**32, 2], dtype=np.int64)),
                 dict(patch_ids=np.array([0., 1.5, 2.])),
                 dict(pass_ids=np.array([0, 2**16, 1])),
                 dict(pass_ids=np.array([0, -1, 1]))]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.write(**overrides)
                self.assertFalse(self.las_path.exists())
                self.assertFalse(self.ply_path.exists())

    def test_distinct_paths_are_required(self):
        with self.assertRaises(ValueError):
            self.write(ply_path=self.las_path)
        self.assertFalse(self.las_path.exists())

    def test_las_coordinate_overflow_is_rejected_before_creating_outputs(self):
        invalid_xyz = self.xyz.copy()
        invalid_xyz[0, 0] = 1e12
        with self.assertRaises((ValueError, OverflowError)):
            self.write(xyz=invalid_xyz)
        self.assertFalse(self.las_path.exists())
        self.assertFalse(self.ply_path.exists())


if __name__ == "__main__":
    unittest.main()
