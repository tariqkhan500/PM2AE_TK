"""CPU checks for reconstruction I/O; CUDA smoke runs separately."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_DEPS = all(importlib.util.find_spec(n) for n in ('numpy', 'laspy', 'matplotlib'))


class ReconstructionNotebookTests(unittest.TestCase):
    def test_notebook_exists_and_cells_compile(self):
        path = ROOT / 'Point2MAE_reconstruction.ipynb'
        self.assertTrue(path.is_file(), 'Reconstruction notebook is missing')
        nb = json.loads(path.read_text())
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])


@unittest.skipUnless(HAS_DEPS, 'Requires the pointm2ae environment')
class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('point2mae_reconstruction'),
                             'Reconstruction helper module is missing')
        import point2mae_reconstruction as reconstruction
        import numpy as np
        self.r = reconstruction
        self.np = np

    def test_invalid_mask_override_fails_before_loading_checkpoint(self):
        for ratio in (0, 1, -0.1, 1.1, True, float('nan'), float('inf'), '0.8'):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, 'mask_ratio'):
                self.r.load_checkpoint_model('missing.pt', 'missing', mask_ratio=ratio)

    def test_preprocessing_roundtrip_uses_checkpoint_policy(self):
        np = self.np
        xyz = np.array([[1e6 + i, 2e6 + 2*i, 100 + 3*i] for i in range(8)])
        saved = xyz.copy()
        for policy, scale in [('subtract cloud XYZ centroid only; no rescaling', 1.),
                              ('subtract cloud XYZ mean, divide by max XY radius', (3.5**2 + 7**2)**.5)]:
            points, centroid, actual_scale = self.r.prepare_cloud(xyz, policy)
            self.assertEqual(points.dtype, np.float32)
            np.testing.assert_allclose(centroid, [1000003.5, 2000007., 110.5])
            self.assertAlmostEqual(actual_scale, scale)
            np.testing.assert_allclose(points[0], [-3.5/scale, -7/scale, -10.5/scale])
            np.testing.assert_allclose(self.r.restore_coordinates(points, centroid, actual_scale),
                                       xyz, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(xyz, saved)

    def test_bad_coordinates_and_unknown_normalization_fail(self):
        np = self.np
        for xyz in (np.zeros((7, 3)), np.zeros((8, 2)), np.full((8, 3), np.nan)):
            with self.assertRaises(ValueError):
                self.r.prepare_cloud(xyz, 'subtract cloud XYZ centroid only; no rescaling')
        with self.assertRaisesRegex(ValueError, 'normalization'):
            self.r.prepare_cloud(np.zeros((8, 3)), 'unknown')
        with self.assertRaises(ValueError):
            self.r.prepare_cloud(np.zeros((8, 3)), 'subtract cloud XYZ mean, divide by max XY radius')
        points, _, _ = self.r.prepare_cloud(np.zeros((8, 3)),
                                          'subtract cloud XYZ centroid only; no rescaling')
        self.assertEqual(points.shape, (8, 3))

    def test_las_export_preserves_coordinates_crs_and_source(self):
        import laspy
        from laspy.vlrs.known import WktCoordinateSystemVlr
        np = self.np
        header = laspy.LasHeader(point_format=6, version='1.4')
        header.scales = [.001, .002, .003]
        header.offsets = [1000000., 2000000., 100.]
        header.global_encoding.wkt = True
        wkt = 'LOCAL_CS["Local terrain",UNIT["metre",1]]'
        header.vlrs.append(WktCoordinateSystemVlr(wkt))
        xyz = np.array([[1000001.234, 2000005.678, 109.876], [1000003.21, 2000004.32, 108.765]])
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'original.las'
            source.write_bytes(b'original unchanged')
            out = Path(d) / 'reconstruction.las'
            self.r.save_reconstruction_las(out, xyz, header, source)
            saved = laspy.read(out)
            np.testing.assert_allclose(saved.xyz, xyz, rtol=0, atol=.0016)
            np.testing.assert_allclose(saved.header.scales, header.scales)
            self.assertEqual(saved.header.point_count, 2)
            self.assertEqual(saved.header.vlrs[0].string, wkt)
            self.assertTrue(saved.header.global_encoding.wkt)
            self.assertTrue((saved.intensity == 0).all())
            self.assertTrue((saved.classification == 0).all())
            self.assertEqual(source.read_bytes(), b'original unchanged')
            with self.assertRaises(FileExistsError):
                self.r.save_reconstruction_las(out, xyz, header, source)
            with self.assertRaises(ValueError):
                self.r.save_reconstruction_las(source, xyz, header, source)
            for invalid in (np.empty((0, 3)), np.full((2, 3), np.inf), np.full((2, 3), 1e15)):
                with self.assertRaises(ValueError):
                    self.r.save_reconstruction_las(Path(d)/'invalid.las', invalid, header, source)
                self.assertFalse((Path(d)/'invalid.las').exists())

    def test_plot_uses_shared_limits_and_only_subsamples_display(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        np = self.np
        original = np.arange(60, dtype=float).reshape(20, 3)
        predicted = original[:10] * 2
        before = predicted.copy()
        fig = self.r.plot_comparison(original, predicted, max_points=5)
        left, right = fig.axes[:2]
        self.assertEqual(left.get_xlim(), right.get_xlim())
        self.assertEqual(left.get_ylim(), right.get_ylim())
        self.assertEqual(left.get_zlim(), right.get_zlim())
        self.assertEqual(left.collections[0].get_clim(), right.collections[0].get_clim())
        self.assertEqual(len(left.collections[0]._offsets3d[0]), 5)
        np.testing.assert_array_equal(predicted, before)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
