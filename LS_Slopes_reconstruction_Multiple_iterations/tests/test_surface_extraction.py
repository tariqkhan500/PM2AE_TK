"""Analytical geometry and file-level surface pipeline checks."""
from pathlib import Path
import sys
import tempfile
import unittest

import laspy
import numpy as np
from pyproj import CRS

FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOLDER))


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        import surface_extraction
        self.s = surface_extraction

    def test_selected_methods_use_parameters_without_running_unselected_fits(self):
        from unittest.mock import patch
        points = np.array([[0., 0, 0], [.1, .1, 2], [.2, .2, 4], [2, 2, 30]])
        config = self.s.SurfaceConfig(xy_grid_size=1., percentile=25., voxel_size=100.)
        with patch.object(self.s, 'mls_surface', side_effect=AssertionError('Unselected MLS ran')):
            surfaces = list(self.s.extract_surfaces(points, config, methods=['xy_percentile', 'voxel']))
        self.assertEqual([row[0] for row in surfaces], ['xy_percentile', 'voxel'])
        np.testing.assert_allclose(surfaces[0][1][:, 2], [1., 30.])
        np.testing.assert_allclose(surfaces[1][1], [points.mean(axis=0)])
        for methods in ([], ['unknown'], ['mls', 'mls'], 'mls'):
            with self.subTest(methods=methods), self.assertRaises(ValueError):
                list(self.s.extract_surfaces(points, config, methods=methods))

    def test_voxel_centroids_and_xy_statistics(self):
        points = np.array([[0.,0,0], [.1,.1,2], [.2,.2,4], [.3,.3,10], [2,2,30]])
        xyz, support = self.s.voxel_downsample(points, 100.)
        np.testing.assert_allclose(xyz, [[.52,.52,9.2]])
        np.testing.assert_array_equal(support, [5])
        for method,z in [('mean',4),('median',3),('minimum',0),('maximum',10),('percentile',1.5)]:
            xyz, support = self.s.xy_surface(points, 1., method, percentile=25., minimum_points=2)
            np.testing.assert_allclose(xyz, [[.15,.15,z]])
            np.testing.assert_array_equal(support,[4])

    def test_mls_reproduces_quadratic_in_large_world_coordinates(self):
        x,y=np.meshgrid(np.linspace(-1,1,7),np.linspace(-1,1,7))
        local=np.column_stack((x.ravel(),y.ravel()))
        z=100+2*local[:,0]+3*local[:,1]+.2*local[:,0]**2-.4*local[:,0]*local[:,1]+.3*local[:,1]**2
        xyz=np.column_stack((local+[700000,2000000],z))
        result,info=self.s.mls_surface(xyz, neighbors=32, radius=None, degree=2, bandwidth=1., minimum_neighbors=6)
        np.testing.assert_allclose(result,xyz,rtol=0,atol=1e-8)
        self.assertEqual(info['fallback_points'],0)
        np.testing.assert_array_equal(result[:,:2],xyz[:,:2])

    def test_local_plane_projection_preserves_plane_and_reduces_noisy_height(self):
        x,y=np.meshgrid(np.arange(-2,3.),np.arange(-2,3.))
        xyz=np.column_stack((x.ravel(),y.ravel(),np.zeros(25)))
        exact,info=self.s.project_local_planes(xyz,neighbors=25,radius=None,bandwidth=10.,minimum_neighbors=6)
        np.testing.assert_allclose(exact,xyz,atol=1e-12)
        xyz[12,2]=1.
        projected,info=self.s.project_local_planes(xyz,neighbors=25,radius=None,bandwidth=10.,minimum_neighbors=6)
        self.assertLess(abs(projected[12,2]),.1)
        self.assertEqual(info['fallback_points'],0)

    def test_sparse_and_collinear_neighborhoods_leave_input_unchanged(self):
        xyz=np.column_stack((np.arange(8.),np.zeros(8),np.zeros(8)))
        for method,options in [(self.s.mls_surface,{'degree':2}), (self.s.project_local_planes,{})]:
            result,info=method(xyz,neighbors=8,radius=None,bandwidth=1.,minimum_neighbors=6,**options)
            np.testing.assert_array_equal(result,xyz)
            self.assertEqual(info['fallback_points'],8)
            result,info=method(xyz,neighbors=8,radius=.01,bandwidth=1.,minimum_neighbors=6,**options)
            np.testing.assert_array_equal(result,xyz)
            self.assertEqual(info['fallback_points'],8)

    def test_invalid_configuration_and_nonfinite_points_are_rejected(self):
        for kwargs in [{'voxel_size':0},{'xy_grid_size':float('nan')},{'percentile':101},
                       {'mls_degree':3},{'mls_neighbors':5},{'local_radius':0},{'mls_bandwidth':0}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): self.s.SurfaceConfig(**kwargs).validate()
        with self.assertRaises(ValueError): self.s.voxel_downsample(np.full((5,3),np.nan),1.)

    def test_pipeline_exports_eight_methods_in_source_crs_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'reconstructed.las'; output=root/'surfaces'
            header=laspy.LasHeader(point_format=6,version='1.4')
            header.scales=[.001]*3; header.add_crs(CRS.from_epsg(6348))
            cloud=laspy.LasData(header)
            x,y=np.meshgrid(np.arange(6.)*.2,np.arange(6.)*.2)
            cloud.x=x.ravel()+700000; cloud.y=y.ravel()+2000000; cloud.z=100+x.ravel()+y.ravel()
            cloud.classification=np.full(36,5,dtype=np.uint8); cloud.write(source)
            before=source.read_bytes()
            manifest=self.s.run_surface_pipeline(source,output,self.s.SurfaceConfig(),plot_max_points=100)
            self.assertEqual(manifest['counts'],{'selected':1,'processed':1,'failed':0})
            rows=manifest['units'][0]['methods']
            self.assertEqual(len(rows),8)
            for row in rows:
                saved=laspy.read(output/row['las_file'])
                self.assertEqual(saved.header.parse_crs(),CRS.from_epsg(6348))
                self.assertTrue((saved.classification==0).all())
                with np.load(output/row['npz_file']) as data:
                    self.assertTrue(np.all(np.abs(saved.xyz-data['xyz'])<=.0005001))
            self.assertTrue((output/manifest['units'][0]['comparison_file']).is_file())
            self.assertEqual(source.read_bytes(),before)
            with self.assertRaises(FileExistsError): self.s.run_surface_pipeline(source,output,self.s.SurfaceConfig())

    def test_reference_scorer_measures_vertical_offset_outside_hull_and_coverage(self):
        x,y=np.meshgrid(np.arange(11.),np.arange(11.))
        reference=np.column_stack((x.ravel()+700000,y.ravel()+2000000,100+.5*x.ravel()))
        score=self.s.reference_scorer(reference)
        exact=score(reference)
        self.assertEqual(exact['ref_inside_fraction'],1.)
        self.assertAlmostEqual(exact['ref_dz_rmse'],0.,places=8)
        self.assertAlmostEqual(exact['ref_coverage_mean'],0.,places=12)
        shifted=np.vstack((reference+[0,0,.25],[[699990,2000000,100]]))
        result=score(shifted)
        self.assertAlmostEqual(result['ref_inside_fraction'],121/122)
        for key in ('ref_dz_mean','ref_dz_mae','ref_dz_rmse','ref_dz_p95_abs','ref_coverage_mean'):
            self.assertAlmostEqual(result[key],.25,places=8,msg=key)
        self.assertEqual(set(result),set(self.s.REFERENCE_FIELDS))

    def test_pipeline_scores_against_matching_reference_and_records_missing_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            x,y=np.meshgrid(np.arange(6.)*.4,np.arange(6.)*.4)
            for folder,name in [('reconstructed/D','SU_1_first_predictions.las'),('reference/D','SU_1.las'),
                                ('reconstructed/D','SU_2_first_predictions.las')]:
                header=laspy.LasHeader(point_format=6,version='1.4'); header.scales=[.001]*3
                cloud=laspy.LasData(header)
                cloud.x=x.ravel()+700000; cloud.y=y.ravel()+2000000; cloud.z=100+x.ravel()
                (root/folder).mkdir(parents=True,exist_ok=True); cloud.write(root/folder/name)
            manifest=self.s.run_surface_pipeline(root/'reconstructed',root/'out',self.s.SurfaceConfig(),
                                                 plot_units=0,reference_root=root/'reference')
            self.assertEqual(manifest['counts'],{'selected':2,'processed':1,'failed':1})
            self.assertIn('FileNotFoundError',manifest['failures'][0]['reason'])
            unit=manifest['units'][0]
            self.assertTrue(unit['reference_file'].endswith('reference/D/SU_1.las'))
            self.assertAlmostEqual(unit['input_reference']['ref_dz_rmse'],0.,places=6)
            for row in unit['methods']:
                self.assertTrue(all(np.isfinite(row[key]) for key in self.s.REFERENCE_FIELDS),row['method'])
            with (root/'out/summary.csv').open() as stream:
                self.assertTrue(set(self.s.REFERENCE_FIELDS)<=set(stream.readline().strip().split(',')))

    def test_simple_notebook_selects_suffix_method_and_parameters(self):
        from contextlib import redirect_stdout
        import io, json
        from unittest.mock import patch
        notebook = json.loads((FOLDER / 'surface_extraction.ipynb').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / 'input'
            (input_dir / 'D').mkdir(parents=True)
            x, y = np.meshgrid(np.arange(12.) * .5, np.arange(12.) * .5)
            x, y = x.ravel(), y.ravel()
            header = laspy.LasHeader(point_format=6, version='1.4')
            header.scales = [.001]*3
            cloud = laspy.LasData(header)
            cloud.x, cloud.y, cloud.z = x+700000, y+2000000, 100+.5*x
            cloud.write(input_dir / 'D/SU_1_first_predictions.las')
            cloud.write(input_dir / 'D/SU_1_representative.las')
            choices = [('voxel', None), ('mls', None), ('local_plane', None)]
            choices += [('xy_grid', name) for name in ('mean', 'median', 'minimum', 'maximum', 'percentile')]
            for kind, statistic in choices:
                with self.subTest(kind=kind, statistic=statistic):
                    namespace = {'__name__': '__notebook_test__'}
                    method = 'xy_' + statistic if statistic else kind
                    output = root / method
                    with patch.object(Path, 'cwd', return_value=FOLDER), redirect_stdout(io.StringIO()):
                        for cell in notebook['cells']:
                            if cell['cell_type'] != 'code':
                                continue
                            exec(compile(''.join(cell['source']), cell['id'], 'exec'), namespace)
                            if cell['id'] == 'settings':
                                namespace.update(INPUT_DIR=input_dir, OUTPUT_DIR=output, SURFACE_TYPE=kind,
                                                 FILE_SUFFIX='_first_predictions.las')
                                namespace['VOXEL']['size'] = 100.
                                namespace['XY_GRID'].update(statistic=statistic, size=100., percentile=25.)
                    manifest = namespace['manifest']
                    self.assertEqual(manifest['counts'], {'selected': 1, 'processed': 1, 'failed': 0})
                    self.assertEqual(manifest['methods'], [method])
                    rows = manifest['units'][0]['methods']
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]['method'], method)
                    self.assertEqual(rows[0]['source_file'], 'D/SU_1_first_predictions.las')
                    self.assertEqual(rows[0]['las_file'], 'D/SU_1_first_predictions_surface.las')
                    self.assertEqual(rows[0]['npz_file'], 'D/SU_1_first_predictions_surface.npz')
                    exported = laspy.read(output / rows[0]['las_file'])
                    if kind in ('voxel', 'xy_grid'):
                        self.assertEqual(len(exported.points), 1)
                    else:
                        self.assertEqual(len(exported.points), len(x))
                    if statistic == 'percentile':
                        # LAS rounds coordinates to the source's 0.001-unit scale.
                        self.assertAlmostEqual(exported.z[0], np.percentile(cloud.z, 25.), delta=.00051)
                    self.assertEqual(len(list(output.rglob('*.las'))), 1)
                    self.assertTrue((output / 'summary.csv').is_file())
                    self.assertTrue((output / 'failures.json').is_file())

    def test_notebook_run_refreshes_cached_helper(self):
        import json
        from unittest.mock import patch
        notebook = json.loads((FOLDER / 'surface_extraction.ipynb').read_text())
        namespace = {'__name__': '__notebook_test__'}
        for cell in notebook['cells']:
            if cell['id'] in ('imports', 'settings'):
                exec(''.join(cell['source']), namespace)
        run = next(''.join(cell['source']) for cell in notebook['cells'] if cell['id'] == 'run')
        # A long-lived kernel can retain the pre-method-selection signature.
        def old_pipeline(input_path, output_dir, config=None, *, source_glob=None, plot_units=0):
            raise AssertionError('Cached helper was called')
        with tempfile.TemporaryDirectory() as temporary:
            namespace.update(INPUT_DIR=Path(temporary), OUTPUT_DIR=Path(temporary) / 'out')
            with patch.object(self.s, 'run_surface_pipeline', old_pipeline):
                # The refreshed pipeline reaches file discovery, not a signature error.
                with self.assertRaisesRegex(ValueError, 'No matching LAS/LAZ'):
                    exec(run, namespace)


if __name__=='__main__': unittest.main()
