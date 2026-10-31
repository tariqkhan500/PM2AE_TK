"""Dependency-free checks; run with python -m unittest discover -s tests -v."""
import json
from pathlib import Path
import tempfile
import unittest
import ast
import struct
import importlib.util
import sys
import io
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NOTEBOOK = Path(__file__).resolve().parents[1] / 'Point2MAE_training.ipynb'
HAS_DATA_ENVIRONMENT = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'torch', 'laspy'))


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(NOTEBOOK.exists(), 'Training notebook has not been created')
        self.nb = json.loads(NOTEBOOK.read_text())
        self.ns = {}
        cell = next(c for c in self.nb['cells'] if c.get('id') == 'manifest-functions')
        exec(''.join(cell['source']), self.ns)

    def test_interactive_settings_request_divisors_and_use_them_for_groups(self):
        source = ''.join(next(c for c in self.nb['cells'] if c['id'] == 'settings')['source'])
        prompts = []

        def answer(prompt):
            prompts.append(prompt)
            return '[4, 2, 4]' if 'divisors' in prompt.lower() else ''

        ns = {}
        with patch('builtins.input', side_effect=answer), redirect_stdout(io.StringIO()):
            exec(source, ns)
        self.assertTrue(any('divisors' in p.lower() for p in prompts),
                        'Settings still request fixed patch counts instead of divisors')
        self.assertFalse(any('Patch centres at each encoder scale' in p for p in prompts))
        self.assertNotIn('num_groups', ns['CFG'])
        self.assertEqual(ns['CFG']['group_divisors'], [4, 2, 4])
        helper = self.full_cloud_helpers()['hierarchy_for_cloud']
        groups, _ = helper(4096, ns['CFG']['group_divisors'],
                           ns['CFG']['group_sizes'], ns['CFG']['mask_ratio'])
        self.assertEqual(groups, [1024, 512, 128])

    def test_all_cells_compile_and_are_unexecuted(self):
        for cell in self.nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])

    def fixture(self, root):
        for name in ['all', 'head', 'other']:
            (root / name).mkdir()
        for district, number in [('A', i) for i in range(10)] + [('B', 1)]:
            p = root / 'all' / district / f'SU_{number}.las'
            p.parent.mkdir(exist_ok=True)
            p.touch()
        for name, number in [('head', 1), ('other', 2)]:
            p = root / name / 'A' / f'SU_{number}.laz'
            p.parent.mkdir()
            p.touch()
        (root / 'all' / 'A' / 'SU_3.las:Zone.Identifier').touch()
        return root / 'all', [root / 'head', root / 'other']

    def test_exclusions_district_identity_and_split_reproducibility(self):
        with tempfile.TemporaryDirectory() as d:
            all_dir, exclusions = self.fixture(Path(d))
            eligible, excluded, audit = self.ns['discover_eligible'](all_dir, exclusions)
            keys = {r['unit_key'] for r in eligible}
            self.assertEqual(len(keys), 9)
            self.assertNotIn('A/SU_1', keys)
            self.assertNotIn('A/SU_2', keys)
            self.assertIn('B/SU_1', keys)
            self.assertEqual(len(excluded), 2)
            split = self.ns['make_splits'](eligible, 0.6, 0.2, 7)
            self.assertEqual(split, self.ns['make_splits'](eligible, 0.6, 0.2, 7))
            seen = [r['unit_key'] for rows in split.values() for r in rows]
            self.assertEqual(set(seen), keys)
            self.assertEqual(len(seen), len(set(seen)))
            self.assertTrue(all(split.values()))

    def test_missing_exclusion_and_duplicate_units_fail(self):
        with tempfile.TemporaryDirectory() as d:
            all_dir, exclusions = self.fixture(Path(d))
            with self.assertRaises(FileNotFoundError):
                self.ns['discover_eligible'](all_dir, [Path(d) / 'missing'])
            (all_dir / 'A' / 'SU_0.laz').touch()
            with self.assertRaises(ValueError):
                self.ns['discover_eligible'](all_dir, exclusions)

    def test_invalid_split_sizes_fail(self):
        for rows, train, val in [([], .8, .1), ([{}] * 3, .9, .2), ([{}] * 3, 1, 0)]:
            with self.assertRaises(ValueError):
                self.ns['make_splits'](rows, train, val, 42)

    def test_patch_configuration_retains_masked_reconstruction_targets(self):
        cell = next(c for c in self.nb['cells'] if c['id'] == 'settings')
        source = ''.join(cell['source']).replace('INTERACTIVE = True', 'INTERACTIVE = False')
        tree = ast.parse(source)
        defaults = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'DEFAULTS' for t in n.targets))
        index = next(i for i, key in enumerate(defaults.keys) if key.value == 'mask_ratio')
        defaults.values[index] = ast.Constant(value=.5)
        with self.assertRaisesRegex(ValueError, 'masked reconstruction'):
            exec(compile(ast.fix_missing_locations(tree), 'settings-test', 'exec'), {})

    def test_preflight_records_ratio_groups_for_each_cloud(self):
        audit = self.full_cloud_helpers()['audit_full_clouds']
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for n in (7, 8, 2048, 61907):
                header = bytearray(227)
                header[:4] = b'LASF'
                header[24:26] = bytes([1, 2])
                struct.pack_into('<H', header, 94, 227)
                struct.pack_into('<I', header, 107, n)
                path = Path(d) / f'{n}.las'
                path.write_bytes(header)
                rows.append({'path': str(path)})
            ready, excluded, inventory = audit(rows, dict(minimum_cloud_points=8,
                group_divisors=[8, 2, 4], group_sizes=[8, 4, 4], mask_ratio=.8))
            self.assertEqual([r['point_count'] for r in ready], [8, 2048, 61907])
            self.assertEqual([r['point_count'] for r in excluded], [7])
            self.assertEqual([r['num_groups'] for r in inventory],
                             [[8, 4, 3], [256, 128, 32], [7739, 3870, 968]])

    def full_cloud_helpers(self):
        cell = next((c for c in self.nb['cells'] if c['id'] == 'full-cloud-helpers'), None)
        self.assertIsNotNone(cell, 'Full-cloud helpers have not been added')
        ns = {}
        exec(''.join(cell['source']), ns)
        return ns

    def test_full_cloud_hierarchy_fits_small_and_large_clouds(self):
        helper = self.full_cloud_helpers()['hierarchy_for_cloud']
        for n in (8, 16, 19, 255, 511, 512, 2048, 61907):
            groups, sizes = helper(n, [8, 2, 4], [16, 8, 8], .8)
            self.assertLessEqual(groups[0], n)
            available = n
            for groups_at_scale, size in zip(groups, sizes):
                self.assertLessEqual(groups_at_scale, available)
                self.assertLessEqual(size, available)
                self.assertGreaterEqual(size, 2)
                available = groups_at_scale
            self.assertGreaterEqual(groups[-1], 3)
            masked = int(.8 * groups[-1])
            self.assertGreater(masked, 0)
            self.assertLess((groups[-1] - masked) * sizes[-1] + 1, groups[-2])
        self.assertEqual(helper(61907, [8, 2, 4], [16, 8, 8], .8),
                         ([7739, 3870, 968], [16, 8, 8]))
        for n in (0, 1, 2, 7):
            with self.assertRaises(ValueError):
                helper(n, [8, 2, 4], [16, 8, 8], .8)

    def test_accumulation_averages_complete_and_partial_windows(self):
        helper = self.full_cloud_helpers()['accumulation_window_size']
        self.assertEqual([helper(i, 10, 4) for i in range(10)], [4]*8 + [2]*2)
        self.assertEqual([helper(i, 3, 8) for i in range(3)], [3]*3)
        for begin in (0, 4, 8):
            self.assertAlmostEqual(sum(1/helper(i, 10, 4) for i in range(begin, min(begin+4, 10))), 1)

    def test_las_header_counts_legacy_and_extended(self):
        read_count = self.full_cloud_helpers()['las_point_count']
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'SU_1.las'
            for minor, legacy, extended, expected in [(2, 61907, 0, 61907), (4, 0, 123456, 123456)]:
                header = bytearray(375)
                header[:4] = b'LASF'
                header[24:26] = bytes([1, minor])
                struct.pack_into('<H', header, 94, 375 if minor == 4 else 227)
                struct.pack_into('<I', header, 107, legacy)
                struct.pack_into('<Q', header, 247, extended)
                path.write_bytes(header)
                self.assertEqual(read_count(path), expected)
            path.write_bytes(b'invalid')
            with self.assertRaises(ValueError):
                read_count(path)

    def test_dataset_has_no_point_selection_or_padding(self):
        cell = next(c for c in self.nb['cells'] if c['id'] == 'dataset')
        tree = ast.parse(''.join(cell['source']))
        # Data must pass through normalization to tensor conversion without row selection.
        forbidden = {'choice', 'permutation', 'pad', 'repeat', 'tile', 'randperm'}
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & forbidden, f'Input resampling/padding remains: {calls & forbidden}')
        text = ''.join(cell['source'])
        self.assertNotIn("CFG['num_points']", text)

    @unittest.skipUnless(HAS_DATA_ENVIRONMENT, 'NumPy, PyTorch and laspy installation is deferred')
    def test_real_las_loading_centres_xyz_without_rescaling_or_dropping_points(self):
        import numpy as np
        import torch
        import laspy
        from torch.utils.data import Dataset, DataLoader

        source = ''.join(next(c for c in self.nb['cells'] if c['id'] == 'dataset')['source'])
        tree = ast.parse(source)
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SlopeUnitDataset')
        ns = {'np': np, 'torch': torch, 'laspy': laspy, 'Dataset': Dataset}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), 'dataset-test', 'exec'), ns)
        with tempfile.TemporaryDirectory() as d:
            rows, expected = [], []
            for n in (8, 16, 2051):
                header = laspy.LasHeader(point_format=3, version='1.2')
                header.scales = np.array([.001, .001, .001])
                cloud = laspy.LasData(header)
                # Centroid subtraction also works for a vertical cloud (zero XY extent).
                cloud.x = 1000000 + (np.zeros(n) if n == 8 else np.arange(n) * .01)
                cloud.y = 2000000 + (np.zeros(n) if n == 8 else np.sin(np.arange(n)))
                cloud.z = 100 + np.arange(n) * .03
                path = Path(d) / f'SU_{n}.las'
                cloud.write(path)
                rows.append({'path': str(path), 'point_count': n})
                xyz = laspy.read(path).xyz
                xyz -= xyz.mean(axis=0, keepdims=True)
                expected.append(xyz.astype(np.float32))
            dataset = ns['SlopeUnitDataset'](rows)
            batches = list(DataLoader(dataset, batch_size=1))
            self.assertEqual([tuple(b.shape) for b in batches], [(1, 8, 3), (1, 16, 3), (1, 2051, 3)])
            for batch, points in zip(batches, expected):
                np.testing.assert_allclose(batch[0].numpy(), points, rtol=1e-6, atol=1e-7)
            rows[0]['point_count'] += 1
            with self.assertRaisesRegex(RuntimeError, 'differs from the audited cloud'):
                dataset[0]


if __name__ == '__main__':
    unittest.main()
