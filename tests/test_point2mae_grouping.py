"""Grouping policy tests run without CUDA; GPU integration lives in the notebook."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('point2mae_grouping'),
                             'Reusable grouping module is missing')
        import point2mae_grouping
        self.grouping = point2mae_grouping

    def test_successive_ratios_round_up_and_have_no_fixed_cap(self):
        for n, divisors, expected in [
            (2048, [4, 2, 4], [512, 256, 64]),
            (2049, [4, 2, 4], [513, 257, 65]),
            (2048, [8, 2, 4], [256, 128, 32]),
            (61907, [8, 2, 4], [7739, 3870, 968]),
        ]:
            with self.subTest(n=n, divisors=divisors):
                self.assertEqual(self.grouping.hierarchy_for_cloud(n, divisors, [8, 4, 4], .8),
                                 (expected, [8, 4, 4]))

    def test_small_clouds_fit_and_always_retain_reconstruction_targets(self):
        for n in range(8, 1025):
            groups, sizes = self.grouping.hierarchy_for_cloud(n, [8, 2, 4], [16, 8, 8], .8)
            available = n
            for count, size in zip(groups, sizes):
                self.assertLessEqual(count, available)
                self.assertLessEqual(size, available)
                self.assertGreaterEqual(size, 2)
                available = count
            self.assertGreaterEqual(groups[-1], 3)
            masked = int(.8 * groups[-1])
            self.assertGreater(masked, 0)
            self.assertLess(masked, groups[-1])
            self.assertLess((groups[-1] - masked) * sizes[-1] + 1, groups[-2])

    def test_invalid_counts_divisors_sizes_and_masks(self):
        helper = self.grouping.hierarchy_for_cloud
        for n in (0, 1, 7, True, 8.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                helper(n, [8, 2, 4], [8, 4, 4], .8)
        for divisors in ([0, 2, 4], [-1, 2, 4], [2.5, 2, 4], [True, 2, 4], [2, 4], None):
            with self.subTest(divisors=divisors), self.assertRaises(ValueError):
                helper(2048, divisors, [8, 4, 4], .8)
        for sizes in ([8, 1, 4], [8, 4], [8, 4, False], None):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                helper(2048, [8, 2, 4], sizes, .8)
        for ratio in (0, 1, -.1, float('nan'), float('inf'), True, '0.8'):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                helper(2048, [8, 2, 4], [8, 4, 4], ratio)
        with self.assertRaisesRegex(ValueError, 'masked reconstruction'):
            helper(8, [8, 2, 4], [8, 4, 4], .5)

    def test_divider_settings_follow_each_cloud_and_restore_knn_sizes(self):
        # Only replace the unavailable native KNN constructor, not the adapter.
        dividers = [SimpleNamespace(num_group=1, group_size=8, knn=object()) for _ in range(3)]
        fake_knn = SimpleNamespace(KNN=lambda k, transpose_mode: SimpleNamespace(k=k))
        with patch.dict(sys.modules, knn_cuda=fake_knn):
            for n, groups, sizes in [(8, [8, 4, 3], [8, 4, 2]),
                                     (61907, [7739, 3870, 968], [8, 4, 4]),
                                     (8, [8, 4, 3], [8, 4, 2])]:
                self.grouping.configure_group_dividers(dividers, n, [8, 2, 4], [8, 4, 4], .8)
                self.assertEqual([d.num_group for d in dividers], groups)
                self.assertEqual([d.group_size for d in dividers], sizes)
                self.assertEqual(dividers[-1].knn.k, sizes[-1])
            before = [(d.num_group, d.group_size, d.knn) for d in dividers]
            with self.assertRaises(ValueError):
                self.grouping.configure_group_dividers(dividers, 7, [8, 2, 4], [8, 4, 4], .8)
            self.assertEqual(before, [(d.num_group, d.group_size, d.knn) for d in dividers])
            with self.assertRaises(ValueError):
                self.grouping.configure_group_dividers(dividers[:2], 2048, [8, 2, 4], [8, 4, 4], .8)


if __name__ == '__main__':
    unittest.main()
