import unittest

from utils.dataset import normalize_dataset_paths_and_repeats


class MultiDatasetConfigTest(unittest.TestCase):
    def test_single_path_remains_backward_compatible(self):
        self.assertEqual(
            normalize_dataset_paths_and_repeats("/data/a", 3),
            (["/data/a"], [3]),
        )

    def test_scalar_repeat_applies_to_all_paths(self):
        self.assertEqual(
            normalize_dataset_paths_and_repeats(["/data/a", "/data/b"], 3),
            (["/data/a", "/data/b"], [3, 3]),
        )

    def test_independent_repeats_follow_path_order(self):
        self.assertEqual(
            normalize_dataset_paths_and_repeats(["/data/a", "/data/b"], [100, 4]),
            (["/data/a", "/data/b"], [100, 4]),
        )

    def test_list_like_values_are_supported(self):
        class ListLike:
            def __init__(self, values):
                self.values = values

            def __iter__(self):
                return iter(self.values)

        self.assertEqual(
            normalize_dataset_paths_and_repeats(
                ListLike(["/data/a", "/data/b"]), ListLike([100, 4])
            ),
            (["/data/a", "/data/b"], [100, 4]),
        )

    def test_repeat_count_must_match_path_count(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            normalize_dataset_paths_and_repeats(["/data/a", "/data/b"], [1])


if __name__ == "__main__":
    unittest.main()
