import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from utils.dataset import detect_camera_lmdb_paths, normalize_dataset_paths_and_repeats


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

    def test_detect_camera_lmdb_paths_accepts_omegaconf_path_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "data.mdb").touch()
            (second / "data.mdb").touch()

            configured_paths = OmegaConf.create([str(first), str(second)])
            paths, flags = detect_camera_lmdb_paths(configured_paths)

            self.assertEqual(paths, [str(first), str(second)])
            self.assertEqual(flags, [True, True])


if __name__ == "__main__":
    unittest.main()
