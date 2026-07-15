import unittest

from utils.dataset import RepeatDataset


class RepeatDatasetTest(unittest.TestCase):
    def test_repeats_indices_without_copying_samples(self):
        base = ["a", "b", "c"]
        repeated = RepeatDataset(base, repeats=3)
        self.assertEqual(len(repeated), 9)
        self.assertEqual([repeated[i] for i in range(len(repeated))],
                         ["a", "b", "c"] * 3)
        self.assertEqual(repeated[-1], "c")

    def test_invalid_repeat_values(self):
        with self.assertRaises(ValueError):
            RepeatDataset([1], repeats=0)
        with self.assertRaises(ValueError):
            RepeatDataset([], repeats=2)


if __name__ == "__main__":
    unittest.main()
