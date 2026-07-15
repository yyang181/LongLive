import unittest

from omegaconf import OmegaConf

from utils.config import normalize_config, section_get


class SectionGetTest(unittest.TestCase):
    def test_none_key_returns_complete_section(self):
        config = normalize_config(OmegaConf.create({
            "inference": {
                "action_overlay": True,
                "overlay_corner": "bottom-left",
            },
        }))

        inference = section_get(config, "inference", None, None)

        self.assertIsNotNone(inference)
        self.assertTrue(inference.action_overlay)
        self.assertEqual(inference.overlay_corner, "bottom-left")

    def test_none_key_uses_default_for_missing_section(self):
        default = {"action_overlay": False}

        self.assertIs(
            section_get(OmegaConf.create({}), "inference", None, default),
            default,
        )


if __name__ == "__main__":
    unittest.main()
