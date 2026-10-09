import importlib.util
from pathlib import Path
import unittest


class ValidationSamplingTest(unittest.TestCase):
    def test_excludes_development_and_repeated_videos_with_stable_indices(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_scene_validation.py"
        spec = importlib.util.spec_from_file_location("prepare_scene_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [{"video": name} for name in ["dev.mp4", "dev.mp4", "new.mp4", "new.mp4", "other.mp4"]]
        development = [{"video": "/external/cache/dev.mp4"}]
        self.assertEqual(module.choose_indices(rows, development, 1, 2), [2, 4])
        with self.assertRaisesRegex(ValueError, "distinct unused"):
            module.choose_indices(rows, development, 1, 3)


if __name__ == "__main__":
    unittest.main()
