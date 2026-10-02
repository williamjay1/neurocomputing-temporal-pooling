"""Meaningful tests of the portability layer, using temporary files only."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"reproducibility"))
from legacy_runtime import LegacyRuntime, digest
TEST_TMP = Path(__file__).resolve().parents[1]/"reproducibility/.temporary-tests"
TEST_TMP.mkdir(parents=True, exist_ok=True)


class RuntimeTests(unittest.TestCase):
    def test_historical_paths_and_same_source_bytes(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp); (root/"scripts").mkdir()
            script = root/"scripts/example.py"
            script.write_text("from pathlib import Path\nROOT=Path('D:/MLWork/Neurocomputing_TailTransfer')\n"
                              "def f(): return Path('D:/MLWork/Neurocomputing_TailTransfer/results/x.json')\n", encoding="utf-8")
            before = digest(script)
            runtime = LegacyRuntime(root, root/"generated")
            mod = runtime.load("example.py")
            self.assertEqual(mod.ROOT, root)
            self.assertEqual(mod.f(), root/"results/x.json")
            self.assertEqual(before, digest(script))
            self.assertEqual(runtime.loaded["scripts/example.py"], before)

    def test_nested_original_script_loader_is_relocated(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            root = Path(tmp); (root/"scripts").mkdir()
            (root/"scripts/inner.py").write_text("from pathlib import Path\nROOT=Path('D:/MLWork/Neurocomputing_TailTransfer')\n", encoding="utf-8")
            (root/"scripts/outer.py").write_text("from pathlib import Path\nimport importlib.util\n"
                "ROOT=Path('D:/MLWork/Neurocomputing_TailTransfer')\n"
                "def nested():\n"
                "    s=importlib.util.spec_from_file_location('inner',ROOT/'scripts/inner.py')\n"
                "    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.ROOT\n", encoding="utf-8")
            runtime = LegacyRuntime(root, root/"generated")
            self.assertEqual(runtime.load("outer.py").nested(), root)
            self.assertEqual(len(runtime.loaded), 2)


if __name__ == "__main__":
    unittest.main()
