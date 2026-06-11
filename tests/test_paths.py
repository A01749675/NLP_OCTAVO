import os
import tempfile
import unittest
from pathlib import Path

from paths import resolve_input_path, resolve_output_path, resolve_model_path


class TestPaths(unittest.TestCase):

    def test_resolve_input_path_finds_file_in_files_dir(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                os.makedirs("files", exist_ok=True)
                file_path = Path("files") / "data.csv"
                file_path.write_text("col\n1\n")

                resolved = resolve_input_path("data.csv")
                self.assertEqual(resolved, str(file_path.resolve()))
            finally:
                os.chdir(cwd)

    def test_resolve_input_path_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "input.csv"
            file_path.write_text("col\n1\n")
            resolved = resolve_input_path(str(file_path))
            self.assertEqual(resolved, str(file_path))

    def test_resolve_output_path_creates_files_dir(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                resolved = resolve_output_path("out.csv")
                self.assertTrue(Path(resolved).parent.exists())
                self.assertTrue(str(Path(resolved)).endswith(os.path.join("files", "out.csv")))
            finally:
                os.chdir(cwd)

    def test_resolve_model_path_creates_model_files_dir(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                resolved = resolve_model_path("model.pkl")
                self.assertTrue(Path(resolved).parent.exists())
                self.assertTrue(str(Path(resolved)).endswith(os.path.join("model_files", "model.pkl")))
            finally:
                os.chdir(cwd)

    def test_none_returns_none(self):
        self.assertIsNone(resolve_input_path(None))
        self.assertIsNone(resolve_output_path(None))
        self.assertIsNone(resolve_model_path(None))
