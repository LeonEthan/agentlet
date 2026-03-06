"""Smoke tests for the installed package and CLI entrypoint."""

import importlib
import shutil
import subprocess
import unittest


class ImportSmokeTest(unittest.TestCase):
    def test_agentlet_package_imports(self) -> None:
        module = importlib.import_module("agentlet")
        self.assertEqual(module.__version__, "0.1.0")

    def test_agentlet_script_runs(self) -> None:
        script_path = shutil.which("agentlet")
        self.assertIsNotNone(script_path)

        completed = subprocess.run(
            [script_path, "--help"],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("usage: agentlet", completed.stdout)
