"""Tests for layered configuration and the scrubbed executor environment."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchkit import config
from benchkit.executor import execute_with_feedback


class LayeredConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cwd = self.root / "project"
        self.cwd.mkdir()
        self.xdg = self.root / "xdg"
        (self.xdg / "benchkit").mkdir(parents=True)
        self.environ: dict[str, str] = {"XDG_CONFIG_HOME": str(self.xdg)}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, path: Path, text: str) -> Path:
        path.write_text(text)
        return path

    def _load(self, env_file: str | None = None) -> config.Loaded:
        return config.load(env_file, environ=self.environ, cwd=self.cwd)

    def test_user_config_fills_unset_keys(self) -> None:
        user = self._write(
            self.xdg / "benchkit" / ".env", "BENCHKIT_HOST=http://user:1\n"
        )
        loaded = self._load()
        self.assertEqual(self.environ["BENCHKIT_HOST"], "http://user:1")
        self.assertEqual(loaded.origins["BENCHKIT_HOST"], str(user))

    def test_xdg_unset_falls_back_to_dot_config(self) -> None:
        with patch.dict(os.environ, {"HOME": str(self.root)}):
            self.assertEqual(
                config.user_env_path({}), self.root / ".config" / "benchkit" / ".env"
            )

    def test_precedence_env_over_project_over_user(self) -> None:
        self._write(
            self.xdg / "benchkit" / ".env",
            "BENCHKIT_HOST=http://user\nBENCHKIT_TIMEOUT=10\nBENCHKIT_PROVIDER=ollama\n",
        )
        project = self._write(
            self.cwd / ".env", "BENCHKIT_HOST=http://project\nBENCHKIT_TIMEOUT=20\n"
        )
        self.environ["BENCHKIT_HOST"] = "http://real"
        loaded = self._load()
        self.assertEqual(self.environ["BENCHKIT_HOST"], "http://real")
        self.assertEqual(loaded.origins["BENCHKIT_HOST"], "env")
        self.assertEqual(self.environ["BENCHKIT_TIMEOUT"], "20")
        self.assertEqual(loaded.origins["BENCHKIT_TIMEOUT"], str(project))
        self.assertEqual(self.environ["BENCHKIT_PROVIDER"], "ollama")

    def test_env_file_replaces_project_env(self) -> None:
        self._write(self.cwd / ".env", "BENCHKIT_HOST=http://project\n")
        self._write(self.xdg / "benchkit" / ".env", "BENCHKIT_TIMEOUT=10\n")
        explicit = self._write(self.root / "ci.env", "BENCHKIT_PROVIDER=openai\n")
        loaded = self._load(str(explicit))
        self.assertNotIn("BENCHKIT_HOST", self.environ)
        self.assertEqual(self.environ["BENCHKIT_PROVIDER"], "openai")
        self.assertEqual(self.environ["BENCHKIT_TIMEOUT"], "10")
        roles = {entry.role: entry for entry in loaded.files}
        self.assertTrue(roles["project"].skipped)

    def test_env_file_variable_and_relative_path(self) -> None:
        self._write(self.cwd / "cron.env", "BENCHKIT_HOST=http://cron\n")
        self.environ[config.ENV_FILE_VAR] = "cron.env"
        self._load()
        self.assertEqual(self.environ["BENCHKIT_HOST"], "http://cron")

    def test_flag_beats_env_file_variable(self) -> None:
        self._write(self.root / "a.env", "BENCHKIT_HOST=http://a\n")
        self._write(self.root / "b.env", "BENCHKIT_HOST=http://b\n")
        self.environ[config.ENV_FILE_VAR] = str(self.root / "a.env")
        self._load(str(self.root / "b.env"))
        self.assertEqual(self.environ["BENCHKIT_HOST"], "http://b")

    def test_missing_env_file_is_an_error(self) -> None:
        with self.assertRaises(config.ConfigError):
            self._load(str(self.root / "nope.env"))

    def test_only_benchkit_keys_are_applied(self) -> None:
        self._write(
            self.cwd / ".env",
            "AWS_SECRET_ACCESS_KEY=leak\nDATABASE_URL=postgres://x\n"
            "OPENAI_API_KEY=sk-ok\nOLLAMA_HOST=http://o\nEVALPLUS_TIMEOUT_PER_TASK=5\n"
            "BENCHKIT_ENV_FILE=/elsewhere\n",
        )
        self._load()
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", self.environ)
        self.assertNotIn("DATABASE_URL", self.environ)
        self.assertNotIn(config.ENV_FILE_VAR, self.environ)
        self.assertEqual(self.environ["OPENAI_API_KEY"], "sk-ok")
        self.assertEqual(self.environ["OLLAMA_HOST"], "http://o")
        self.assertEqual(self.environ["EVALPLUS_TIMEOUT_PER_TASK"], "5")

    def test_parent_directory_env_is_ignored(self) -> None:
        self._write(self.root / ".env", "BENCHKIT_HOST=http://parent\n")
        self._load()
        self.assertNotIn("BENCHKIT_HOST", self.environ)

    def test_resolve_reports_origins_and_masks_secrets(self) -> None:
        user = self._write(
            self.xdg / "benchkit" / ".env",
            "BENCHKIT_API_KEY=sk-abcdefghijkl1234\nBENCHKIT_CUSTOM=1\n",
        )
        self.environ["BENCHKIT_TIMEOUT"] = "42"
        self._load()
        rows = {
            key: (value, origin)
            for key, value, origin in config.resolve(
                {"BENCHKIT_HOST": "http://flag"}, environ=self.environ
            )
        }
        self.assertEqual(rows["BENCHKIT_HOST"], ("http://flag", "flag"))
        self.assertEqual(rows["BENCHKIT_TIMEOUT"], ("42", "env"))
        self.assertEqual(rows["BENCHKIT_API_KEY"], ("****1234", str(user)))
        self.assertEqual(rows["BENCHKIT_CUSTOM"], ("1", str(user)))
        self.assertEqual(rows["BENCHKIT_PROVIDER"], ("auto", "default"))

    def test_mask(self) -> None:
        self.assertEqual(config.mask(""), "")
        self.assertEqual(config.mask("short"), "****")
        self.assertEqual(config.mask("sk-1234567890"), "****7890")
        self.assertTrue(config.is_secret("OPENAI_API_KEY"))
        self.assertFalse(config.is_secret("BENCHKIT_HOST"))

    def test_save_user_writes_private_file_and_round_trips(self) -> None:
        path = config.save_user(
            {"BENCHKIT_HOST": "http://saved:8080/v1", "BENCHKIT_API_KEY": "sk-x y"},
            environ=self.environ,
        )
        self.assertEqual(path, self.xdg / "benchkit" / ".env")
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        config.save_user({"BENCHKIT_API_KEY": ""}, environ=self.environ)
        self._load()
        self.assertEqual(self.environ["BENCHKIT_HOST"], "http://saved:8080/v1")
        self.assertNotIn("BENCHKIT_API_KEY", self.environ)

    def test_save_user_rejects_foreign_keys(self) -> None:
        with self.assertRaises(ValueError):
            config.save_user({"PATH": "/x"}, environ=self.environ)


class ExecutorEnvironmentTests(unittest.TestCase):
    def test_generated_code_cannot_read_api_keys(self) -> None:
        code = (
            "import os\n"
            "assert 'BENCHKIT_API_KEY' not in os.environ\n"
            "assert 'OPENAI_API_KEY' not in os.environ\n"
            "home = os.path.realpath(os.environ['HOME'])\n"
            "assert home == os.path.realpath(os.getcwd())\n"
        )
        with patch.dict(
            os.environ, {"BENCHKIT_API_KEY": "sk-secret", "OPENAI_API_KEY": "sk-2"}
        ):
            result = execute_with_feedback(code)
        self.assertTrue(result.passed, result.feedback)

    def test_evalplus_child_gets_scrubbed_environment(self) -> None:
        import multiprocessing

        from benchkit.benchmarks.evalplus import _fork_multiprocessing

        def probe(flag) -> None:
            flag.value = int(
                "BENCHKIT_API_KEY" not in os.environ
                and os.environ.get("EVALPLUS_MAX_MEMORY_BYTES") == "-1"
            )

        with patch.dict(
            os.environ,
            {"BENCHKIT_API_KEY": "sk-secret", "EVALPLUS_MAX_MEMORY_BYTES": "-1"},
        ):
            with _fork_multiprocessing():
                flag = multiprocessing.Value("i", 0)
                process = multiprocessing.Process(target=probe, args=(flag,))
                process.start()
                process.join(30)
            self.assertEqual(os.environ["BENCHKIT_API_KEY"], "sk-secret")
        self.assertEqual(flag.value, 1)


if __name__ == "__main__":
    unittest.main()
