"""Build metadata resolution and scoped Windows DLL search lifetime."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import native_library as native


class NativeLibraryTests(unittest.TestCase):
    def setUp(self):
        native._loaded.clear()
        self.addCleanup(native._loaded.clear)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "工程"
        self.root.mkdir()
        self.binary = self.root / "any-native-artifact"
        self.binary.touch()
        self.manifest = self.root / "tts-runtime.json"
        self.manifest.write_text(
            json.dumps({"library": self.binary.name}), encoding="utf-8"
        )

    def test_explicit_artifact_directory_and_environment(self):
        for selected in (self.binary, self.root, self.manifest):
            self.assertEqual(native.library_path(selected), self.binary)
        with patch.dict(os.environ, {"TTS_LIBRARY": str(self.root)}):
            self.assertEqual(native.library_path(), self.binary)
            self.assertEqual(native.library_path(self.binary), self.binary)

    def test_multiple_builds_require_selection(self):
        project = self.root / "sdk"
        for config in ("Debug", "Release"):
            directory = project / "build" / config
            directory.mkdir(parents=True)
            (directory / "tts-runtime.json").write_text(
                json.dumps({"library": str(self.binary)})
            )
        with patch.object(
            native, "__file__", str(project / "python" / "native_library.py")
        ):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    native.library_path()
                direct = project / "build/tts-runtime.json"
                direct.write_text(json.dumps({"library": str(self.binary)}))
                self.assertEqual(native.library_path(), self.binary)

    def test_missing_binary_is_reported(self):
        self.binary.unlink()
        with self.assertRaises(FileNotFoundError):
            native.library_path(self.manifest)

    def test_windows_search_directories_survive_with_library(self):
        dependency = self.root / "依赖"
        dependency.mkdir()
        self.manifest.write_text(
            json.dumps(
                dict(
                    library=self.binary.name,
                    runtime_libraries=["依赖/some-library"],
                    runtime_directories=["依赖", "missing"],
                )
            ),
            encoding="utf-8",
        )
        handles = [Mock(), Mock()]
        register = Mock(side_effect=handles)
        windows = SimpleNamespace(name="nt", environ={}, add_dll_directory=register)
        loaded = SimpleNamespace()
        with (
            patch.object(native, "os", windows),
            patch.object(native.tvm_ffi, "load_module", return_value=loaded),
            patch.object(native, "register_types"),
        ):
            library = native.load_library(self.root)
        self.assertIs(library, loaded)
        self.assertEqual(native._loaded[self.binary][1], handles)
        self.assertEqual(
            [c.args[0] for c in register.call_args_list],
            [str(self.root), str(dependency)],
        )
        for handle in handles:
            handle.close.assert_not_called()

    def test_load_failure_closes_search_registration(self):
        handle = Mock()
        windows = SimpleNamespace(
            name="nt", environ={}, add_dll_directory=Mock(return_value=handle)
        )
        with (
            patch.object(native, "os", windows),
            patch.object(
                native.tvm_ffi, "load_module", side_effect=OSError("dependency missing")
            ),
        ):
            with self.assertRaises(OSError):
                native.load_library(self.root)
        handle.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
