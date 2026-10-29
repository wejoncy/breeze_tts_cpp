"""Dependency guards for the CPU boundary and leaf execution modules."""

import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_native_include_graph_is_acyclic(self):
        files = {
            p.name: p
            for p in (ROOT / "src").iterdir()
            if p.suffix in (".h", ".cuh", ".cpp", ".cu")
        }
        edges = {
            name: [
                s
                for s in re.findall(r'#include "([^"]+)"', p.read_text())
                if s in files
            ]
            for name, p in files.items()
        }
        visited = set()

        def visit(name, stack):
            self.assertNotIn(name, stack, " -> ".join([*stack, name]))
            if name in visited:
                return
            for dep in edges[name]:
                visit(dep, [*stack, name])
            visited.add(name)

        for name in files:
            visit(name, [])

    def test_leaf_modules_do_not_reach_upward(self):
        leaves = (
            "tensor.h",
            "request.h",
            "prepared_batch.h",
            "safetensors.h",
            "mapped_file.h",
            "graph.h",
            "sampling.h",
            "audio_kernels.h",
            "kv_cache.h",
            "execution_resources.h",
        )
        for name in leaves:
            content = (ROOT / "src" / name).read_text()
            self.assertNotRegex(
                content, r'#include "(?:engine|model_runner|model|backend)\.h"', name
            )
        for name in ("engine.h", "prepared_batch.h", "safetensors.h", "mapped_file.h"):
            self.assertNotRegex(
                (ROOT / "src" / name).read_text(), r"cuda|sys/mman|windows\.h", name
            )
        for name in ("graph.h", "execution_resources.h"):
            self.assertNotRegex(
                (ROOT / "src" / name).read_text(),
                r'#include "(?:request|generation_state|model_config)\.h"|\bGenerationOptions\b',
                name,
            )
        backend = (ROOT / "src/backend.h").read_text()
        self.assertNotRegex(
            backend, r"\b(?:sample|audio_embedding|rvq|snake|conv_accumulate|pcm)\("
        )
        self.assertNotRegex((ROOT / "src/engine.h").read_text(), r"\bfriend\b")

    def test_device_allocations_have_one_pool_entry(self):
        for folder in ("src", "tests", "bench"):
            for path in (ROOT / folder).iterdir():
                if path.suffix not in (".h", ".cuh", ".cu", ".cpp"):
                    continue
                source = path.read_text()
                self.assertNotRegex(
                    source,
                    r"\b(?:cudaMalloc(?!FromPoolAsync)\w*|cudaFree|cuMemAlloc\w*|cudaHostAlloc)\s*\(",
                    str(path),
                )
                if path.name != "device.h":
                    self.assertNotRegex(
                        source,
                        r"\b(?:cudaMallocFromPoolAsync|cudaMemPoolCreate|cudaFreeAsync)\s*\(",
                        str(path),
                    )

    def test_python_layers_are_independent(self):
        for name in ("request_schema.py", "model_source.py", "tts_cpp.py"):
            tree = ast.parse((ROOT / "python" / name).read_text())
            imports = [
                n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
            ]
            imports += [
                a.name
                for n in ast.walk(tree)
                if isinstance(n, ast.Import)
                for a in n.names
            ]
            self.assertFalse(
                any(s == "server" or s.startswith("http") for s in imports), name
            )
            if name != "tts_cpp.py":
                self.assertNotIn("tts_cpp", imports, name)

    def test_ffi_stays_at_the_boundary(self):
        for path in (ROOT / "src").iterdir():
            if path.suffix in (".h", ".cuh", ".cu", ".cpp"):
                self.assertNotIn("tvm/ffi", path.read_text(), str(path))
        for path in (ROOT / "python").glob("*.py"):
            self.assertNotIn("ctypes", path.read_text(), str(path))

    def test_logging_stays_out_of_execution_loops(self):
        for name in ("model_runner.cpp", "model.cpp", "graph.h", "device.h",
                     "sampling.cu", "audio_kernels.cu", "backend.cu"):
            self.assertNotIn("TTS_LOG(", (ROOT / "src" / name).read_text(), name)
        for path in (ROOT / "src").iterdir():
            if path.suffix in (".cpp", ".cu", ".h", ".cuh"):
                self.assertNotRegex(path.read_text(), r"std::(?:cerr|clog)|\bfprintf\s*\(", str(path))
        for name in ("log.h", "log.cpp"):
            self.assertNotRegex((ROOT / "src" / name).read_text(),
                                r'#include "(?:engine|backend|model|device)\.h"')


if __name__ == "__main__":
    unittest.main()
