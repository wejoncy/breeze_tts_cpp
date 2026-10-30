"""Dependency guards for the CPU boundary and leaf execution modules."""

import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PUBLIC = ROOT / "include"


def native_sources():
    return [p for p in SRC.rglob("*") if p.suffix in (".h", ".cuh", ".cpp", ".cu")]


def read_source(name):
    paths = [*SRC.rglob(name), *PUBLIC.rglob(name)]
    if len(paths) != 1:
        raise ValueError(f"Expected one source for {name}, found {paths}")
    return paths[0].read_text()


class ArchitectureTests(unittest.TestCase):
    def test_native_include_graph_is_acyclic(self):
        files = {
            p.relative_to(SRC).as_posix(): p for p in native_sources()
        }
        files.update({p.relative_to(PUBLIC).as_posix(): p for p in PUBLIC.rglob("*.h")})
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
            "request_planner.h",
            "request_executor.h",
            "mailbox.h",
            "safetensors.h",
            "mapped_file.h",
            "graph.h",
            "sampling.h",
            "audio_kernels.h",
            "kv_cache.h",
            "execution_resources.h",
        )
        for name in leaves:
            content = read_source(name)
            self.assertNotRegex(
                content, r'#include "(?:[^"]+/)?(?:engine|model_runner|model|backend)\.h"', name
            )
        for name in (
            "engine.h", "prepared_batch.h", "request_planner.h", "request_executor.h",
            "safetensors.h", "mapped_file.h",
        ):
            self.assertNotRegex(
                read_source(name), r"cuda|sys/mman|windows\.h", name
            )
        for name in ("graph.h", "execution_resources.h"):
            self.assertNotRegex(
                read_source(name),
                r'#include "(?:[^"]+/)?(?:request|generation_state|model_config)\.h"|\bGenerationOptions\b',
                name,
            )
        backend = (ROOT / "src/backends/cuda/backend.h").read_text()
        self.assertNotRegex(
            backend, r"\b(?:sample|audio_embedding|rvq|snake|conv_accumulate|pcm)\("
        )
        self.assertNotRegex((PUBLIC / "tts/engine.h").read_text(), r"\bfriend\b")

    def test_request_state_does_not_own_execution_resources(self):
        runner = (ROOT / "src/execution/model_runner.cpp").read_text()
        state = runner.split("struct ModelRunner::RequestState::Impl {", 1)[1].split(
            "\n};", 1
        )[0]
        self.assertNotRegex(
            state, r"\b(?:Graph|ExecutionResources|DeviceArena|GenerationState)\b"
        )
        self.assertNotIn("ModelRunner::Session", runner)
        engine = (ROOT / "src/engine/engine.cpp").read_text()
        self.assertNotIn("model.backbone", engine)
        self.assertNotIn("reserve_attention_workspace", engine)
        resources = (ROOT / "src/runtime/execution_resources.h").read_text()
        self.assertNotIn("PagedKVCache", resources)
        scheduler = (ROOT / "src/engine/request_scheduler.cpp").read_text()
        self.assertNotRegex(scheduler, r'#include "(?:[^"]+/)?(?:model_runner|model|backend|device)\.h"')
        self.assertNotIn("execution_mutex", scheduler)
        self.assertNotRegex(
            scheduler, r"std::(?:mutex|condition_variable|lock_guard|unique_lock)"
        )
        for path in (SRC / "engine").rglob("*"):
            if path.suffix in (".h", ".cpp"):
                self.assertNotRegex(path.read_text(), r"std::atomic|#include <atomic>", str(path))
        messages = read_source("request_executor.h")
        self.assertNotIn("shared_ptr", messages)
        self.assertIn("std::unique_ptr<RequestPlan>", messages)

    def test_device_allocations_have_one_pool_entry(self):
        for folder in ("src", "bindings", "tests", "bench"):
            for path in (ROOT / folder).rglob("*"):
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
        for name in (
            "request_schema.py",
            "model_source.py",
            "serving.py",
            "tts_cpp.py",
        ):
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
        for path in native_sources():
            if path.suffix in (".h", ".cuh", ".cu", ".cpp"):
                self.assertNotIn("tvm/ffi", path.read_text(), str(path))
        for path in (ROOT / "python").glob("*.py"):
            self.assertNotIn("ctypes", path.read_text(), str(path))

    def test_logging_stays_out_of_execution_loops(self):
        for name in (
            "model_runner.cpp",
            "model.cpp",
            "graph.h",
            "device.h",
            "sampling.cu",
            "audio_kernels.cu",
            "backend.cu",
        ):
            source = read_source(name)
            if name == "model_runner.cpp":
                # Initialization progress is useful; frame execution never logs.
                before, constructor = source.split("ModelRunner::ModelRunner(", 1)
                _, after = constructor.split("void ModelRunner::initialize_graphs()", 1)
                source = before + after
            self.assertNotIn("TTS_LOG(", source, name)
        for path in native_sources():
            if path.suffix in (".cpp", ".cu", ".h", ".cuh"):
                self.assertNotRegex(
                    path.read_text(), r"std::(?:cerr|clog)|\bfprintf\s*\(", str(path)
                )
        for name in ("log.h", "log.cpp"):
            self.assertNotRegex(
                read_source(name),
                r'#include "(?:[^"]+/)?(?:engine|backend|model|device)\.h"',
            )

    def test_functional_directories_do_not_depend_upward(self):
        forbidden = {
            "core": ("runtime/", "backends/", "kernels/", "engine/", "execution/", "models/"),
            "runtime": ("backends/", "kernels/", "engine/", "execution/", "models/"),
            "kernels": ("engine/", "execution/", "models/"),
            "backends": ("engine/", "execution/", "models/"),
        }
        self.assertFalse(any(p.is_file() for p in SRC.iterdir()))
        for path in native_sources():
            owner = path.relative_to(SRC).parts[0]
            for dependency in re.findall(r'#include "([^"]+)"', path.read_text()):
                self.assertFalse(dependency.startswith(forbidden.get(owner, ())),
                                 f"{path.relative_to(SRC)} -> {dependency}")
        for path in PUBLIC.rglob("*.h"):
            for dependency in re.findall(r'#include "([^"]+)"', path.read_text()):
                self.assertTrue(dependency.startswith("tts/"), f"{path} -> {dependency}")


if __name__ == "__main__":
    unittest.main()
