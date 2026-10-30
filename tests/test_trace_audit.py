"""Trace contract checks, including false passes from graph-only exports."""

import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from audit_trace import audit


class TraceAuditTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript("""
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, demangledName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER, nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, copyKind INTEGER, bytes INTEGER);
            INSERT INTO NVTX_EVENTS VALUES (10, 20, 'tts.scheduler.prepare', NULL);
            INSERT INTO NVTX_EVENTS VALUES (20, 40, 'tts.batch.admission', NULL);
            INSERT INTO NVTX_EVENTS VALUES (50, 90, 'tts.batch.frame', NULL);
            INSERT INTO NVTX_EVENTS VALUES (60, 80, 'tts.batch.decode', NULL);
            INSERT INTO NVTX_EVENTS VALUES (100, 120, 'tts.reference_encoder', NULL);
        """)
        self.kernel(65, "residual_rms_quantize")

    def name(self, value):
        return self.db.execute(
            "INSERT INTO StringIds(value) VALUES (?)", (value,)
        ).lastrowid

    def kernel(self, time, name):
        self.db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
            (time, time + 1, self.name(name)),
        )

    def test_explicit_uploads_and_result_downloads(self):
        self.db.executescript("""
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (30, 1, 1000);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (55, 1, 128);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (79, 2, 4);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (89, 2, 3840);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (110, 1, 96000);
        """)
        result = audit(self.db)
        self.assertEqual(result["reference_encodings"], 1)
        self.assertFalse(result["unexpected_memcopies"])
        self.assertFalse(result["allocations_or_captures"])
        self.assertFalse(result["forbidden_kernels"])
        self.assertEqual(result["copies_by_kind"][1]["count"], 3)

    def test_graph_only_trace_cannot_pass(self):
        self.db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
        self.kernel(30, "prefill_kernel")
        with self.assertRaisesRegex(ValueError, "graph-trace=node"):
            audit(self.db)

    def test_resources_between_frames_and_in_reference_are_rejected(self):
        for table, time, name in (
            ("RUNTIME", 45, "cudaMallocFromPoolAsync"),
            ("RUNTIME", 85, "cudaGraphInstantiateWithFlags"),
            ("RUNTIME", 105, "cudaEventCreateWithFlags"),
            ("DRIVER", 110, "cuMemAlloc_v2"),
        ):
            self.db.execute(
                f"INSERT INTO CUPTI_ACTIVITY_KIND_{table} VALUES (?, ?)",
                (time, self.name(name)),
            )
        self.assertEqual(len(audit(self.db)["allocations_or_captures"]), 4)

    def test_loading_resources_outside_serving_are_allowed(self):
        self.db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?)",
            (5, self.name("cudaMallocFromPoolAsync")),
        )
        self.assertFalse(audit(self.db)["allocations_or_captures"])

    def test_tuning_materialization_and_device_copies_are_rejected(self):
        self.kernel(110, "cast_kernel")
        self.kernel(35, "initialize_quant_tuning")
        self.db.executescript("""
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (35, 8, 1024);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (70, 1, 128);
        """)
        result = audit(self.db)
        self.assertEqual(len(result["forbidden_kernels"]), 2)
        self.assertEqual(len(result["unexpected_memcopies"]), 2)


if __name__ == "__main__":
    unittest.main()
