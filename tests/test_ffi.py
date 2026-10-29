"""Boundary tests for reflected options and managed handles; no GPU required."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from native_library import load_library
from request_schema import validate
from tts_cpp import Engine, Options


class FFIFieldsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_library()

    def test_full_unsigned_seed_and_independent_options(self):
        for seed in (0, 42, 2**63 - 1, 2**63, 2**64 - 1):
            _, options = validate(
                dict(text="Hi", seed=seed), False, self.module.default_options()
            )
            self.assertIsInstance(options, Options)
            self.assertEqual(options.seed, seed)
            other = self.module.copy_options(options)
            other.seed = 7
            self.assertEqual(options.seed, seed)
            self.assertEqual(other.seed, 7)
        for seed in (-1, 2**64, True, 1.5):
            with self.assertRaises(ValueError):
                self.module.default_options().seed = seed

    def test_sampling_validation_and_float_overflow(self):
        for values in (
            dict(max_frames=2**40),
            dict(do_sample=2),
            dict(temperature=1e100),
            dict(depth_top_p=0),
            dict(temperature=float("nan")),
        ):
            with self.assertRaises(ValueError):
                validate(
                    dict(text="Hi", **values), False, self.module.default_options()
                )
        _, options = validate(
            dict(text="Hi", temperature=0.5, depth_temperature=0.7),
            False,
            self.module.default_options(),
        )
        self.assertAlmostEqual(options.temperature, 0.5)
        self.assertAlmostEqual(options.depth_temperature, 0.7)

    def test_reference_cpu_lifetime(self):
        # Importing reusable codes needs no model or device allocation.
        engine = Engine.__new__(Engine)
        engine._module = self.module
        engine._format = dict(
            sample_rate=16000, samples_per_frame=320, codebooks=4, codebook_size=32
        )
        reference = engine.reference_from_codes(list(range(4)), "Reference")
        self.assertEqual(reference.frames, 1)
        self.assertEqual(reference.codes(), list(range(4)))
        reference.close()
        reference.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            reference.codes()


if __name__ == "__main__":
    unittest.main()
