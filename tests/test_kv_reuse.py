"""Request admission and graph validity across KV/workspace shape changes."""

import argparse
from pathlib import Path
import sys
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--library", type=Path)
parser.add_argument("--model", type=Path, required=True)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from tts_cpp import Engine


class CacheReuseTests(unittest.TestCase):
    def test_capacity_is_a_limit_not_a_reservation(self):
        # The old worst-case reservation exceeds the entire H100 NVL at these
        # legal limits, even when the actual request contains a single short lane.
        with Engine(library=args.library, model=args.model, max_batch=32, context=8192) as engine:
            short = [dict(text="你好。")]
            options = dict(max_frames=4, temperature=0)
            original, _ = engine.generate(short, options)
            for requests, opts in [
                ([dict(text="Welcome aboard. " * n) for n in (1, 21, 97)], options),
                (
                    [dict(text="你好。", instruction="A calm voice.")] * 8,
                    dict(options, cfg_scale=3),
                ),
                (short, dict(options, max_frames=129)),
                (short, options),
            ]:
                graph, _ = engine.generate(requests, opts)
                eager, _ = engine.generate(requests, dict(opts, use_cuda_graph=0))
                self.assertEqual(graph, eager)
            with self.assertRaises(RuntimeError):
                engine.generate([dict(text="测试" * 10000)], options)
            after, _ = engine.generate(short, options)
            self.assertEqual(after, original)
            stats = engine.generate(short, options, lambda *_: False)
            self.assertTrue(stats["cancelled"])
            after, _ = engine.generate(short, options)
            self.assertEqual(after, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
