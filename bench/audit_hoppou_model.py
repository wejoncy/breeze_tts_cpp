"""Verify official F16 GGUF tensors against the local checkpoint and its converter."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import gguf
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repository", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--gguf", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repository / "scripts"))
import convert_hf_to_gguf as converter

reader = gguf.GGUFReader(str(args.gguf))
tensors = {t.name: t for t in reader.tensors}
types = Counter(t.tensor_type.name for t in reader.tensors)
if set(types) - {"F16", "F32"}:
    raise RuntimeError(f"Quantized tensor types present: {types}")
compared, mismatches = [], []


class AuditWriter:
    def add_tensor(self, name, data):
        compared.append(name)
        t = tensors.get(name)
        if t is None:
            mismatches.append(dict(name=name, error="absent from GGUF"))
            return
        actual = np.asarray(t.data).reshape(data.shape)
        if not np.array_equal(data, actual):
            diff = data.astype(np.float32) - actual.astype(np.float32)
            mismatches.append(
                dict(
                    name=name,
                    max_abs=float(np.max(np.abs(diff))),
                    rmse=float(np.sqrt(np.mean(diff * diff))),
                )
            )


writer = AuditWriter()
source = converter.ShardReader(args.checkpoint)
config = json.loads((args.checkpoint / "config.json").read_text())
converter.write_text_encoder(writer, source, config, "f16")
converter.write_backbone(writer, source, config, "f16")
converter.write_depth(writer, source, config, "f16")
converter.write_codec_all(writer, args.checkpoint, "f16")
result = dict(
    tensor_types=types,
    published_tensor_count=len(tensors),
    checked_tensor_count=len(compared),
    mismatches=mismatches,
    unchecked_tensors=sorted(set(tensors) - set(compared)),
)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
