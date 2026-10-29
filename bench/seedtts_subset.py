#!/usr/bin/env python3
"""Freeze a small Seed-TTS evaluation subset before generating any model output."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "zhaochenyang20/seed-tts-eval"  # Public mirror; source paper: arXiv:2406.02430.
REVISION = "8f5e1aa2a35d42f42e940074c1983358b9491f89"


def prepare(output):
    cases, metadata_hashes = [], {}
    for language in ("en", "zh"):

        def download(name, language=language):
            return Path(
                hf_hub_download(
                    REPO, f"{language}/{name}", repo_type="dataset", revision=REVISION
                )
            )

        metadata = download("meta.lst").read_bytes()
        metadata_hashes[language] = hashlib.sha256(metadata).hexdigest()
        rows = [line.split("|") for line in metadata.decode().splitlines()]
        rows.sort(key=lambda row: (len(row[3]), row[0]))
        used = set()
        for bucket, name in enumerate(("short", "medium", "long")):
            group = rows[len(rows) * bucket // 3 : len(rows) * (bucket + 1) // 3]
            group.sort(
                key=lambda row: hashlib.sha256(
                    f"20261027:{language}:{row[0]}".encode()
                ).digest()
            )
            selected = 0
            for row in group:
                case, ref_text, ref_path, text = row
                if ref_path in used:
                    continue
                path = download(ref_path)
                cases.append(
                    {
                        "case": f"{language}-{case}",
                        "language": language,
                        "expected": text,
                        "length_bucket": name,
                        "reference": {
                            "wav": str(path.resolve()),
                            "text": ref_text,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        },
                    }
                )
                used.add(ref_path)
                selected += 1
                if selected == 4:
                    break
            assert selected == 4
    result = {
        "source": "https://github.com/BytedanceSpeech/seed-tts-eval",
        "paper": "https://arxiv.org/abs/2406.02430",
        "mirror": REPO,
        "revision": REVISION,
        "metadata_sha256": metadata_hashes,
        "selection": "12/language; 4 per target-character-length tercile; SHA256 order "
        "with salt 20261027; distinct reference files within each language; "
        "no filtering by model output",
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(cases)} fixed cases to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
