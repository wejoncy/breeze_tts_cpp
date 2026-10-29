#!/usr/bin/env python3
"""Small paired Seed-TTS evaluation using the paper's ASR and speaker models.

Run in a separate evaluation environment. The engine has no scorer dependency.
Text normalization follows seed-tts-eval/run_wer.py; report every output, including
failures/truncation, with paired confidence intervals clustered by source case.
"""

import argparse
import gc
import json
import string
import sys
from pathlib import Path
from unittest.mock import patch

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio
from zhon.hanzi import punctuation


def units(text, language):
    for mark in punctuation + string.punctuation:
        if mark != "'":
            text = text.replace(mark, "")
    text = text.replace("  ", " ")
    return " ".join(text) if language == "zh" else text.lower()


def audio16(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"invalid audio: {path}")
    mono = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
    return torchaudio.functional.resample(mono, rate, 16000)


def asr(records, language, model_path):
    if language == "en":
        from scipy.signal import resample
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        model = (
            WhisperForConditionalGeneration.from_pretrained(
                model_path, local_files_only=True, dtype=torch.float32
            )
            .cuda()
            .eval()
        )
    else:
        import zhconv
        from funasr import AutoModel

        model = AutoModel(model=model_path, device="cuda:0", disable_update=True)
    for row in records:
        if row["language"] != language:
            continue
        if language == "en":
            audio, rate = sf.read(row["wav"])
            # Same Fourier resampling as the published English evaluator.
            if rate != 16000:
                audio = resample(audio, int(len(audio) * 16000 / rate))
            if len(audio) > 30 * 16000:
                raise ValueError(
                    "English output exceeds single-window ASR; use long-form scoring"
                )
            features = processor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                tokens = model.generate(
                    features.input_features.cuda(),
                    language="english",
                    task="transcribe",
                    do_sample=False,
                    max_new_tokens=440,
                )
            text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        else:
            # Supply 16 kHz PCM to avoid coupling the scorer to torchaudio's file backend.
            result = model.generate(
                input=audio16(row["wav"]).squeeze(0).numpy(), batch_size_s=300
            )
            text = zhconv.convert(result[0]["text"], "zh-cn")
        score = jiwer.process_words(
            units(row["expected"], language), units(text, language)
        )
        errors = score.substitutions + score.deletions + score.insertions
        row.update(
            transcription=text,
            errors=errors,
            units=score.hits + score.substitutions + score.deletions,
            error_rate=score.wer,
            substitutions=score.substitutions,
            deletions=score.deletions,
            insertions=score.insertions,
        )
        audio, _ = sf.read(row["wav"])
        row.update(
            peak=float(np.abs(audio).max()),
            clipping_fraction=float(np.mean(np.abs(audio) >= 0.999)),
        )
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "case",
                        "precision",
                        "seed",
                        "error_rate",
                        "transcription",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def similarity(records, suite, checkpoint):
    sys.path.insert(
        0, str(suite / "thirdparty/UniSpeech/downstreams/speaker_verification")
    )
    from models.ecapa_tdnn import ECAPA_TDNN_SMALL
    from s3prl.upstream.wavlm.hubconf import wavlm_large

    # Use the installed, pinned WavLM entry point. The upstream torch.hub entry
    # imports unrelated models that call a removed torchaudio.set_audio_backend.
    with patch("torch.hub.load", lambda repo, name: wavlm_large()):
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="wavlm_large", config_path=None
        )
    state = torch.load(checkpoint, map_location="cpu")["model"]
    incompatible = model.load_state_dict(state, strict=False)
    # The training-only classifier is absent from the embedding inference model.
    if incompatible.missing_keys or set(incompatible.unexpected_keys) - {
        "loss_calculator.projection.weight"
    }:
        raise ValueError(f"speaker checkpoint mismatch: {incompatible}")
    model.cuda().eval()
    references = {}

    @torch.inference_mode()
    def embedding(path):
        value = model(audio16(path).cuda()).float().flatten()
        return torch.nn.functional.normalize(value, dim=0).cpu()

    for row in records:
        path = row["reference"]["wav"]
        if path not in references:
            references[path] = embedding(path)
        row["speaker_similarity"] = float(
            torch.dot(embedding(row["wav"]), references[path])
        )
        print(
            row["precision"],
            row["case"],
            row["seed"],
            row["speaker_similarity"],
            flush=True,
        )


def summarize(records):
    result = {}
    for language in ("en", "zh"):
        group = [r for r in records if r["language"] == language]
        summary = {}
        for precision in ("bf16", "fp8"):
            rows = [r for r in group if r["precision"] == precision]
            summary[precision] = {
                "outputs": len(rows),
                "cases": len({r["case"] for r in rows}),
                "mean_error_rate": float(np.mean([r["error_rate"] for r in rows])),
                "errors": sum(r["errors"] for r in rows),
                "units": sum(r["units"] for r in rows),
                "speaker_similarity": float(
                    np.mean([r["speaker_similarity"] for r in rows])
                ),
                "truncated": sum(bool(r["stats"]["truncated"]) for r in rows),
                "clipped_outputs": sum(r["clipping_fraction"] > 0 for r in rows),
            }
        paired = {}
        for row in group:
            key = (row["case"], row["seed"])
            if row["precision"] in paired.setdefault(key, {}):
                raise ValueError(f"duplicate paired output: {key}")
            paired[key][row["precision"]] = row
        if any(set(pair) != {"bf16", "fp8"} for pair in paired.values()):
            raise ValueError("incomplete precision pairs")
        for metric in ("error_rate", "speaker_similarity"):
            by_case = {}
            for (case, seed), pair in paired.items():
                by_case.setdefault(case, []).append(
                    pair["fp8"][metric] - pair["bf16"][metric]
                )
            deltas = np.array([np.mean(v) for v in by_case.values()])
            rng = np.random.default_rng(20261027)
            bootstrap = rng.choice(deltas, (10000, len(deltas)), replace=True).mean(
                axis=1
            )
            summary[metric + "_fp8_minus_bf16"] = {
                "mean": float(deltas.mean()),
                "paired_case_bootstrap_95ci": np.quantile(
                    bootstrap, [0.025, 0.975]
                ).tolist(),
            }
        result[language] = summary
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("asr", "similarity", "summary"), required=True
    )
    parser.add_argument("--asr-models", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--speaker-checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "asr":
        records = json.loads((args.output / "speech.json").read_text())
        models = json.loads(args.asr_models.read_text())
        for language in ("en", "zh"):
            asr(records, language, models[language])
            (args.output / "seed-asr.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n"
            )
    elif args.stage == "similarity":
        records = json.loads((args.output / "seed-asr.json").read_text())
        similarity(records, args.suite, args.speaker_checkpoint)
        (args.output / "seed-scored.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n"
        )
    else:
        records = json.loads((args.output / "seed-scored.json").read_text())
        cases = json.loads((args.output / "subset.json").read_text())["cases"]
        seeds = json.loads((args.output / "run.json").read_text())["seeds"]
        expected = {
            (c["case"], seed, precision)
            for c in cases
            for seed in seeds
            for precision in ("bf16", "fp8")
        }
        actual = [(r["case"], r["seed"], r["precision"]) for r in records]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("scored outputs do not cover the fixed manifest and seeds")
        summary = summarize(records)
        (args.output / "seed-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2))
