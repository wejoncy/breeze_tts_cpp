"""Offline conversion of official GPT-SoVITS v5 checkpoints to native weights.

PyTorch is used only by this explicit conversion tool, never by the C++ runtime.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from text_frontend import export_frontend
from english_linguistics import export_english
from export_linguistics import export_cantonese
from korean_linguistics import export_korean
from language_routing import export_router
from japanese_linguistics import export_japanese


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a tensor dictionary: {path.name}")
    state = checkpoint.get("weight", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"Invalid state dictionary: {path.name}")
    return checkpoint, state


def convert(state, dtype):
    result = {}
    for name, value in state.items():
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            base = name[:-len("weight_v")]
            gain = state[base + "weight_g"].float()
            value = value.float()
            # HuBERT positional convolutions normalize dim=2; acoustic WN uses dim=0.
            if gain.ndim != value.ndim:
                raise ValueError(f"Invalid weight-norm gain: {name}")
            retained = [i for i, size in enumerate(gain.shape) if size != 1]
            if len(retained) > 1 or any(gain.shape[i] != value.shape[i] for i in retained):
                raise ValueError(f"Invalid weight-norm dimensions: {name}")
            axes = tuple(i for i in range(value.ndim) if i not in retained)
            value = value * (gain / value.square().sum(axes, keepdim=True).sqrt())
            name = base + "weight"
        if not value.is_floating_point():
            continue  # Nonpersistent position IDs and training counters are not weights.
        result[name] = value.to(dtype).contiguous().clone()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("v5dev", "v5turbo"), required=True)
    parser.add_argument("--dtype", choices=("float16", "bf16"), default="float16")
    parser.add_argument("--text-source", type=Path, required=True, help="Official GPT-SoVITS source checkout")
    parser.add_argument("--g2pw-source", type=Path, required=True, help="Extracted official G2PWModel_1.1")
    parser.add_argument("--g2pw-tokenizer", type=Path, required=True, help="G2PW's BERT tokenizer.json")
    parser.add_argument("--english-g2p", type=Path, required=True, help="g2p_en package data directory")
    parser.add_argument("--english-segmenter", type=Path, required=True, help="wordsegment package data directory")
    parser.add_argument("--english-tagger", type=Path, required=True, help="NLTK averaged_perceptron_tagger_eng JSON directory")
    parser.add_argument("--cantonese-jyutping", type=Path, help="ToJyutping package directory; enables Cantonese")
    parser.add_argument("--korean-g2p", type=Path, help="g2pk2 package directory")
    parser.add_argument("--korean-dictionary", type=Path, help="UTF-8 MeCab-KO dictionary directory")
    parser.add_argument("--korean-jamo", type=Path, help="jamo package directory; enables Korean with the other two assets")
    parser.add_argument("--language-detector", type=Path, help="Official full FastText lid.176.bin classifier")
    parser.add_argument("--language-splitter", type=Path, help="split_lang package directory")
    parser.add_argument("--language-budoux", type=Path, help="BudouX package directory; enables auto routing with the other two assets")
    parser.add_argument("--japanese-dictionary", type=Path, help="UTF-8 Open JTalk dictionary directory")
    parser.add_argument("--japanese-user-dictionary", type=Path, help="Official GPT-SoVITS compiled Japanese user dictionary")
    args = parser.parse_args()
    korean = (args.korean_g2p, args.korean_dictionary, args.korean_jamo)
    if any(korean) and not all(korean):
        parser.error("Korean requires --korean-g2p, --korean-dictionary and --korean-jamo together")
    routing = (args.language_splitter, args.language_budoux, args.language_detector)
    if any(routing) and not all(routing):
        parser.error("Automatic routing requires --language-detector, --language-splitter and --language-budoux together")
    japanese = (args.japanese_dictionary, args.japanese_user_dictionary)
    if any(japanese) and not all(japanese):
        parser.error("Japanese requires --japanese-dictionary and --japanese-user-dictionary together")
    dtype = {"float16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    export_frontend(args.text_source, args.g2pw_source,
                    args.g2pw_tokenizer, output, dtype)
    export_english(args.text_source / "GPT_SoVITS/text", args.english_g2p,
                   args.english_segmenter, args.english_tagger, output)
    languages = {
        "zh": ["tokenizer.json", "g2pw_tokenizer.json", "g2pw.json", "jieba.dict.utf8", "jieba.hmm.utf8",
               "pinyin.bin", "polyphone.bin", "linguistics.json"],
        "en": ["english.json", "english.safetensors", "linguistics.json"],
    }
    mecab_files = ["char.bin", "matrix.bin", "sys.dic", "unk.dic", "dicrc"]
    if args.cantonese_jyutping:
        export_cantonese(args.text_source / "GPT_SoVITS/text", args.cantonese_jyutping, output)
        languages["yue"] = ["cantonese.bin", "pinyin.bin", "linguistics.json"]
    if all(korean):
        export_korean(args.text_source / "GPT_SoVITS/text", *korean, output)
        languages["ko"] = ["korean.json", "linguistics.json"] + ["korean_mecab/" + name for name in mecab_files]
    if all(routing):
        export_router(args.text_source / "GPT_SoVITS/text", *routing, output)
        languages["auto"] = ["language_policy.json", "language_detector.bin",
                             "language_splitter.json"]
    if all(japanese):
        export_japanese(args.text_source / "GPT_SoVITS/text", *japanese, output)
        languages["ja"] = ["japanese.json", "japanese_user.dic", "linguistics.json"] + ["japanese_mecab/" + name for name in mecab_files]
    sources = {
        "semantic": args.source / "s1v3.ckpt",
        "acoustic": args.source / "gsv-v5-pretrained" / f"s2G{args.variant}.pth",
        "vocoder": args.source / "gsv-v5-pretrained/vocoder.pth",
        "hubert": args.source / "chinese-hubert-base/pytorch_model.bin",
        "bert": args.source / "chinese-roberta-wwm-ext-large/pytorch_model.bin",
    }
    configs = {"frontend": {"languages": languages}}
    for component, path in sources.items():
        checkpoint, state = load(path)
        save_file({f"{component}.{name}": value for name, value in convert(state, dtype).items()},
                  str(output / f"{component}.safetensors"))
        if component == "semantic":
            # Whitelist architecture fields: upstream training configs contain local paths.
            keys = ("vocab_size", "phoneme_vocab_size", "embedding_dim", "hidden_dim",
                    "head", "linear_units", "n_layer", "EOS")
            configs[component] = {k: checkpoint["config"]["model"][k] for k in keys}
        elif component == "acoustic":
            keys = ("inter_channels", "hidden_channels", "filter_channels", "n_heads",
                    "n_layers", "kernel_size", "gin_channels", "semantic_frame_rate")
            configs[component] = {k: checkpoint["config"]["model"][k] for k in keys}
            data = checkpoint["config"]["data"]
            configs["reference"] = {k: data[k] for k in
                ("sampling_rate", "filter_length", "hop_length", "win_length")}
        elif component in ("hubert", "bert"):
            directory = path.parent
            data = json.loads((directory / "config.json").read_text())
            # Transformers metadata can contain the checkpoint author's absolute paths.
            data = {k: v for k, v in data.items() if not k.startswith("_") and k != "transformers_version"}
            data["torch_dtype"] = {"bf16": "bfloat16"}.get(args.dtype, args.dtype)
            configs[component] = data
            if component == "bert":
                shutil.copyfile(directory / "tokenizer.json", output / "tokenizer.json")
        del state, checkpoint
        print(f"Converted {component}", flush=True)
    weight_map = {}
    for component in ("g2pw", *sources):
        filename = f"{component}.safetensors"
        with safe_open(output / filename, framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name in weight_map:
                    raise ValueError(f"Duplicate checkpoint weight: {name}")
                weight_map[name] = filename
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2) + "\n")
    # These are the official v5 vocoder/CFM architecture definitions; convolution
    # channel/kernel dimensions are independently validated against the weights.
    configs["vocoder"] = {"upsample_rates": [10, 6, 2, 2, 2],
                          "resblock_dilations": [[1, 3, 5]] * 3}
    configs["acoustic"].update(style_heads=2, style_temporal_layers=2, mrte_heads=4,
                               wn_layers=8, wn_dilation_rate=1,
                               semantic_repeat=2, feature_repeat=2)
    configs["flow"] = {"hidden": 1024, "layers": 22, "heads": 16, "head_dim": 64,
                       "text_channels": 512, "text_layers": 4, "mel_channels": 100,
                       "ff_multiplier": 2, "steps": 4 if args.variant == "v5turbo" else 32,
                       "cfg_scale": 0.0 if args.variant == "v5turbo" else 1.30,
                       "noise_temperature": 0.875}
    configs["reference"].update(hubert_sample_rate=16000, min_seconds=3, max_seconds=10,
        semantic_silence_seconds=0.3, style_bins=704,
        mel_fft=1280, mel_hop=320, mel_window=1280, mel_sample_rate=32000,
        mel_min=-12, mel_max=2)
    configs["synthesis"] = {"window_frames": 1000, "reference_frames": 500,
                            "rolling_frames": 32, "target_frames": 640,
                            "stream_core_frames": 64, "stream_future_frames": 16,
                            "stream_holdback_frames": 16, "vocoder_context_frames": 24,
                            "stream_overlap_frames": 4, "min_semantic_tokens": 11}
    configs.update(model_type="gpt_sovits_v5", variant=args.variant, dtype=args.dtype,
                   sample_rate=48000, samples_per_frame=480, semantic_rate=25)
    (output / "config.json").write_text(json.dumps(configs, indent=2) + "\n")


if __name__ == "__main__":
    main()
