"""Export CPU linguistic assets alongside the native TTS checkpoint."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil


def export_frontend(source, g2pw, tokenizer, output, dtype):
    def package(name):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f"Offline text conversion requires {name}")
        return Path(next(iter(spec.submodule_search_locations)))

    pinyin = package("pypinyin")
    jieba = package("jieba_fast")
    opencc = package("opencc")
    text = source / "GPT_SoVITS/text"
    def rows(path):
        return [line.split("\t") for line in path.read_text(encoding="utf-8").strip().splitlines()]

    polyphonic = rows(g2pw / "POLYPHONIC_CHARS.txt")
    labels = sorted({phoneme for _, phoneme in polyphonic})
    characters = sorted({char for char, _ in polyphonic})
    label_ids = {value: i for i, value in enumerate(labels)}
    allowed = {char: [] for char in characters}
    for char, phoneme in polyphonic:
        allowed[char].append(label_ids[phoneme])
    bopomofo = json.loads((g2pw / "bopomofo_to_pinyin_wo_tune_dict.json").read_text(encoding="utf-8"))

    def reading(phoneme):
        # Upstream returns None for missing syllables and uses pypinyin fallback.
        return bopomofo[phoneme[:-1]] + phoneme[-1] if phoneme[:-1] in bopomofo else ""

    def dictionary(name):
        return {key: value.split()[0] for key, value in rows(opencc / "dictionary" / name)}

    tree = ast.parse((text / "g2pw/onnx_api.py").read_text(encoding="utf-8"))
    excluded = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and target.attr in ("non_polyphonic", "non_monophonic"):
                excluded[target.attr] = ast.literal_eval(node.value)
    if len(excluded) != 2:
        raise ValueError("Cannot locate official G2PW exclusion rules")
    from pypinyin.phonetic_symbol import phonetic_symbol

    data = dict(
        labels=[reading(value) for value in labels], characters=characters,
        allowed=[allowed[char] for char in characters],
        exclude_polyphonic=sorted(excluded["non_polyphonic"]),
        monophonic={char: reading(value) for char, value in rows(g2pw / "MONOPHONIC_CHARS.txt")
                    if char not in excluded["non_monophonic"]},
        context_chars=16, max_tokens=512, tone_marks=phonetic_symbol,
        traditional_phrases=dictionary("STPhrases.txt"),
        traditional_characters=dictionary("STCharacters.txt"), taiwan_variants=dictionary("TWVariants.txt"),
    )
    (output / "g2pw.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    from export_linguistics import export_linguistics
    export_linguistics(text, jieba, pinyin, output)
    for source_file, destination in ((tokenizer, output / "g2pw_tokenizer.json"),):
        if source_file.resolve() != destination.resolve():
            shutil.copyfile(source_file, destination)
    from g2pw_weights import export_weights
    export_weights(g2pw / "g2pW.onnx", output, dtype)

    overrides = {}
    for name in ("polyphonic.rep", "polyphonic-fix.rep"):
        for line in (text / "g2pw" / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                word, value = line.split(":", 1)
                overrides[word.strip()] = ast.literal_eval(value.strip())
    (output / "polyphone.json").write_text(
        json.dumps(overrides, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
