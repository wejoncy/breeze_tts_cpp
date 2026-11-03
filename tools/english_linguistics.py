"""Export GPT-SoVITS English assets for the native CPU phonemizer.

Conversion reads the existing g2p_en/wordsegment data and NLTK's English
perceptron JSON files. These packages are not native runtime dependencies.
"""
import json
from pathlib import Path
import pickle
import unicodedata

from export_linguistics import module_data


def export_english(text, g2p, segmenter, tagger, output):
    import inflect
    import numpy as np
    from safetensors.numpy import save_file

    text, g2p, segmenter, tagger, output = map(
        Path, (text, g2p, segmenter, tagger, output)
    )
    dictionary = {}
    for line in (text / "cmudict.rep").read_text(encoding="utf-8").splitlines()[56:]:
        word, phones = line.strip().split("  ", 1)
        dictionary[word.lower()] = phones.split()
    for filename, replace in (("cmudict-fast.rep", False), ("engdict-hot.rep", True)):
        for line in (text / filename).read_text(encoding="utf-8").splitlines():
            word, *phones = line.strip().split(" ")
            if replace or word.lower() not in dictionary:
                dictionary[word.lower()] = phones
    for word in ("AE", "AI", "AR", "IOS", "HUD", "OS"):
        dictionary.pop(word.lower(), None)
    homographs = {}
    for line in (g2p / "homographs.en").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        word, first, second, pos = line.split("|")
        homographs[word.lower()] = [first.split(), second.split(), pos]
    homographs["read"] = [["R", "IY1", "D"], ["R", "EH1", "D"], "VBP"]
    homographs["complex"] = [
        ["K", "AH0", "M", "P", "L", "EH1", "K", "S"],
        ["K", "AA1", "M", "P", "L", "EH0", "K", "S"], "JJ",
    ]
    names_path = text / "namedict_cache.pickle"
    # This is an explicit conversion of the user's upstream source assets.
    names = pickle.loads(names_path.read_bytes()) if names_path.exists() else {}
    names = {word: phones[0] for word, phones in names.items()}

    def counts(filename):
        return {word: float(count) for word, count in
                (line.split("\t") for line in (segmenter / filename).read_text(encoding="utf-8").splitlines())}

    engine = inflect.engine()
    cardinal = [engine.number_to_words(i, andword="") for i in range(1000)]
    cardinal_and = [engine.number_to_words(i) for i in range(1000)]
    magnitudes = [engine.number_to_words(10 ** (3 * i)).removeprefix("one ")
                  for i in range(1, len(inflect.mill))]
    last_words = {word.split()[-1].split("-")[-1]
                  for word in cardinal_and + magnitudes}
    ordinals = {word: engine.ordinal(word) for word in last_words}
    accent_map, combining = {}, []
    for cp in range(0x110000):
        char = chr(cp)
        if unicodedata.category(char) == "Mn":
            combining.append(cp)
        stripped = "".join(c for c in unicodedata.normalize("NFD", char)
                           if unicodedata.category(c) != "Mn")
        if stripped != char and stripped and stripped.isascii():
            accent_map[str(cp)] = stripped
    # Number words and phoneme alphabets are asset data, not model constants in
    # the runtime. Extract the alphabet from the upstream constructor literally.
    import ast
    tree = ast.parse((g2p / "g2p.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "phonemes"
            for target in node.targets
        ):
            parts = node.value
            phonemes = ast.literal_eval(parts.left) + ast.literal_eval(parts.right)
            break
    else:
        raise ValueError("Cannot locate g2p_en phoneme alphabet")
    english = module_data(text / "english.py")
    normalization = module_data(text / "en_normalization/expend.py")
    data = dict(
        dictionary=dictionary, names=names, homographs=homographs,
        replacements=list(english["rep_map"].items()),
        measurement=normalization["measurement_map"],
        measurement_order=list(normalization["measurement_map"]),
        cardinal=cardinal, cardinal_and=cardinal_and, magnitudes=magnitudes,
        ordinals=ordinals,
        years={str(i): normalization_year(engine, i) for i in range(1001, 3000)},
        accents=accent_map, combining=combining,
        phonemes=phonemes, graphemes=["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz"),
        symbols=module_data(text / "symbols2.py")["symbols"],
        unigrams=counts("unigrams.txt"), bigrams=counts("bigrams.txt"),
        word_total=1024908267229.0, word_limit=24, decoder_steps=20,
        pos={kind: json.loads((tagger / f"averaged_perceptron_tagger_eng.{kind}.json").read_text(encoding="utf-8"))
             for kind in ("weights", "tagdict", "classes")},
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "english.json").write_text(json.dumps(data, ensure_ascii=False,
                                                   separators=(",", ":")), encoding="utf-8")
    with np.load(g2p / "checkpoint20.npz", allow_pickle=False) as weights:
        save_file({"english." + name: np.asarray(weights[name], dtype=np.float32)
                   for name in weights.files}, str(output / "english.safetensors"))


def normalization_year(engine, number):
    if number == 2000:
        return "two thousand"
    if 2000 < number < 2010:
        return "two thousand " + engine.number_to_words(number % 100)
    if number % 100 == 0:
        return engine.number_to_words(number // 100) + " hundred"
    return engine.number_to_words(number, andword="", zero="oh", group=2).replace(", ", " ")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("text", "g2p", "segmenter", "tagger", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    export_english(args.text, args.g2p, args.segmenter, args.tagger, args.output)
