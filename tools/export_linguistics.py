"""Export language data used by the native frontend; no runtime Python dependency.

The dictionaries and HMM probabilities come from the checkpoint's official
jieba_fast/pypinyin versions. GPT-SoVITS and PaddleSpeech rules are extracted as
literal data, never executed. Containers retain the existing checkpoint format.
"""
import ast
import json
from pathlib import Path
import struct
import shutil


def literal_assignments(path):
    values = {}
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
            if name:
                try:
                    values[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values, tree


def evaluate(node, values):
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        convert = {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)]
        return convert(evaluate(item, values) for item in node.elts)
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return evaluate(node.left, values) + evaluate(node.right, values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("list", "set", "sorted") and len(node.args) == 1 and not node.keywords:
        return {"list": list, "set": set, "sorted": sorted}[node.func.id](evaluate(node.args[0], values))
    return ast.literal_eval(node)


def module_data(path):
    values, tree = literal_assignments(path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = evaluate(node.value, values)
            except (ValueError, TypeError, KeyError):
                pass
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
            values[node.target.id] += evaluate(node.value, values)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "append" and isinstance(call.func.value, ast.Name) and len(call.args) == 1:
                values[call.func.value.id].append(evaluate(call.args[0], values))
    return values


def container(path, magic, sections):
    offset = 16 + 32 * len(sections)
    directory, payload = bytearray(), bytearray()
    for name, data in sections:
        if len(name) > 8:
            raise ValueError("language section names are limited to 8 bytes")
        directory += name.encode("ascii").ljust(8, b"\0") + struct.pack("<QQQ", offset + len(payload), len(data), 0)
        payload += data
    Path(path).write_bytes(magic + struct.pack("<II", 1, len(sections)) + directory + payload)


def integers(values):
    return b"".join(struct.pack("<I", value) for value in values)


def export_jieba(package, output):
    """Preserve the source dictionary/probabilities in cppjieba's UTF-8 formats."""
    import jieba_fast.finalseg as final
    shutil.copyfile(package / "dict.txt", output / "jieba.dict.utf8")
    states = "BEMS"
    lines = [" ".join(repr(final.start_P[state]) for state in states)]
    lines.extend(" ".join(repr(final.trans_P[state].get(next_state, -3.14e100))
                          for next_state in states) for state in states)
    for state in states:
        probabilities = final.emit_P[state]
        if any(character in ",:\r\n" or character.isspace() for character in probabilities):
            raise ValueError("HMM emissions contain characters unsupported by cppjieba's format")
        lines.append(",".join(character + ":" + repr(probabilities[character])
                              for character in sorted(probabilities)))
    (output / "jieba.hmm.utf8").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_pinyin(package, text, output):
    phrases = json.loads((package / "phrases_dict.json").read_text(encoding="utf-8"))
    characters = json.loads((package / "pinyin_dict.json").read_text(encoding="utf-8"))
    conversion, _ = literal_assignments(text / "zh_normalization/char_convert.py")
    simplified, traditional = conversion["simplified_charcters"], conversion["traditional_characters"]
    if len(simplified) != len(traditional):
        raise ValueError("traditional/simplified mapping length mismatch")
    t2s = {ord(t): ord(s) for s, t in zip(simplified, traditional)}
    t2s = sorted((t, s) for t, s in t2s.items() if t != s)
    pool, codepoints, readings = bytearray(), [], set()

    def add(runs):
        offset = len(pool)
        for run in runs:
            if not run or len(run) >= 256:
                raise ValueError("invalid pronunciation candidate count")
            pool.append(len(run))
            for syllable in run:
                readings.add(syllable)
                encoded = syllable.encode("utf-8")
                if len(encoded) >= 256:
                    raise ValueError("pronunciation exceeds format limit")
                pool.extend(bytes([len(encoded)]) + encoded)
        return offset, len(runs)

    phrase_data = bytearray(struct.pack("<I", len(phrases)))
    for word, runs in sorted(phrases.items()):
        offset, count = add(runs)
        phrase_data += struct.pack("<IHIHH", len(codepoints), len(word), offset, count, 0)
        codepoints.extend(map(ord, word))
    char_data = bytearray(struct.pack("<I", len(characters)))
    for cp, syllables in sorted((int(cp), value) for cp, value in characters.items()):
        offset, count = add([syllables.split(",")])
        char_data += struct.pack("<IIHH", cp, offset, count, 0)
        codepoints.append(cp)
    ranges, start = [], None
    for cp in range(0x10001):
        if cp < 0x10000 and chr(cp).isnumeric():
            start = cp if start is None else start
        elif start is not None:
            ranges.append((start, cp - 1))
            start = None
    container(output / "pinyin.bin", b"GSPPYX01", [
        ("meta", struct.pack("<II", 1, 0)), ("tags", b""), ("cps", integers(codepoints)),
        ("phrases", phrase_data), ("sylpool", pool), ("chars", char_data),
        ("t2s", struct.pack("<I", len(t2s)) + b"".join(struct.pack("<II", *pair) for pair in t2s)),
        ("numrange", struct.pack("<I", len(ranges)) + b"".join(struct.pack("<II", *pair) for pair in ranges)),
    ])
    from pypinyin.contrib.tone_convert import to_initials, to_finals_tone3, to_tone3, to_normal
    from pypinyin.constants import RE_HANS
    # Style conversion is a finite linguistic table, including all dictionary
    # readings and the numbered variants returned by G2PW/word overrides.
    extra = output / "g2pw.json"
    if extra.exists():
        g2pw = json.loads(extra.read_text(encoding="utf-8"))
        readings.update(value for value in g2pw["labels"] + list(g2pw["monophonic"].values()) if value)
    for filename in ("polyphonic.rep", "polyphonic-fix.rep"):
        for line in (text / "g2pw" / filename).read_text(encoding="utf-8").splitlines():
            if line.strip():
                _, value = line.split(":", 1)
                readings.update(ast.literal_eval(value.strip()))
    variants = set(readings)
    for reading in readings:
        variants.add(to_tone3(reading, neutral_tone_with_five=True))
        base = to_normal(reading)
        variants.update(base + str(tone) for tone in range(1, 6))
    style = {reading: [to_initials(reading), to_finals_tone3(reading, neutral_tone_with_five=True)] for reading in sorted(variants)}
    import re
    ranges = []
    for match in re.finditer(r"(\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})(?:-(\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}))?", RE_HANS.pattern):
        first = int(match[1][2:], 16)
        last = int(match[2][2:], 16) if match[2] else first
        ranges.append([first, last])
    if not ranges:
        raise ValueError("cannot extract pinyin Han ranges")
    return dict(styles=style, han_ranges=ranges)


def export_rules(text):
    data = {}
    sandhi, _ = literal_assignments(text / "tone_sandhi.py")
    chinese, _ = literal_assignments(text / "chinese2.py")
    old_symbols = module_data(text / "symbols.py")
    for source, target in (("must_neural_tone_words", "neutral_words"), ("must_not_neural_tone_words", "not_neutral_words")):
        data[target] = sorted(sandhi[source])
    data["sandhi_punctuation"] = sandhi["punc"]
    data["sandhi"] = dict(particles="吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶", structural="的地得", aspect="了着过", plural="们子", location="上下里", direction="来去", direction_prefix="上下进出回过起开", ge_prefix="几有两半多各整每做是")
    data["phoneme_normalization"] = {name: chinese[name] for name in ("v_rep_map", "pinyin_rep_map", "single_rep_map")}
    data.update(must_erhua=sorted(chinese["must_erhua"]), not_erhua=sorted(chinese["not_erhua"]), replacements=list(chinese["rep_map"].items()), punctuation=old_symbols["punctuation"])
    data["symbols"] = module_data(text / "symbols2.py")["symbols"]
    data["aliases"] = {"嗯": "恩", "呣": "母"}
    data["speakable_ranges"] = [[0x4e00, 0x9fa5]]
    data["phones"] = {word: spoken.split(" ") for word, spoken in (line.split("\t") for line in (text / "opencpop-strict.txt").read_text(encoding="utf-8").splitlines())}
    normalization, patterns = {}, {}
    for filename in ("num.py", "chronology.py", "quantifier.py", "phonecode.py"):
        values, tree = literal_assignments(text / "zh_normalization" / filename)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "compile":
                try:
                    pattern = evaluate(node.value.args[0], values)
                    if len(node.value.args) > 1:
                        flag = node.value.args[1]
                        if not isinstance(flag, ast.Attribute) or flag.attr != "VERBOSE":
                            raise ValueError("unsupported normalization regex flags")
                        pattern = "(?x)" + pattern
                    patterns[node.targets[0].id] = pattern
                except (ValueError, TypeError, KeyError):
                    pass
        if filename == "num.py":
            for node in tree.body:
                if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
                    continue
                if node.targets[0].id == "DIGITS" and isinstance(node.value, ast.DictComp):
                    generator = node.value.generators[0]
                    if len(node.value.generators) != 1 or not isinstance(generator.iter, ast.Call) or not isinstance(generator.iter.func, ast.Name) or generator.iter.func.id != "enumerate":
                        raise ValueError("unsupported digit table")
                    values["DIGITS"] = {str(i): cp for i, cp in enumerate(ast.literal_eval(generator.iter.args[0]))}
                elif node.targets[0].id == "UNITS" and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "OrderedDict":
                    values["UNITS"] = ast.literal_eval(node.value.args[0])
        for name, key in (("DIGITS", "digits"), ("UNITS", "units"), ("asmd_map", "asmd"), ("power_map", "power")):
            if name in values:
                normalization[key] = values[name]
        if "measure_dict" in values:
            normalization["measure"] = list(values["measure_dict"].items())
    rules = [("date", "RE_DATE"), ("date2", "RE_DATE2"), ("time_range", "RE_TIME_RANGE"), ("time", "RE_TIME"), ("to_range", "RE_TO_RANGE"), ("temperature", "RE_TEMPERATURE"), ("measure", ""), ("asmd", "RE_ASMD"), ("power", "RE_POWER"), ("frac", "RE_FRAC"), ("percentage", "RE_PERCENTAGE"), ("mobile", "RE_MOBILE_PHONE"), ("phone", "RE_TELEPHONE"), ("national_phone", "RE_NATIONAL_UNIFORM_NUMBER"), ("range", "RE_RANGE"), ("negative", "RE_INTEGER"), ("version", "RE_VERSION_NUM"), ("decimal", "RE_DECIMAL_NUM"), ("quantifier", "RE_POSITIVE_QUANTIFIERS"), ("default", "RE_DEFAULT_NUM"), ("number", "RE_NUMBER")]
    normalization["rules"] = [[kind, patterns[name] if name else ""] for kind, name in rules]
    _, tree = literal_assignments(text / "zh_normalization/text_normlization.py")
    post = []

    def replacement_chain(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "replace":
            replacement_chain(node.func.value)
            post.append([ast.literal_eval(arg) for arg in node.args])

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_post_replace":
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    replacement_chain(statement.value)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "compile" and isinstance(node.targets[0], ast.Attribute):
            normalization["splitter"] = ast.literal_eval(node.value.args[0])
    normalization["post"] = post
    normalization["alt_one"] = "幺"
    normalization["quantifier_two"] = "两"
    normalization["strip_before"] = "——《》【】<>{}()（）#&@“”^_|\\"
    normalization["strip_after"] = "-——《》【】<=>{}()（）#&@“”^_|\\"
    data["normalization"] = normalization
    return data


def export_linguistics(text, jieba, pinyin, output):
    output = Path(output)
    data = export_rules(Path(text))
    export_jieba(Path(jieba), output)
    data["pinyin"] = export_pinyin(Path(pinyin), Path(text), output)
    (output / "linguistics.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def export_cantonese(text, jyutping, output):
    """Export ToJyutping's complete default phrase selection without a runtime library."""
    text, jyutping, output = Path(text), Path(jyutping), Path(output)
    rules = module_data(text / "cantonese.py")
    alphabets, _ = literal_assignments(jyutping / "Jyutping.py")
    onset, nucleus, rhyme, coda = (alphabets[name] for name in ("onset", "nucleus", "rhyme", "coda"))
    tones = 6
    stride = tones * (len(nucleus) * len(coda) + len(rhyme))

    def syllable(value):
        first, rest = divmod(value, stride)
        middle, tone = divmod(rest, tones)
        vowel = rhyme[middle - len(nucleus) * len(coda)] if middle >= len(nucleus) * len(coda) else nucleus[middle // len(coda)] + coda[middle % len(coda)]
        return onset[first] + vowel + str(tone + 1)

    # ToJyutping serializes its ordered candidates with two printable bytes
    # per syllable. '~' continues a multi-syllable character reading.
    serialized = (jyutping / "trie.txt").read_text(encoding="utf-8")
    root = [dict(), None]
    stack, depths, cursor = [root], [0], 1
    while stack:
        node, depth = stack[-1], depths[-1]
        while ord(serialized[cursor]) >= 256:
            node = node[0].setdefault(serialized[cursor], [dict(), None])
            cursor += 1
            depth += 1
        values = []
        while ord(serialized[cursor]) < 123:
            value, length = [], 0
            while length < depth:
                value.append(syllable((ord(serialized[cursor]) - 33) * 90 + ord(serialized[cursor + 1]) - 33))
                cursor += 2
                if serialized[cursor] == "~":
                    cursor += 1
                else:
                    length += 1
            values.append(value)
        if values:
            node[1] = values[0]
        if serialized[cursor] == "{":
            stack.append(node)
            depths.append(depth)
            cursor += 1
        elif serialized[cursor] == "}":
            stack.pop()
            depths.pop()
            cursor += 1
        elif ord(serialized[cursor]) < 256:
            raise ValueError("unsupported ToJyutping dictionary encoding")
    if cursor != len(serialized):
        raise ValueError("trailing ToJyutping dictionary data")
    data = json.loads((output / "linguistics.json").read_text(encoding="utf-8"))
    ids = {value: index for index, value in enumerate(data["symbols"])}

    def phones(reading):
        tone, base = int(reading[-1]), reading[:-1]
        for initial in rules["INITIALS"]:
            if base.startswith(initial):
                if base.startswith("nga"):
                    initial, final = base[:2], base[2:] or base[-1]
                else:
                    final = base[len(initial):] or initial[-1]
                return [ids.get("Y" + initial, ids["UNK"]), ids.get("Y" + final + str(tone), ids["UNK"])]
        # Upstream emits no word2ph entry for these readings; cleaner rejects
        # the resulting character alignment. Preserve that terminal/error.
        return None

    encoded, count = bytearray(), 0
    pending = [("", root)]
    while pending:
        key, node = pending.pop()
        if node[1]:
            readings = [" ".join(node[1])] if len(key) == 1 else node[1][:len(key)]
            mapped = [phones(reading) for reading in readings]
            values = [phone for group in mapped for phone in group] if all(mapped) else []
            encoded += struct.pack("<HH", len(key), len(values)) + integers(map(ord, key)) + integers(values)
            count += 1
        pending.extend((key + cp, child) for cp, child in sorted(node[0].items(), reverse=True))
    container(output / "cantonese.bin", b"GSVYUE01", [("entries", struct.pack("<I", count) + encoded)])
    data["cantonese"] = dict(replacements=list(rules["rep_map"].items()))
    (output / "linguistics.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def export_morphology(dictionary, destination):
    """Keep MeCab's native dictionary files; the library validates their format."""
    dictionary, destination = Path(dictionary), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["char.bin", "matrix.bin", "sys.dic", "unk.dic"]:
        shutil.copyfile(dictionary / name, destination / name)
    configuration = (dictionary / "dicrc").read_text(encoding="utf-8") if (dictionary / "dicrc").is_file() else ""
    # Open JTalk dictionaries can omit dicrc; standalone MeCab requires this
    # standard boundary feature. Other supplied dictionary settings stay intact.
    if not any(line.partition("=")[0].strip() == "bos-feature" for line in configuration.splitlines()):
        configuration += "\nbos-feature = BOS/EOS,*,*,*,*,*,*,*,*\n"
    (destination / "dicrc").write_text(configuration, encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jyutping", type=Path, help="ToJyutping package directory for Cantonese assets")
    arguments = parser.parse_args()
    import jieba_fast
    import pypinyin
    arguments.output.mkdir(parents=True, exist_ok=True)
    export_linguistics(arguments.text, Path(jieba_fast.__file__).parent, Path(pypinyin.__file__).parent, arguments.output)
    if arguments.jyutping:
        export_cantonese(arguments.text, arguments.jyutping, arguments.output)
