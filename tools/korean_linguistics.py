"""Export the official Korean frontend rules and MeCab dictionaries offline."""
import ast
import json
from pathlib import Path
import re
from re import _parser as regex_parser
from export_linguistics import export_morphology


def substitution(pattern, replacement):
    """Capture slices are measured in Unicode codepoints, never UTF-8 bytes."""
    parsed = regex_parser.parse(pattern, 0)
    captures = {}

    def width(items):
        return regex_parser.SubPattern(parsed.state, data=items).getwidth()

    def visit(items, prefix=(0, 0), suffix=(0, 0)):
        for i, (operation, value) in enumerate(items):
            before, after = width(items[:i]), width(items[i + 1:])
            before = tuple(a + b for a, b in zip(prefix, before))
            after = tuple(a + b for a, b in zip(suffix, after))
            if str(operation) == "SUBPATTERN":
                index, _, _, body = value
                low, high = body.getwidth()
                if low == high and before[0] == before[1]:
                    captures[index] = [0, before[0], low]
                elif low == high and after[0] == after[1]:
                    captures[index] = [1, after[0], low]
                elif before[0] == before[1] and after[0] == after[1]:
                    captures[index] = [2, before[0], after[0]]
                else:
                    raise ValueError(f"Unsupported variable Korean capture: {pattern}")
                visit(body.data, before, after)
            elif str(operation) == "BRANCH":
                if any("SUBPATTERN" in str(branch) for branch in value[1]):
                    raise ValueError(f"Unsupported conditional Korean capture: {pattern}")

    visit(parsed.data)
    pieces, literal, cursor = [], "", 0
    while cursor < len(replacement):
        cp = replacement[cursor]
        if cp != "\\":
            literal += cp
            cursor += 1
            continue
        cursor += 1
        if cursor == len(replacement):
            raise ValueError("Trailing Korean replacement escape")
        if replacement[cursor] == "\\":
            literal += "\\"
            cursor += 1
            continue
        end = cursor
        while end < len(replacement) and replacement[end].isdigit():
            end += 1
        if end == cursor:
            raise ValueError(f"Unsupported Korean replacement escape: {replacement}")
        if literal:
            pieces.append(literal)
            literal = ""
        pieces.append(captures[int(replacement[cursor:end])])
        cursor = end
    if literal:
        pieces.append(literal)
    # Python's Unicode \w is letters/numbers/underscore. Oniguruma's \w
    # also includes marks/connectors, so express the Python boundary explicitly.
    return [pattern.replace(r"\W", r"[^\p{L}\p{N}_]"), pieces]


def function_rules(path, names):
    """Interpret only literal assignments, loops and substitutions, not Python code."""
    functions = {node.name: node for node in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.FunctionDef)}
    rules = []
    for name in names:
        values = {"inp": "", "descriptive": False, "verbose": False}

        def value(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return values[node.id]
            if isinstance(node, (ast.List, ast.Tuple)):
                return [value(item) for item in node.elts]
            if isinstance(node, ast.JoinedStr):
                return "".join(str(value(item.value)) if isinstance(item, ast.FormattedValue) else value(item) for item in node.values)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                arguments = [value(item) for item in node.args]
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "re" and node.func.attr == "sub":
                    rules.append(substitution(arguments[0], arguments[1]))
                    return arguments[2]
                if node.func.attr == "replace":
                    rules.append(substitution(re.escape(arguments[0]), arguments[1].replace("\\", "\\\\")))
                    return value(node.func.value)
            raise ValueError(f"Unsupported Korean rule expression: {ast.dump(node)}")

        def statements(items):
            for node in items:
                if isinstance(node, ast.Assign):
                    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                        raise ValueError("Unsupported Korean rule assignment")
                    if node.targets[0].id != "rule":
                        values[node.targets[0].id] = value(node.value)
                elif isinstance(node, ast.For):
                    for entry in value(node.iter):
                        for target, part in zip(node.target.elts, entry):
                            values[target.id] = part
                        statements(node.body)
                elif isinstance(node, ast.If):
                    statements(node.body if value(node.test) else node.orelse)
                elif isinstance(node, (ast.Expr, ast.Return)):
                    continue
                else:
                    raise ValueError(f"Unsupported Korean rule statement: {ast.dump(node)}")
        statements(functions[name].body)
    return rules


def export_korean(text, g2pk, dictionary, jamo, output):
    text, g2pk, jamo, output = map(Path, (text, g2pk, jamo, output))
    tree = ast.parse((text / "korean.py").read_text(encoding="utf-8"))
    lists, replacements = {}, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("_latin_to_hangul", "_hangul_divided"):
                lists[name] = ast.literal_eval(node.value.generators[0].iter)
        if isinstance(node, ast.FunctionDef) and node.name == "post_replace_ph":
            replacements = next(ast.literal_eval(item.value) for item in node.body
                if isinstance(item, ast.Assign) and item.targets[0].id == "rep_map")
    if replacements is None or len(lists) != 2:
        raise ValueError("Cannot locate official Korean frontend tables")
    latin = {}
    candidates = [chr(cp) for cp in range(0x110000) if chr(cp).lower() != chr(cp).upper()]
    for letter, spoken in lists["_latin_to_hangul"]:
        for cp in candidates:
            if re.fullmatch(letter, cp, re.IGNORECASE):
                latin[cp] = spoken
    idioms = []
    for line in (g2pk / "idioms.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "===" in line:
            idioms.append(substitution(*line.split("===")))
    special = function_rules(g2pk / "special.py", ["jyeo", "ye", "consonant_ui", "josa_ui",
        "vowel_ui", "jamo", "rieulgiyeok", "rieulbieub", "verb_nieun", "balb", "palatalize", "modifying_rieul"])
    regular = []
    rows = (g2pk / "table.csv").read_text(encoding="utf-8").splitlines()
    onsets = rows[0].split(",")
    for row in rows[1:]:
        columns = row.split(",")
        for index, onset in enumerate(onsets[1:], 1):
            if columns[index]:
                regular.append(substitution(columns[0] + onset, columns[index].split("(")[0]))
    links = function_rules(g2pk / "regular.py", ["link1", "link2", "link3", "link4"])
    names = json.loads((jamo / "data/U+11xx.json").read_text(encoding="utf-8"))
    hcj = json.loads((jamo / "data/U+31xx.json").read_text(encoding="utf-8"))
    reverse = {name: cp for cp, name in hcj.items()}
    compatibility = {cp: reverse.get(re.sub(r"(?<=HANGUL )(\w+)", "LETTER", name), cp)
                     for cp, name in names.items()}
    ranges = []
    for cp in range(0x110000):
        if chr(cp).isdecimal():
            if ranges and ranges[-1][1] == cp - 1:
                ranges[-1][1] = cp
            else:
                ranges.append([cp, cp])
    data = dict(latin=latin, idioms=idioms, special=special,
        regular=regular, links=links, compatibility=compatibility,
        divided=lists["_hangul_divided"], replacements={k: v for k, v in replacements.items() if len(k) == 1}, decimal=ranges,
        bound_nouns="군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통",
        digits="영일이삼사오육칠팔구", modifiers="한 두 세 네 다섯 여섯 일곱 여덟 아홉".split(),
        tens="열 스물 서른 마흔 쉰 예순 일흔 여든 아흔".split(),
        scales="십백천만십백천억십백천조십백천", twenty="스무")
    (output / "korean.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    export_morphology(dictionary, output / "korean_mecab")
