"""Export the GPT-SoVITS LangSegmenter and BudouX rule assets offline."""
import ast
import json
from pathlib import Path
from language_detection import export_detector


def export_router(text, splitter, budoux, detector, output):
    text, splitter, budoux, output = map(Path, (text, splitter, budoux, output))

    def assignments(path):
        result = {}
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    result[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        return result

    segmenter = text / 'LangSegmenter/langsegmenter.py'
    values = assignments(segmenter)
    config = assignments(splitter / 'config.py')
    utilities = assignments(splitter / 'split/utils.py')
    functions = {node.name: node for node in ast.parse(segmenter.read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef)}
    def pattern(function):
        return next(ast.literal_eval(node.value) for node in ast.walk(functions[function])
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'pattern')
    parts = [ast.literal_eval(node.value) for node in ast.walk(functions['split_jako'])
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'pattern']
    space, digit = [], []
    for cp in range(0x110000):
        for ranges, included in [(space, chr(cp).isspace()), (digit, chr(cp).isdigit())]:
            if included:
                if ranges and ranges[-1][1] == cp - 1:
                    ranges[-1][1] = cp
                else:
                    ranges.append([cp, cp])
    data = dict(language_map=values['DEFAULT_LANG_MAP'],
                chinese_japanese_map=config['ZH_JA_LANG_MAP'],
                punctuation=utilities['PUNCTUATION'], space=space, digit=digit,
                cjk_ranges=values['cjk_ranges'], cjk_punctuation=pattern('full_cjk'),
                japanese_parts=parts[0], korean_parts=parts[1],
                budoux={name: json.loads((budoux / 'models' / filename).read_text(encoding='utf-8'))
                        for name, filename in [('ja', 'ja.json'), ('zh', 'zh-hans.json')]})
    (output / 'language_splitter.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    export_detector(detector, output)
