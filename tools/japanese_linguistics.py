"""Export UTF-8 Japanese dictionaries and Open JTalk linguistic rules offline."""
import ast
import json
from pathlib import Path
import re
from export_linguistics import export_morphology, module_data


def export_japanese(text, openjtalk, dictionary, user_dictionary, output):
    text, openjtalk, output = map(Path, (text, openjtalk, output))
    data = dict(constants={}, tables={})
    for component in ['njd_set_pronunciation', 'njd_set_digit', 'njd_set_accent_phrase',
                      'njd_set_accent_type', 'njd_set_unvoiced_vowel', 'text2mecab', 'jpcommon']:
        source = (openjtalk / component / f'{component}_rule_utf_8.h').read_text(encoding='utf-8')
        data['constants'].update((key, ast.literal_eval(value)) for key, value in
            re.findall(r'^#define\s+(\w+)\s+("(?:\\.|[^"\\])*")', source, re.M))
        for name, body in re.findall(r'static const char \*(\w+)\[\] = \{(.*?)\};', source, re.S):
            body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
            if re.sub(r'"(?:\\.|[^"\\])*"|NULL|[\s,]', '', body):
                raise ValueError(f'Unsupported Japanese rule initializer: {name}')
            data['tables'][name] = [None if value == 'NULL' else ast.literal_eval(value)
                                   for value in re.findall(r'"(?:\\.|[^"\\])*"|NULL', body)]
    source = ast.parse((text / 'japanese.py').read_text(encoding='utf-8'))
    def pattern(name):
        return next(ast.literal_eval(node.value.args[0]) for node in ast.walk(source)
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == name)
    data['marks'] = pattern('_japanese_marks')
    replacement_function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'post_replace_ph')
    data['replacements'] = next(ast.literal_eval(node.value) for node in ast.walk(replacement_function)
                                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict))
    data['symbol_replacements'] = next(ast.literal_eval(node.value.generators[0].iter) for node in ast.walk(source)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == '_symbols_to_japanese')
    labels = (openjtalk / 'jpcommon/jpcommon_label.c').read_text(encoding='utf-8')
    data['max_mora'] = int(re.search(r'^#define\s+MAX_M\s+(\d+)', labels, re.M)[1])
    data['punctuation'] = module_data(text / 'symbols.py')['punctuation']
    # Python's Unicode lowercase operation can expand a codepoint. Export the
    # finite mapping so the runtime does not depend on process locale.
    data['lower'] = {chr(cp): chr(cp).lower() for cp in range(0x110000) if chr(cp).lower() != chr(cp)}
    export_morphology(dictionary, output / 'japanese_morphology.bin', [user_dictionary])
    (output / 'japanese.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
