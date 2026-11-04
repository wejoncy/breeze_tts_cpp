"""Package native MeCab dictionaries and TTS phone-mapping policy."""
import ast
import json
from pathlib import Path
import shutil
from export_linguistics import export_morphology, module_data


def export_japanese(text, dictionary, user_dictionary, output):
    text, output = Path(text), Path(output)
    data = {}
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
    data['punctuation'] = module_data(text / 'symbols.py')['punctuation']
    # Python's Unicode lowercase operation can expand a codepoint. Export the
    # finite mapping so the runtime does not depend on process locale.
    data['lower'] = {chr(cp): chr(cp).lower() for cp in range(0x110000) if chr(cp).lower() != chr(cp)}
    export_morphology(dictionary, output / 'japanese_mecab')
    shutil.copyfile(user_dictionary, output / 'japanese_user.dic')
    (output / 'japanese.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
