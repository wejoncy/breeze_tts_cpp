"""Package the standard FastText model with the TTS input normalization policy."""
import json
from pathlib import Path
import shutil


def export_detector(model, output):
    model, output = Path(model), Path(output)
    destination = output / "language_detector.bin"
    if model.resolve() != destination.resolve():
        shutil.copyfile(model, destination)
    lower, upper, cased, ignorable = {}, [], [], []

    def include(ranges, cp):
        if ranges and ranges[-1][1] == cp - 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp])

    for cp in range(0x110000):
        char = chr(cp)
        if char.lower() != char:
            lower[char] = char.lower()
        if char.isupper():
            include(upper, cp)
        is_cased = char.isupper() or char.islower() or char.istitle()
        if is_cased:
            include(cased, cp)
        # Infer the interpreter's Case_Ignorable property from the public
        # Unicode lowercase operation. This also covers cased modifier letters.
        is_ignorable = (('AΣ' + char + '.').lower()[1] == 'ς') if is_cased else (('AΣ' + char + 'A').lower()[1] == 'σ')
        if is_ignorable:
            include(ignorable, cp)
    data = dict(max_input_length=80, normalize_input=True,
                lower=lower, upper=upper, cased=cased, case_ignorable=ignorable)
    (output / "language_policy.json").write_text(
        json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding="utf-8")
