"""Convert the official full FastText language classifier to CPU model assets."""
import json
from pathlib import Path
import struct


def export_detector(model, output):
    import numpy as np
    from safetensors.numpy import save_file
    raw = Path(model).read_bytes()
    cursor = 0

    def read(format):
        nonlocal cursor
        size = struct.calcsize(format)
        if cursor + size > len(raw):
            raise ValueError("Truncated FastText model")
        values = struct.unpack_from(format, raw, cursor)
        cursor += size
        return values

    if read('<ii') != (793712314, 12):
        raise ValueError("Expected the official unquantized FastText v12 classifier")
    dim, _, _, _, _, word_ngrams, loss, kind, bucket, minn, maxn, _, _ = read('<12id')
    if kind != 3 or loss != 1 or dim < 1 or bucket < 1 or minn < 1 or maxn < minn:
        raise ValueError("Expected a supervised hierarchical-softmax language classifier")
    size, nwords, nlabels, _, prune_size = read('<iiiqq')
    if min(nwords, nlabels) < 1 or size != nwords + nlabels or prune_size != -1:
        raise ValueError("Expected a complete unpruned language dictionary")
    words, labels, counts = [], [], []
    for i in range(size):
        end = raw.find(b'\0', cursor)
        if end < 0:
            raise ValueError("Unterminated FastText dictionary entry")
        word = raw[cursor:end].decode('utf-8')
        cursor = end + 1
        count, category = read('<qb')
        if category != int(i >= nwords):
            raise ValueError("FastText dictionary entries are not grouped by category")
        if i < nwords:
            words.append(word)
        else:
            if not word.startswith('__label__'):
                raise ValueError("Invalid FastText language label")
            labels.append(word[len('__label__'):])
            counts.append(count)
    matrices = {}
    for name in ['input', 'output']:
        if read('<?')[0]:
            raise ValueError("Use the full unquantized lid.176.bin classifier")
        rows, columns = read('<qq')
        if rows < 1 or columns != dim:
            raise ValueError("Invalid FastText classifier matrix geometry")
        length = rows * columns * 4
        if cursor + length > len(raw):
            raise ValueError("Truncated FastText classifier matrix")
        matrices['detector.' + name] = np.frombuffer(raw, dtype='<f4', count=rows * columns, offset=cursor).reshape(rows, columns).copy()
        cursor += length
    if cursor != len(raw) or matrices['detector.input'].shape != (nwords + bucket, dim) or matrices['detector.output'].shape != (nlabels, dim):
        raise ValueError("Inconsistent FastText classifier dimensions")
    nodes = [[-1, -1, count] for count in counts]
    nodes.extend([[-1, -1, 10**15] for _ in range(nlabels - 1)])
    leaf, node = nlabels - 1, nlabels
    for i in range(nlabels, 2 * nlabels - 1):
        children = []
        for _ in range(2):
            if leaf >= 0 and nodes[leaf][2] < nodes[node][2]:
                children.append(leaf)
                leaf -= 1
            else:
                children.append(node)
                node += 1
        nodes[i] = [*children, sum(nodes[j][2] for j in children)]
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
    data = dict(words=words, labels=labels, tree=[entry[:2] for entry in nodes],
                dimensions=dim, bucket=bucket, minn=minn, maxn=maxn, word_ngrams=word_ngrams,
                max_input_length=80, normalize_input=True,
                lower=lower, upper=upper, cased=cased, case_ignorable=ignorable)
    output = Path(output)
    (output / 'language_detector.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    save_file(matrices, str(output / 'language_detector.safetensors'))
