"""Extract native weights from the official G2PW asset during offline conversion."""
import mmap
import json
import re
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file


def _varint(data, offset):
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7
        if shift >= 70:
            raise ValueError("Invalid asset varint")


def _fields(data):
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 2:
            length, offset = _varint(data, offset)
            value = data[offset:offset + length]
            offset += length
        elif wire in (1, 5):
            length = 8 if wire == 1 else 4
            value = data[offset:offset + length]
            offset += length
        else:
            raise ValueError("Unsupported asset wire type")
        if offset > len(data):
            raise ValueError("Truncated asset")
        yield number, wire, value


def _tensor(data):
    dims, name, dtype, raw = [], "", None, None
    for number, wire, value in _fields(data):
        if number == 1:
            if wire == 0:
                dims.append(value)
            else:
                offset = 0
                while offset < len(value):
                    dim, offset = _varint(value, offset)
                    dims.append(dim)
        elif number == 2:
            dtype = value
        elif number == 8:
            name = bytes(value).decode()
        elif number == 9:
            raw = value
    return name, dims, dtype, raw


def export_weights(source, output, dtype):
    # Only asset deserialization: no graph interpreter or inference dependency.
    with Path(source).open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        view = memoryview(mapped)
        graph = next(value for number, _, value in _fields(view) if number == 7)
        tensors, nodes, constants = {}, [], {}
        for number, _, data in _fields(graph):
            if number == 5:
                name, shape, kind, raw = _tensor(data)
                if raw is None:
                    raise ValueError("G2PW asset requires embedded raw weights")
                tensors[name] = (shape, kind, bytes(raw))
            elif number == 1:
                inputs, outputs, op = [], [], ""
                for field, _, value in _fields(data):
                    if field in (1, 2):
                        (inputs if field == 1 else outputs).append(bytes(value).decode())
                    elif field == 4:
                        op = bytes(value).decode()
                    elif field == 5:
                        for attr, _, item in _fields(value):
                            if attr == 5:
                                _, shape, kind, raw = _tensor(item)
                                if not shape and kind == 1 and raw is not None:
                                    constants[outputs[0]] = struct.unpack("<f", raw)[0]
                nodes.append((op, inputs, outputs))
        # Drop all views before closing the mapped file.
        del data, value, item, raw, graph, view

    state = {}
    for name, (shape, kind, raw) in tensors.items():
        if "." in name and kind == 1:
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
            target_type = torch.float32 if "descriptor" in name else dtype
            state[name] = tensor.to(target_type).contiguous()
    consumers = {name: node for node in nodes if node[0] == "Add" for name in node[1]}
    for op, inputs, outputs in nodes:
        if op != "MatMul":
            continue
        for name in inputs:
            if name not in tensors or len(tensors[name][0]) != 2:
                continue
            add = consumers.get(outputs[0])
            bias = next((v for v in add[1] if v.endswith(".bias")), None) if add else None
            if bias:
                shape, kind, raw = tensors[name]
                if kind != 1:
                    raise ValueError("Unexpected G2PW projection type")
                state[bias[:-4] + "weight"] = torch.frombuffer(
                    bytearray(raw), dtype=torch.float32).reshape(shape).T.to(dtype).contiguous()
    producers = {name: node for node in nodes for name in node[2]}
    divisors = {constants[inputs[1]] for op, inputs, _ in nodes
                if op == "Div" and inputs[1] in constants and
                producers.get(inputs[0], (None,))[0] == "MatMul"}
    epsilons = set()
    for op, inputs, _ in nodes:
        if op == "Sqrt" and producers.get(inputs[0], (None,))[0] == "Add":
            epsilons.update(constants[v] for v in producers[inputs[0]][1] if v in constants)
    if len(divisors) != 1 or len(epsilons) != 1:
        raise ValueError("G2PW attention/norm configuration is ambiguous")
    hidden = state["bert.embeddings.word_embeddings.weight"].shape[1]
    head_dim = round(next(iter(divisors)) ** 2)
    if head_dim < 1 or hidden % head_dim:
        raise ValueError("Invalid G2PW attention heads")
    layers = {int(match[1]) for name in state
              if (match := re.match(r"bert\.encoder\.layer\.(\d+)\.", name))}
    if layers != set(range(len(layers))):
        raise ValueError("G2PW transformer layers are not contiguous")
    metadata = json.loads((output / "g2pw.json").read_text(encoding="utf-8"))
    metadata["model"] = dict(hidden_size=hidden, num_attention_heads=hidden // head_dim,
        num_hidden_layers=len(layers), layer_norm_eps=next(iter(epsilons)),
        max_position_embeddings=state["bert.embeddings.position_embeddings.weight"].shape[0],
        position_embedding_type="absolute", hidden_act="gelu")
    if state["classifier.weight"].shape[0] != len(metadata["labels"]):
        raise ValueError("G2PW labels and classifier disagree")
    save_file({f"g2pw.{name}": value for name, value in state.items()},
              str(output / "g2pw.safetensors"))
    (output / "g2pw.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
