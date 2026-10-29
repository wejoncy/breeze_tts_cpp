"""Checkpoint resolution and metadata, independent of engine lifetime."""

import hashlib
import json
from pathlib import Path


def resolve_model_dir(model, revision=None):
    local = Path(model).expanduser()
    if local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(model), revision=revision))


def model_info(model):
    model = Path(model)
    config = json.loads((model / "config.json").read_text())
    names = {"config.json"}
    index = model / "model.safetensors.index.json"
    if index.exists():
        names.add(index.name)
        names.update(json.loads(index.read_text())["weight_map"].values())
    else:
        names.add("model.safetensors")
    digest = hashlib.sha256(b"breeze-checkpoint-v1\0")
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid checkpoint path")
        path = model / relative
        encoded = relative.as_posix().encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    return dict(
        frame_rate=float(config["codec_config"]["_frame_rate"]),
        model_digest=digest.hexdigest(),
        max_new_tokens=1500,
        max_batch_texts=128,
    )
