"""Checkpoint resolution and metadata, independent of engine lifetime."""

import hashlib
from pathlib import Path
from request_schema import MAX_INPUTS


def resolve_model_dir(model, revision=None):
    local = Path(model).expanduser()
    if local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(model), revision=revision))


def model_info(model, description):
    model = Path(model)
    names = description["checkpoint_files"]
    digest = hashlib.sha256(b"tts-checkpoint-v1\0")
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
        frame_rate=description["frame_rate"],
        model_type=description["model_type"],
        model_digest=digest.hexdigest(),
        max_new_tokens=description["max_new_tokens"],
        max_batch_texts=MAX_INPUTS,
    )
