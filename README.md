# TTS.cpp

Native C++17/CUDA inference for [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2),
with reference voice cloning, instruction/CFG controls, offline WAV, GPU batches
and streaming PCM. Python is optional; native inference needs no PyTorch or model conversion.

## Build

Requires CMake 3.28+, a C++17 compiler, CUDA 13+, Rust/Cargo, Git, libsndfile and
libsoxr. Rust is a build dependency only. CUDA/cuBLAS and audio libraries are
found locally; [FetchContent](cmake/Dependencies.cmake) fetches pinned JSON,
spdlog, Corrosion, CUTLASS, FlashAttention and missing audio headers.

```sh
cmake -S . -B build -DTTS_BUILD_PYTHON=OFF
cmake --build build --config Release --parallel 4
```

Select GPU targets with `-DCMAKE_CUDA_ARCHITECTURES=86` for an RTX 3060 or `90`
for an H100. BF16 is the default. `--quantization int8` supports SM80+;
`--quantization fp8` supports SM90+. Both use per-channel weights and dynamic
per-token activations; FlashAttention and its Q/K/V/KV storage remain BF16.
Quantization speed and accuracy depend on the workload; FP8 is not always faster.

On Windows, use an x64 MSVC developer shell, CUDA and the MSVC Rust toolchain.
Set `-DCMAKE_PREFIX_PATH=...` to the x64 audio library installation prefixes and
`-DTTS_RUNTIME_DIRECTORIES=...` for extra DLL search directories. Multi-configuration
builds produce `build/Release/tts.exe`; required DLLs are deployed beside the binaries.
Windows cross-compilation has been checked, but a full native NVCC/MSVC build and
Windows GPU execution remain unverified.

For offline builds, populate dependencies first, then use
`-DFETCHCONTENT_FULLY_DISCONNECTED=ON`. Fresh offline builds need local source
overrides (`-DFETCHCONTENT_SOURCE_DIR_<NAME>=...`) and a populated Cargo cache.
Run `cargo fetch --locked --manifest-path tokenizer/Cargo.toml` before going offline;
use `CARGO_NET_OFFLINE=true` during the build.

## Run

Download the checkpoint separately into `models/Breeze-TTS-2`, including
`audio_tokenizer/`, or pass another location with `--model`. Weights and media
are not distributed here. The model has its own
[license](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).

```sh
build/tts --model models/Breeze-TTS-2 --text '你好，欢迎使用语音合成。' --output speech.wav

# Supply your own reference file and its exact transcript.
build/tts --model models/Breeze-TTS-2 --text '今天很高兴见到你。' \
  --ref-audio reference.wav --ref-text '参考音频对应的文字。' --output clone.wav

# JSONL: one object per line, with text and optional instruction/ref_audio_path/ref_text.
build/tts --model models/Breeze-TTS-2 --batch requests.jsonl --output batch

# Streaming: signed PCM16, 24 kHz, mono. ffplay is an optional external player.
build/tts --model models/Breeze-TTS-2 --text 'Hello.' --stream - --output speech.wav \
  | ffplay -f s16le -ar 24000 -ac 1 -
```

Batch output is `batch.0.wav`, `batch.1.wav`, etc.; lanes have independent EOS.
Add `--instruction TEXT --cfg-scale 3` for instruction guidance, or
`--cfg-scale-ref 2 --cfg-scale-ins 3` for independent reference/instruction guidance.
`--temperature 0` enables greedy sampling. `--help` lists sampling, logging,
context and quantization options. `--repeat 3` separates cold and warm requests;
metrics go to stderr. Keep an engine process alive to reuse weights, memory and graphs.

## Python and HTTP

Install dependencies into your Python environment and enable the bridge:

```sh
python -m pip install -r python/requirements.txt
cmake -S . -B build -DTTS_BUILD_PYTHON=ON
cmake --build build --config Release --parallel 4
python examples/server.py --model models/Breeze-TTS-2 --host 127.0.0.1 --port 8000 --max-batch 8
```

Use `-DPython_EXECUTABLE=...` to select an interpreter. For another build directory,
set `TTS_LIBRARY` or pass `library=` to the Python engine.

```python
import sys
sys.path.insert(0, "python")
from tts_cpp import Engine

with Engine(model="models/Breeze-TTS-2", max_batch=8) as engine:
    with engine.prepare_reference("reference.wav", "参考音频的准确文字。") as ref:
        pcm, stats = engine.generate(
            [{"text": "你好。", "reference": ref}, {"text": "Welcome."}],
            {"max_frames": 100},
        )
```

`pcm` contains one PCM byte buffer per request. A third `generate` argument can
receive `(batch_index, pcm_bytes)` incrementally; returning `False` cancels generation.
Prepared references can be reused; `codes()` / `reference_from_codes()` support persistence.

| HTTP endpoint | Result |
|---|---|
| `POST /v1/audio/speech` | Streaming PCM, or WAV with `"stream": false` |
| `POST /v1/audio/batch` | `{"requests": [...]}` to a ZIP of WAVs and metrics |
| `GET /health` | Readiness |

Speech requests accept JSON text/instruction/sampling fields; reference input is
base64 `ref_audio` plus `ref_text`, or a multipart audio upload. Larger HTTP batches
use bounded GPU microbatches. Disconnects cancel generation.

## Validate and profile

With the model and Python bridge available:

```sh
make test MODEL_DIR=models/Breeze-TTS-2
make test-fp8 MODEL_DIR=models/Breeze-TTS-2
make test-int8 MODEL_DIR=models/Breeze-TTS-2
make build/kernels
python bench/benchmark.py --model models/Breeze-TTS-2 --output artifacts/benchmark.json
bash bench/profile.sh models/Breeze-TTS-2
```

`profile.sh` uses nsys for timelines and ncu for kernel compute/bandwidth metrics.
Run GPU measurements sequentially, separating load/capture from warm execution.
Keep debug dumps disabled when profiling. `tests/run_reference.py` and
`tests/run_parity_oracles.py` provide numerical checks with the original Python
model's separate development environment. For the Hoppou comparison, configure
`checkpoint`, `reference_audio` and `reference_text` in `bench/hoppou_plan.json`
with your local inputs. Generated audio and reports remain ignored.

See [architecture and ownership](docs/DESIGN.md) and the
[C interface](include/tts/tts.h). CPU inference, continuous batching, multi-GPU
and arbitrary model architectures are not supported.
