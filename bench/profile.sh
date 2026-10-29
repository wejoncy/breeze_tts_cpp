#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:?usage: profile.sh MODEL_DIR [OUTPUT_DIR]}
output=${2:-"$project/artifacts/final"}
mkdir -p "$output"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem --format=csv > "$output/gpu.csv"
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true \
  -o "$output/runtime" "$project/build/tts" --model "$model" \
  --text '你好，欢迎使用语音合成。这是一段原生引擎生成的测试语音，我们正在检查连续输出是否清晰流畅。' \
  --max-frames 24 --repeat 3 --output "$output/profile.wav" > "$output/nsys.log" 2>&1
nsys export --type=sqlite --force-overwrite=true --output="$output/runtime.sqlite" "$output/runtime.nsys-rep"
python3 "$project/bench/audit_trace.py" "$output/runtime.sqlite" > "$output/trace-audit.json"
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum --format csv "$output/runtime.nsys-rep" > "$output/nsys-stats.csv"
metrics=gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
ncu --clock-control none --kernel-name regex:nvjet --launch-skip 8 --launch-count 1 \
  --metrics "$metrics" --force-overwrite --export "$output/gemm-decode" \
  "$project/build/kernels" 12288 2048 1 4 > "$output/ncu-decode.log" 2>&1
ncu --clock-control none --kernel-name regex:nvjet --launch-skip 8 --launch-count 1 \
  --metrics "$metrics" --force-overwrite --export "$output/gemm-prefill" \
  "$project/build/kernels" 13824 1152 256 4 > "$output/ncu-prefill.log" 2>&1
ncu --clock-control none --kernel-name regex:rms_bf16_kernel --launch-count 1 \
  --metrics "$metrics" --force-overwrite --export "$output/rms-decode" \
  "$project/build/kernels" --rms 1 1024 100 > "$output/ncu-rms.log" 2>&1
ncu --clock-control none --kernel-name regex:depth_attention_kernel --launch-count 1 \
  --metrics "$metrics" --force-overwrite --export "$output/depth-attention" \
  "$project/build/kernels" --depth-attention 1 16 100 > "$output/ncu-attention.log" 2>&1
for report in gemm-decode gemm-prefill rms-decode depth-attention; do
  ncu --import "$output/$report.ncu-rep" --page raw --csv > "$output/$report.csv"
done
