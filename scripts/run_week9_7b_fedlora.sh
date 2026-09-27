#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

threads="$(nproc)"
if (( threads > 16 )); then threads=16; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$threads}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$threads}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$threads}"

mode="${1:-preflight}"
case "$mode" in
  preflight|smoke|run|retry|report) ;;
  *) echo "Mode: preflight|smoke|run|retry|report"; exit 2 ;;
esac

profile=qwen7b-fedlora
output=outputs/week9_7b_fedlora_v1
if [[ "$mode" == smoke ]]; then
  output=outputs/week9_7b_fedlora_smoke_v1
fi

mkdir -p outputs/remote_environment
exec 9>outputs/remote_environment/week9_7b_fedlora.lock
flock -n 9 || { echo "A Week 9 FedLoRA job is already running"; exit 1; }
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked source changed since release freeze; deploy a clean release before training"
  exit 1
fi

workers="${WEEK9_FEDLORA_WORKERS_PER_GPU:-1}"
if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "WEEK9_FEDLORA_WORKERS_PER_GPU must be a positive integer"
  exit 2
fi

base=(python scripts/run_week9_generation.py --profile "$profile" --gpu 0 --output-root "$output")

preflight() {
  "${base[@]}" --dry-run
  "${base[@]}" --prepare-only
  "${base[@]}" --plan-only
  CUDA_VISIBLE_DEVICES="" python -m pytest -q \
    tests/test_spectral_fedlora.py \
    tests/test_persistent_factor_baselines.py \
    tests/test_fed_lora_baselines.py \
    tests/test_week8_spectral_integration.py \
    tests/test_week9_generation.py \
    tests/test_week9_7b_fedlora.py
}

check_gpu() {
  WEEK9_FEDLORA_WORKERS_PER_GPU="$workers" CUDA_VISIBLE_DEVICES=0 python -c \
    'import os, torch; assert torch.cuda.is_available(), "CUDA unavailable"; workers=int(os.environ["WEEK9_FEDLORA_WORKERS_PER_GPU"]); p=torch.cuda.get_device_properties(0); free,total=torch.cuda.mem_get_info(); required=(20+14*(workers-1))*2**30; print({"gpu":p.name,"capability":f"{p.major}.{p.minor}","free_GiB":free/2**30,"total_GiB":total/2**30,"workers":workers,"required_free_GiB":required/2**30}); assert free>=required, "Insufficient free VRAM; reduce workers or use a larger GPU"'
}

report() {
  python scripts/analyze_week9_generation.py --input-dir "$output" --target flexlora
}

case "$mode" in
  preflight)
    preflight
    ;;
  report)
    report
    ;;
  smoke|run|retry)
    busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
    if [[ -n "$busy" ]]; then echo "GPU0 already has compute processes: $busy"; exit 1; fi
    check_gpu
    command=("${base[@]}" --workers-per-gpu "$workers")
    if [[ "$mode" == smoke ]]; then command+=(--smoke); fi
    if [[ "$mode" == retry ]]; then command+=(--retry-incomplete); fi
    set +e
    "${command[@]}"
    status=$?
    if [[ "$mode" != smoke ]]; then report; report_status=$?; else report_status=0; fi
    set -e
    printf '%s\n' "$status" > outputs/remote_environment/last_fedlora_worker_exit.txt
    if [[ "$status" != 0 ]]; then exit "$status"; fi
    exit "$report_status"
    ;;
esac
