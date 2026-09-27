#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null
command -v git >/dev/null
command -v tmux >/dev/null
command -v flock >/dev/null
nvidia-smi
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
mkdir -p outputs/remote_environment
exec 9>outputs/remote_environment/bootstrap.lock
flock -n 9 || { echo "Another bootstrap is running"; exit 1; }
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
# RTX 50-series/Blackwell needs a CUDA >= 12.8 wheel. The host driver is newer.
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c configs/week9_remote_requirements.txt -r configs/week9_remote_requirements.txt -e '.[scale,dev,generation]'
python -m pip check
python -c 'import torch, transformers, peft, bitsandbytes; assert torch.cuda.is_available(), "CUDA unavailable: check host driver/template"; print(torch.__version__, transformers.__version__, peft.__version__, bitsandbytes.__version__)'
python -m pip freeze > outputs/remote_environment/pip_freeze.txt
nvidia-smi -q > outputs/remote_environment/nvidia_smi.txt
echo "Environment ready. Next: bash scripts/run_week9_7b_remote.sh preflight"
