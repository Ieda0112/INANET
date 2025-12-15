#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/collect_env.sh [conda_env_name]
# If no env name is given, defaults to INA-DBNet_py38

ENV_NAME=${1:-INA-DBNet_py38}
OUTDIR="$(cd "$(dirname "$0")/.." && pwd)/outputs/processing_logs"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/env_probe_$(date +%Y%m%d_%H%M%S).log"

echo "Collecting environment info -> $OUTFILE"
{
  echo "===== date ====="
  date
  echo
  echo "===== python / platform ====="
  python - <<'PY'
import sys, platform
print('python:', sys.version.replace('\n',' '))
print('platform:', platform.platform())
PY
  echo
  echo "===== PyTorch / CUDA / cuDNN ====="
  python - <<'PY'
try:
    import torch
    print('torch:', torch.__version__)
    try:
        print('torch.cuda.is_available():', torch.cuda.is_available())
        print('torch.version.cuda (CUDA build):', torch.version.cuda)
    except Exception as e:
        print('torch.cuda info error:', e)
    try:
        print('cudnn version (torch.backends.cudnn.version()):', torch.backends.cudnn.version())
    except Exception as e:
        print('cudnn info error:', e)
except Exception as e:
    print('torch import error:', e)
PY
  echo
  echo "===== Key package versions (attempt imports) ====="
  python - <<'PY'
pkgs = ['torchvision','timm','numpy','cv2','opencv','opencv_python','mmcv','detectron2']
for p in pkgs:
    try:
        m = __import__(p)
        ver = getattr(m,'__version__',None)
        print(p, ver)
    except Exception as e:
        print(p, 'not installed')
PY
  echo
  echo "===== pip/installed (selected) ====="
  python - <<'PY'
import pkg_resources
installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
for key in sorted(installed):
    if any(k in key for k in ('torch','torchvision','timm','opencv','numpy','mmcv','detectron2','cupy')):
        print(key, installed[key])
PY
  echo
  echo "===== nvidia-smi GPU list ====="
  nvidia-smi -L || true
  echo
  echo "===== nvidia-smi GPU summary ====="
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv || true
} > "$OUTFILE" 2>&1

# Try conda activation and repeat PyTorch probe inside the env (if conda is available)
# This block attempts to activate the conda env and re-run the PyTorch probe to show env-specific builds.
if command -v conda >/dev/null 2>&1; then
  echo "\nAttempting to activate conda env: $ENV_NAME" >> "$OUTFILE" 2>&1
  # Try common conda activation methods
  if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
  if conda activate "$ENV_NAME" >/dev/null 2>&1; then
    echo "conda activate succeeded" >> "$OUTFILE" 2>&1
    python - <<'PY' >> "$OUTFILE" 2>&1
import torch
print('\n==== Inside conda env probe (torch) ====')
print('torch:', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
print('torch.version.cuda (CUDA build):', torch.version.cuda)
print('cudnn version (torch.backends.cudnn.version()):', torch.backends.cudnn.version())
PY
    conda deactivate >/dev/null 2>&1 || true
  else
    echo "conda activate $ENV_NAME failed or env not found" >> "$OUTFILE" 2>&1
  fi
else
  echo "conda not found on PATH; skipping conda env activation" >> "$OUTFILE" 2>&1
fi

echo "Saved environment probe to: $OUTFILE"

exit 0
