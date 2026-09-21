#!/usr/bin/env bash
# Set up a training environment on Linux with an NVIDIA GPU.
#
#   ./setup.sh && source .venv/bin/activate && python -m qroute.pipeline
#
# Written against Arch Linux with an RTX 5090, which needs two things that
# trip up a default `pip install torch`:
#
#   * Blackwell cards are compute capability sm_120, and only CUDA 12.8+
#     wheels carry sm_120 kernels. An older wheel still reports
#     `torch.cuda.is_available() == True` and then dies on the first matmul
#     with "no kernel image is available for execution on the device".
#   * Arch tracks Python far ahead of the PyTorch wheel index. If the system
#     interpreter has no matching wheel, pip silently resolves nothing or
#     falls back to a source build. This script checks before installing.

set -euo pipefail

CUDA_CHANNEL="${CUDA_CHANNEL:-cu128}"
VENV="${VENV:-.venv}"

echo "==> looking for a supported Python"
PY=""
for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
        major=${ver%%.*}; minor=${ver##*.}
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 13 ]; then
            PY="$cand"; break
        fi
        echo "    $cand is $ver (PyTorch wheels may not cover it yet)"
    fi
done

if [ -z "$PY" ]; then
    echo "    no Python 3.10-3.13 found."
    echo "    Arch ships a newer interpreter than the PyTorch wheel index covers."
    echo "    Install one, e.g.:  sudo pacman -S python313"
    echo "    then re-run:        PY=python3.13 ./setup.sh"
    exit 1
fi
echo "    using $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

echo "==> checking the NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "    nvidia-smi not found. Install the driver first:"
    echo "      sudo pacman -S nvidia nvidia-utils   (or nvidia-open for Blackwell)"
    echo "    Blackwell (RTX 50xx) needs driver 570 or newer."
fi

echo "==> creating $VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

echo "==> installing torch from the $CUDA_CHANNEL channel"
pip install --no-cache-dir torch --index-url "https://download.pytorch.org/whl/$CUDA_CHANNEL"

echo "==> installing the rest"
pip install --no-cache-dir -r requirements.txt

echo "==> verifying the GPU can actually run a kernel"
python -c "from qroute.pipeline import environment_report; environment_report(require_cuda=True)"

cat <<'EOF'

Ready. Start the full run with:

    source .venv/bin/activate
    python -m qroute.pipeline 2>&1 | tee runs/train.log

EOF
