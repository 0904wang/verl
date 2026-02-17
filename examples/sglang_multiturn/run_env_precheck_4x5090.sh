#!/usr/bin/env bash

set -euo pipefail
set -x

nvidia-smi
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import ray; print(ray.__version__)"

ulimit -n 65535
ulimit -n
df -h /dev/shm
