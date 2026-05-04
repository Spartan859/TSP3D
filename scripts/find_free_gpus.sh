#!/usr/bin/env bash
# Find N free GPUs and print them as a comma-separated list (e.g. "0,2,3").
# Exits with code 1 if not enough free GPUs are found.
#
# Usage (source or call as subprocess):
#   cvd=$(bash scripts/find_free_gpus.sh 1)          # find 1 free GPU
#   cvd=$(bash scripts/find_free_gpus.sh 4 1024 10)  # 4 GPUs, mem<1024MiB, util<10%
#
# Arguments:
#   $1  N              number of GPUs needed (default: 1)
#   $2  MEM_THRESHOLD  max used memory in MiB to consider a GPU free (default: 1024)
#   $3  UTIL_THRESHOLD max GPU utilization % to consider a GPU free (default: 10)

N="${1:-1}"
MEM_TH="${2:-1024}"
UTIL_TH="${3:-10}"

python - "$N" "$MEM_TH" "$UTIL_TH" <<'PY'
import subprocess, sys
need     = int(sys.argv[1])
mem_th   = int(sys.argv[2])
util_th  = int(sys.argv[3])
try:
    out = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'
    ], encoding='utf-8', errors='ignore')
except Exception:
    print('ERROR: nvidia-smi failed', file=sys.stderr)
    sys.exit(2)
free = []
for line in out.splitlines():
    parts = [x.strip() for x in line.split(',')]
    if len(parts) != 3:
        continue
    try:
        idx, mem, util = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        continue
    if mem < mem_th and util < util_th:
        free.append(str(idx))
if len(free) < need:
    print(
        f'WARNING: need {need} free GPU(s) (mem<{mem_th}MiB, util<{util_th}%), '
        f'found {len(free)}: {",".join(free) or "none"}',
        file=sys.stderr
    )
if free:
    print(','.join(free[:need]))
    sys.exit(0)
else:
    sys.exit(1)
PY
