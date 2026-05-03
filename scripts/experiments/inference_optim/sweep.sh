#!/usr/bin/env bash
# Usage:
#   bash scripts/experiments/inference_optim/sweep.sh \
#       --checkpoint_path /path/to/ckpt.pth \
#       [--sweep_params '{"com_threshold":[0.05,0.10,0.15,0.20,0.30]}'] \
#       [--output_jsonl /tmp/sweep_results.jsonl]
#
# GPU selection:
#   - Set CVD to a comma-separated list to use specific GPUs: CVD=0,1,2 bash sweep.sh ...
#   - Otherwise all free GPUs are detected automatically.
#   - Each GPU runs independently (single-card, single-batch).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -n "${CVD:-}" ]]; then
    gpu_ids="${CVD}"
else
    gpu_ids=$(bash "${REPO_ROOT}/scripts/find_free_gpus.sh" 16 1024 10 2>/dev/null || true)
    if [[ -z "${gpu_ids}" ]]; then
        echo "Warning: no free GPUs detected, falling back to GPU 0." >&2
        gpu_ids="0"
    fi
fi
echo "Using GPU(s): ${gpu_ids}"

cd "${REPO_ROOT}"

python scripts/experiments/inference_optim/sweep.py \
    --gpu_ids "${gpu_ids}" \
    --data_root /root/lxy/TSP3D/data/ \
    --use_external_attn_bi_layer 0 2 \
    --use_film_text_guided_external_attn_bi_layer 0 2 \
    --sweep_mode grid \
    --output_jsonl /tmp/sweep_results.jsonl \
    "$@"
