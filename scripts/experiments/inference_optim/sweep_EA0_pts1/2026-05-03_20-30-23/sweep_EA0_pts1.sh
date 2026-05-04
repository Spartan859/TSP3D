#!/usr/bin/env bash
# Sweep prune_threshold_1 for the 20260216 / epoch-204 checkpoint.
#
# Architecture flags are copied verbatim from:
#   scripts/experiments/20260216/text_guided_EA_FiLM_totalB28.sh
#
# Usage:
#   bash scripts/experiments/inference_optim/sweep_EA0_pts1.sh
#   CVD=2 bash scripts/experiments/inference_optim/sweep_EA0_pts1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CKPT="/share/lxy/TSP3D_ori/scripts/experiments/20260216/scanrefer/2026-02-16_11-14-34/ckpt_epoch_204.pth"
DATA_ROOT="/root/lxy/TSP3D/data"
MODE="large"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mini) MODE="mini"; shift ;;
        *) break ;;
    esac
done

bash "${REPO_ROOT}/scripts/switch_dataset_mode.sh" "${MODE}" "${DATA_ROOT}"

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_DIR="${SCRIPT_DIR}/${SCRIPT_NAME}/${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

OUTPUT_JSONL="${RUN_DIR}/results.jsonl"

if [[ -n "${CVD:-}" ]]; then
    gpu_ids="${CVD}"
else
    # Find as many free GPUs as possible (up to 16); ignore exit code so partial results are used
    gpu_ids=$(bash "${REPO_ROOT}/scripts/find_free_gpus.sh" 16 1024 10 2>/dev/null || true)
    echo "Found GPU(s): ${gpu_ids}"
    if [[ -z "${gpu_ids}" ]]; then
        echo "Warning: no free GPUs detected, falling back to GPU 0." >&2
        gpu_ids="0"
    fi
fi
echo "Using GPU(s): ${gpu_ids}"
echo "Run dir: ${RUN_DIR}"
echo ""

cd "${REPO_ROOT}"

python scripts/experiments/inference_optim/sweep.py \
    --gpu_ids "${gpu_ids}" \
    --checkpoint_path "${CKPT}" \
    --data_root "${DATA_ROOT}/" \
    --use_color \
    --voxel_size 0.01 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --augment_det \
    --use_external_attn_bi_layer0 \
    --use_text_guided_external_attn_bi_layer0 \
    --use_film_text_guided_external_attn_bi_layer0 \
    --sweep_params '{"prune_threshold_0":[0.25,0.3,0.35,0.4], "prune_threshold_1":[0.4,0.5,0.6,0.7,0.8,0.9]}' \
    --sweep_mode grid \
    --output_jsonl "${OUTPUT_JSONL}"

SUMMARY_MD="${RUN_DIR}/summary.md"
echo ""
python scripts/experiments/inference_optim/summarize.py \
    --input "${OUTPUT_JSONL}" --markdown \
    --extra_cols avg_latency_ms fps mem_peak_res_mib ext_total_ms \
    | tee "${SUMMARY_MD}"
echo ""
echo "Summary written to ${SUMMARY_MD}"
