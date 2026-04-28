#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="ori_eval_4G" bash "${SCRIPT_DIR}/ori_eval.sh" --cudnn_benchmark false --gpu_mem_limit_gb 4
EXP_NAME="EA0_eval_4G" bash "${SCRIPT_DIR}/EA0_eval.sh" --cudnn_benchmark false --gpu_mem_limit_gb 4
EXP_NAME="EA02_eval_4G" bash "${SCRIPT_DIR}/EA02_num_sps_com_2400_eval.sh" --cudnn_benchmark false --gpu_mem_limit_gb 4