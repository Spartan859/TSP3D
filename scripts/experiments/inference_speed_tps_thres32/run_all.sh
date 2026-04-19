#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/EA0_eval.sh" --cudnn_benchmark false --use_deterministic_algorithms true
bash "${SCRIPT_DIR}/EA02_num_sps_com_2400_eval.sh" --cudnn_benchmark false --use_deterministic_algorithms true
bash "${SCRIPT_DIR}/ori_eval.sh" --cudnn_benchmark false --use_deterministic_algorithms true
