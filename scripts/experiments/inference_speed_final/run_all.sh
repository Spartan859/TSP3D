#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mini)
            EXTRA_ARGS+=("-m")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

EXP_NAME="ori_eval" bash "${SCRIPT_DIR}/ori_eval.sh" --cudnn_benchmark false "${EXTRA_ARGS[@]}"
EXP_NAME="EA0_eval" bash "${SCRIPT_DIR}/EA0_eval.sh" --cudnn_benchmark false "${EXTRA_ARGS[@]}"
EXP_NAME="EA02_eval" bash "${SCRIPT_DIR}/EA02_eval.sh" --cudnn_benchmark false "${EXTRA_ARGS[@]}"
EXP_NAME="EA0_seg_eval" bash "${SCRIPT_DIR}/EA0_seg_eval.sh" --cudnn_benchmark false "${EXTRA_ARGS[@]}"
EXP_NAME="EA0_seg_SEGEA_eval" bash "${SCRIPT_DIR}/EA0_seg_SEGEA_eval.sh" --cudnn_benchmark false "${EXTRA_ARGS[@]}"
