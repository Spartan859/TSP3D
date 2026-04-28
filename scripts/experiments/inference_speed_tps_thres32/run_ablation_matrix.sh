#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

extract_metric() {
    local log_file="$1"
    local prefix="$2"
    grep -F "${prefix}" "${log_file}" | tail -n 1 | sed -E "s/.*${prefix}//"
}

run_one() {
    local tag="$1"
    local script_name="$2"
    local exp_name="${tag}_detail"

    echo "=== Running ${tag} (${script_name}) ==="
    EXP_NAME="${exp_name}" bash "${SCRIPT_DIR}/${script_name}" \
        --cudnn_benchmark false \
        --use_deterministic_algorithms false \
        --measure_fps_detail

    local run_dir="${SCRIPT_DIR}/${exp_name}/scanrefer"
    local latest_dir
    latest_dir="$(ls -1dt "${run_dir}"/* | head -n 1)"
    local log_file="${latest_dir}/log.txt"

    local acc25 acc50 fps ext_ratio ext_total detail
    acc25="$(extract_metric "${log_file}" "3dcnn Acc0.25: Top-1: ")"
    acc50="$(extract_metric "${log_file}" "3dcnn Acc0.50: Top-1: ")"
    fps="$(extract_metric "${log_file}" "FPS(single-card): ")"
    ext_total="$(extract_metric "${log_file}" "External-attn-affected module time(s): ")"
    ext_ratio="$(extract_metric "${log_file}" "External-attn-affected path FPS(eqv): ")"
    detail="$(extract_metric "${log_file}" "Detailed stage time(s): ")"

    echo "run_dir=${latest_dir}"
    echo "acc@0.25=${acc25}"
    echo "acc@0.50=${acc50}"
    echo "fps=${fps}"
    echo "external=${ext_total}"
    echo "external_ratio=${ext_ratio}"
    echo "stage_detail=${detail}"
    echo
}

run_one "ori" "ori_eval.sh"
run_one "EA0" "EA0_eval.sh"
run_one "EA02" "EA02_num_sps_com_2400_eval.sh"
