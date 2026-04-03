#!/usr/bin/env bash
set -euo pipefail

# Route B small grid search for com_trans-related hyperparameters.
# Base runner: TGEA_comEA_gnd.sh
# Grid: com_threshold x num_samples_com (3x3)
# Default attention setting here isolates com_trans by only enabling bi_layer 2.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/TGEA_comEA_gnd.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
    echo "Error: base script not found: ${BASE_SCRIPT}"
    exit 1
fi

# Pass-through runtime flags/examples:
#   ./TGEA_comEA_gnd_routeB_grid.sh --cvd 0,1,2,3 -m
#   CVD=0,1 ./TGEA_comEA_gnd_routeB_grid.sh
PASS_ARGS=("$@")

THRESHOLDS=(0.10 0.15 0.20)
SAMPLES=(1200 2400 3600)
GPU_GROUP_A="0,1,2,3"
GPU_GROUP_B="4,5,6,7"
MASTER_PORT_A=11022
MASTER_PORT_B=11023

# Keep a compact run log
GRID_LOG="${SCRIPT_DIR}/routeB_grid_runs_$(date +%Y%m%d_%H%M%S).log"

echo "Route B grid start: $(date)" | tee -a "${GRID_LOG}"
echo "Base script: ${BASE_SCRIPT}" | tee -a "${GRID_LOG}"
echo "Thresholds: ${THRESHOLDS[*]}" | tee -a "${GRID_LOG}"
echo "Num samples com: ${SAMPLES[*]}" | tee -a "${GRID_LOG}"

total=$(( ${#THRESHOLDS[@]} * ${#SAMPLES[@]} ))
jobs=()
for th in "${THRESHOLDS[@]}"; do
    for ns in "${SAMPLES[@]}"; do
        jobs+=("${th}|${ns}")
    done
done

run_one() {
    local global_idx="$1"
    local th="$2"
    local ns="$3"
    local cvd="$4"
    local master_port="$5"
    local exp="TGEA_comEA_gnd_RB_t${th//./p}_n${ns}"

    echo "[$global_idx/$total] start ${exp} on GPUs ${cvd} (master_port=${master_port})" | tee -a "${GRID_LOG}"

    # Use only bi_layer=2 to focus on com_trans.
    # Keep external/text-guided/film settings aligned to avoid mode mismatch.
    EXP_NAME="${exp}" bash "${BASE_SCRIPT}" "${PASS_ARGS[@]}" \
        --cvd "${cvd}" \
        --master_port "${master_port}" \
        --com_threshold "${th}" \
        --num_samples_com "${ns}" 2>&1 | tee -a "${GRID_LOG}"

    echo "[$global_idx/$total] done  ${exp} on GPUs ${cvd} (master_port=${master_port})" | tee -a "${GRID_LOG}"
}

i=0

# Two-slot dynamic pool:
# - slot A binds to GPUs 0,1,2,3
# - slot B binds to GPUs 4,5,6,7
# Whenever one slot is free, dispatch the next pending experiment immediately.
pid_a=""
pid_b=""

launch_on_slot() {
    local slot="$1"
    if [[ $i -ge ${#jobs[@]} ]]; then
        return 1
    fi

    local spec="${jobs[$i]}"
    local idx=$((i + 1))
    local th="${spec%%|*}"
    local ns="${spec##*|}"
    i=$((i + 1))

    if [[ "${slot}" == "A" ]]; then
        run_one "${idx}" "${th}" "${ns}" "${GPU_GROUP_A}" "${MASTER_PORT_A}" &
        pid_a=$!
    else
        run_one "${idx}" "${th}" "${ns}" "${GPU_GROUP_B}" "${MASTER_PORT_B}" &
        pid_b=$!
    fi
    return 0
}

# Initial fill
launch_on_slot "A" || true
launch_on_slot "B" || true

# Bash-compatible dynamic scheduler (no wait -p / wait -n dependency).
while [[ $i -lt ${#jobs[@]} || -n "${pid_a}" || -n "${pid_b}" ]]; do
    # Reap finished slot A
    if [[ -n "${pid_a}" ]] && ! kill -0 "${pid_a}" 2>/dev/null; then
        set +e
        wait "${pid_a}"
        rc_a=$?
        set -e
        if [[ ${rc_a} -ne 0 ]]; then
            echo "Warning: slot A process ${pid_a} exited with code ${rc_a}" | tee -a "${GRID_LOG}"
        fi
        pid_a=""
    fi

    # Reap finished slot B
    if [[ -n "${pid_b}" ]] && ! kill -0 "${pid_b}" 2>/dev/null; then
        set +e
        wait "${pid_b}"
        rc_b=$?
        set -e
        if [[ ${rc_b} -ne 0 ]]; then
            echo "Warning: slot B process ${pid_b} exited with code ${rc_b}" | tee -a "${GRID_LOG}"
        fi
        pid_b=""
    fi

    # Fill any free slot immediately
    if [[ -z "${pid_a}" ]]; then
        launch_on_slot "A" || true
    fi
    if [[ -z "${pid_b}" ]]; then
        launch_on_slot "B" || true
    fi

    # Avoid busy loop while at least one slot is running
    if [[ -n "${pid_a}" || -n "${pid_b}" ]]; then
        sleep 5
    fi
done

echo "Route B grid done: $(date)" | tee -a "${GRID_LOG}"
