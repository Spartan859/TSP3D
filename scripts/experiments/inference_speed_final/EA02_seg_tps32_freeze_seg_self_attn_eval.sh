#!/usr/bin/env bash
# Measure ScanRefer inference speed for EA02+seg+refine+tps32 freeze + seg self-attn.
set -euo pipefail

default_tsp3d_env="${CONDA_PREFIX:-/mnt/share/micromamba/root/envs/TSP3D_4090}"
TSP3D_ENV="${TSP3D_ENV:-${default_tsp3d_env}}"
export PATH="${TSP3D_ENV}/bin:${PATH}"
export CONDA_PREFIX="${TSP3D_ENV}"
export LD_LIBRARY_PATH="${TSP3D_ENV}/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONHOME PYTHONPATH

script_dir="$(dirname "$(readlink -f "$0")")"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel 2>/dev/null || readlink -f "${script_dir}/../../..")"
data_root="${DATA_ROOT:-${repo_root}/data}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BASE_BS="${BASE_BS:-28}"
CUR_BS="${CUR_BS:-1}"
BASE_LR="${BASE_LR:-5e-4}"
BASE_KEEP_TRANS_LR="${BASE_KEEP_TRANS_LR:-5e-4}"
BASE_TEXT_ENCODER_LR="${BASE_TEXT_ENCODER_LR:-1e-5}"
BASE_BOX_SELECT_LR="${BASE_BOX_SELECT_LR:-4e-4}"
BASE_SEG_LR="${BASE_SEG_LR:-1e-4}"
MASTER_PORT_DEFAULT="${MASTER_PORT_DEFAULT:-11022}"
MASTER_PORT_MAX="${MASTER_PORT_MAX:-12022}"
GPU_FREE_MEM_THRESHOLD="${GPU_FREE_MEM_THRESHOLD:-1024}"
GPU_FREE_UTIL_THRESHOLD="${GPU_FREE_UTIL_THRESHOLD:-10}"
TF32_MATMUL="${TF32_MATMUL:-default}"
TF32_CUDNN="${TF32_CUDNN:-default}"
CVD="${CVD:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXTERNAL_ATTN_COEF="${EXTERNAL_ATTN_COEF:-4}"
COM_THRESHOLD="${COM_THRESHOLD:-0.15}"
NUM_SAMPLES_COM="${NUM_SAMPLES_COM:-2400}"
FPS_WARMUP_ITERS="${FPS_WARMUP_ITERS:-100}"
FPS_MAX_ITERS="${FPS_MAX_ITERS:--1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${repo_root}/scripts/experiments/scanrefer_EA02_SEG_freeze/EA02_SEG_freeze_seg_self_attn/scanrefer/2026-07-04_21-16-50/ckpt_epoch_318.pth}"

auto_find_free_port() {
    local start=${1:-${MASTER_PORT_DEFAULT}}
    local end=${2:-${MASTER_PORT_MAX}}
    python - "$start" "$end" <<'PY'
import socket, sys
start = int(sys.argv[1])
end = int(sys.argv[2])
for port in range(start, end):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('127.0.0.1', port))
        except OSError:
            continue
        print(port)
        sys.exit(0)
print('NO_FREE_PORT', file=sys.stderr)
sys.exit(1)
PY
}

choose_free_gpus() {
    local need=${1:-1}
    local mem_th=${2:-${GPU_FREE_MEM_THRESHOLD}}
    local util_th=${3:-${GPU_FREE_UTIL_THRESHOLD}}
    python - "$need" "$mem_th" "$util_th" <<'PY'
import subprocess, sys
need = int(sys.argv[1])
mem_th = int(sys.argv[2])
util_th = int(sys.argv[3])
try:
    out = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'
    ], encoding='utf-8', errors='ignore')
except Exception:
    sys.exit(2)
free = []
for line in out.splitlines():
    parts = [x.strip() for x in line.split(',')]
    if len(parts) != 3:
        continue
    try:
        idx = int(parts[0]); mem = int(parts[1]); util = int(parts[2])
    except ValueError:
        continue
    if mem < mem_th and util < util_th:
        free.append(str(idx))
if len(free) < need:
    sys.exit(1)
print(','.join(free[:need]))
PY
}

mode="large"
single_mode=0
train_extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mini)
            mode="mini"
            shift
            ;;
        -s|--single)
            single_mode=1
            shift
            ;;
        *)
            train_extra_args+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${EXP_NAME:-}" ]]; then
    EXP_NAME="$(basename "$0" .sh)"
fi

log_dir="${script_dir}/${EXP_NAME}"
mkdir -p "${log_dir}"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Error: checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi
run_time="${TSP3D_RUN_TIME:-$(date +%Y-%m-%d_%H-%M-%S)}"
export TSP3D_RUN_TIME="${run_time}"
run_dir="${log_dir}/scanrefer/${run_time}"
mkdir -p "${run_dir}"
terminal_log="${run_dir}/terminal.log"
exec > >(tee -a "${terminal_log}") 2>&1
echo "terminal_log: ${terminal_log}"
echo "run_dir: ${run_dir}"
date
which python
python -c "import sys; print(sys.path)"

lr_scale() {
    python - "$1" "$BASE_BS" "$CUR_BS" <<'PY'
import sys
base_lr = float(sys.argv[1])
base_bs = float(sys.argv[2])
cur_bs = float(sys.argv[3])
print(base_lr * cur_bs / base_bs)
PY
}

bs_per_gpu() {
    python - "$1" "$2" <<'PY'
import sys
total_bs = float(sys.argv[1])
ngpu = int(sys.argv[2])
per = total_bs / ngpu
if per != int(per):
    print(f"Warning: total batch size {total_bs} not divisible by GPUs {ngpu}", file=sys.stderr)
print(int(per))
PY
}

bash "${repo_root}/scripts/switch_dataset_mode.sh" "${mode}" "${data_root}" scanrefer

nproc_per_node="${NPROC_PER_NODE}"
if [[ ${single_mode} -eq 1 ]]; then
    nproc_per_node=1
fi

if [[ -n "${CVD}" ]]; then
    cvd="$(echo "${CVD}" | tr -d '[:space:]')"
    if [[ ! "${cvd}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Error: invalid CVD format '${CVD}', expected like 0,1,2" >&2
        exit 1
    fi
    IFS=',' read -r -a cvd_arr <<< "${cvd}"
    if [[ ${#cvd_arr[@]} -ne ${nproc_per_node} ]]; then
        echo "Error: CVD has ${#cvd_arr[@]} GPUs, but NPROC_PER_NODE is ${nproc_per_node}" >&2
        exit 1
    fi
else
    if ! cvd=$(choose_free_gpus "${nproc_per_node}" "${GPU_FREE_MEM_THRESHOLD}" "${GPU_FREE_UTIL_THRESHOLD}"); then
        echo "Error: failed to find ${nproc_per_node} free GPUs with memory < ${GPU_FREE_MEM_THRESHOLD} MiB and utilization < ${GPU_FREE_UTIL_THRESHOLD}%" >&2
        exit 1
    fi
fi

echo cvd: "${cvd}"
echo log_dir: "${log_dir}"
echo checkpoint_path: "${CHECKPOINT_PATH}"

master_port=$(auto_find_free_port "${MASTER_PORT_DEFAULT}" "${MASTER_PORT_MAX}")
if [[ -z "${master_port}" ]]; then
    echo "Error: failed to find a free master_port in range ${MASTER_PORT_DEFAULT}-${MASTER_PORT_MAX}" >&2
    exit 1
fi
echo master_port: "${master_port}"

if [[ -x "${TSP3D_ENV}/bin/torchrun" ]]; then
    dist_launch_cmd=("${TSP3D_ENV}/bin/torchrun")
else
    dist_launch_cmd=(python -m torch.distributed.launch)
fi

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=${cvd} "${dist_launch_cmd[@]}" \
    --nproc_per_node "${nproc_per_node}" --master_port "${master_port}" \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root "${data_root}/" \
    --val_freq 3 --batch_size "$(bs_per_gpu "${CUR_BS}" "${nproc_per_node}")" --save_freq 3 --print_freq 500 \
    --lr="$(lr_scale "${BASE_LR}")" \
    --keep_trans_lr="$(lr_scale "${BASE_KEEP_TRANS_LR}")" \
    --text_encoder_lr="$(lr_scale "${BASE_TEXT_ENCODER_LR}")" \
    --box_select_lr="$(lr_scale "${BASE_BOX_SELECT_LR}")" \
    --seg_lr="$(lr_scale "${BASE_SEG_LR}")" \
    --voxel_size=0.01 --num_workers "${NUM_WORKERS}" \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --log_dir "${log_dir}" \
    --augment_det \
    --lr_decay_epochs 50 100 \
    --tf32_matmul "${TF32_MATMUL}" \
    --tf32_cudnn "${TF32_CUDNN}" \
    "${train_extra_args[@]}" \
    --use_external_attn_bi_layer 0 2 \
    --use_text_guided_external_attn_bi_layer 0 2 \
    --use_film_text_guided_external_attn_bi_layer 0 2 \
    --external_attn_coef "${EXTERNAL_ATTN_COEF}" \
    --external_attn_k_keep0 64 \
    --external_attn_k_keep1 64 \
    --external_attn_k_com 64 \
    --external_attn_k_seg128 64 \
    --external_attn_k_seg64 64 \
    --com_threshold "${COM_THRESHOLD}" \
    --num_samples_com "${NUM_SAMPLES_COM}" \
    --use_refine \
    --use_seg \
    --top_pts_threshold 32 \
    --top_pts_threshold_det 32 \
    --use_seg_external_self_attn \
    --eval \
    --measure_fps \
    --fps_warmup_iters "${FPS_WARMUP_ITERS}" \
    --fps_max_iters "${FPS_MAX_ITERS}" \
    --checkpoint_path "${CHECKPOINT_PATH}"

if [[ "${OCCUPY_GPU_AFTER_TRAIN:-0}" == "1" ]]; then
    echo "Post-train GPU occupy enabled (OCCUPY_GPU_AFTER_TRAIN=1)."
    sleep infinity
else
    echo "Skip post-train GPU occupy (set OCCUPY_GPU_AFTER_TRAIN=1 to enable)."
fi
