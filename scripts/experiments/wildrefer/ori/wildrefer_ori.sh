# WildRefer baseline experiment, based on the modern TF32 script structure.
TSP3D_ENV="/mnt/share/micromamba/root/envs/TSP3D"
export PATH="${TSP3D_ENV}/bin:${PATH}"
export CONDA_PREFIX="${TSP3D_ENV}"
export LD_LIBRARY_PATH="${TSP3D_ENV}/lib:${LD_LIBRARY_PATH}"
unset PYTHONHOME PYTHONPATH
which python
python -c "import sys; print(sys.path)"

# Configurable parameters
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
BASE_NPROC_PER_NODE=4
BS_PER_GPU=${BS_PER_GPU:-7}
BASE_BS_PER_GPU=7
BASE_LR=${BASE_LR:-5e-4}
BASE_KEEP_TRANS_LR=${BASE_KEEP_TRANS_LR:-5e-4}
BASE_TEXT_ENCODER_LR=${BASE_TEXT_ENCODER_LR:-1e-5}
BASE_BOX_SELECT_LR=${BASE_BOX_SELECT_LR:-4e-4}
BASE_SEG_LR=${BASE_SEG_LR:-1e-4}
RNG_SEED=${RNG_SEED:-42}
MASTER_PORT_DEFAULT=${MASTER_PORT_DEFAULT:-11022}
MASTER_PORT_MAX=${MASTER_PORT_MAX:-12022}
GPU_FREE_MEM_THRESHOLD=${GPU_FREE_MEM_THRESHOLD:-1024}
GPU_FREE_UTIL_THRESHOLD=${GPU_FREE_UTIL_THRESHOLD:-10}
TF32_MATMUL=${TF32_MATMUL:-default}
TF32_CUDNN=${TF32_CUDNN:-default}
CVD="${CVD:-}"
WILDREFER_DATASET="${WILDREFER_DATASET:-strefer}"
NMS_PRE=${NMS_PRE:-50}
NMS_IOU_THR=${NMS_IOU_THR:-0.5}
NMS_SCORE_THR=${NMS_SCORE_THR:-0.01}
MAX_EPOCH=${MAX_EPOCH:-100}
LR_DECAY_EPOCHS="${LR_DECAY_EPOCHS:-30 60 85}"
PRUNE_THRESHOLD_0="${PRUNE_THRESHOLD_0:-}"
PRUNE_THRESHOLD_1="${PRUNE_THRESHOLD_1:-}"
WILDREFER_FRAME_NUM="${WILDREFER_FRAME_NUM:-3}"
WILDREFER_FUSE_FRAMES="${WILDREFER_FUSE_FRAMES:-0}"

data_root="${PWD}/data"
checkpoint_path="${CHECKPOINT_PATH:-}"

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

single_mode=0
train_extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--single)
            single_mode=1
            shift
            ;;
        --dataset)
            WILDREFER_DATASET="$2"
            shift 2
            ;;
        *)
            train_extra_args+=("$1")
            shift
            ;;
    esac
done

if [[ "${WILDREFER_DATASET}" != "strefer" && "${WILDREFER_DATASET}" != "liferefer" ]]; then
    echo "Error: WILDREFER_DATASET must be strefer or liferefer, got '${WILDREFER_DATASET}'" >&2
    exit 1
fi

if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="$(basename "${0}" .sh)_${WILDREFER_DATASET}"
fi

script_dir="$(dirname "$(readlink -f "$0")")"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel 2>/dev/null || readlink -f "${script_dir}/../../../..")"
log_root="${WILDREFER_LOG_ROOT:-${script_dir}}"
log_dir="${log_root}/${EXP_NAME}"
mkdir -p "${log_dir}"
find_free_gpus_script="${repo_root}/scripts/find_free_gpus.sh"
if [[ ! -f "${find_free_gpus_script}" ]]; then
    echo "Error: find_free_gpus.sh not found at '${find_free_gpus_script}' (repo_root='${repo_root}')" >&2
    exit 1
fi

lr_scale() {
    python - "$1" "$BASE_BS_PER_GPU" "$BS_PER_GPU" "$BASE_NPROC_PER_NODE" "$NPROC_PER_NODE" <<'PY'
import sys

base_lr = float(sys.argv[1])
base_bs_per_gpu = float(sys.argv[2])
bs_per_gpu = float(sys.argv[3])
base_nproc = int(sys.argv[4])
nproc = int(sys.argv[5])
print(base_lr * (bs_per_gpu * nproc) / (base_bs_per_gpu * base_nproc))
PY
}

nproc_per_node=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
if [[ ${single_mode} -eq 1 ]]; then
    nproc_per_node=1
fi

if [[ -n "${CVD}" ]]; then
    cvd="$(echo "${CVD}" | tr -d '[:space:]')"
    if [[ ! "${cvd}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Error: invalid CVD format '${CVD}', expected like 0,1,2"
        exit 1
    fi
    IFS=',' read -r -a cvd_arr <<< "${cvd}"
    if [[ ${#cvd_arr[@]} -ne ${nproc_per_node} ]]; then
        echo "Error: CVD has ${#cvd_arr[@]} GPUs, but NPROC_PER_NODE is ${nproc_per_node}"
        exit 1
    fi
else
    if ! cvd=$(bash "${find_free_gpus_script}" "${nproc_per_node}" "${GPU_FREE_MEM_THRESHOLD}" "${GPU_FREE_UTIL_THRESHOLD}"); then
        echo "Error: failed to find ${nproc_per_node} free GPUs with memory < ${GPU_FREE_MEM_THRESHOLD} MiB and utilization < ${GPU_FREE_UTIL_THRESHOLD}%" >&2
        exit 1
    fi
    cvd="$(echo "${cvd}" | tr -d '[:space:]')"
    IFS=',' read -r -a cvd_arr <<< "${cvd}"
    if [[ ${#cvd_arr[@]} -ne ${nproc_per_node} ]]; then
        echo "Error: found ${#cvd_arr[@]} free GPU(s) (${cvd}), but NPROC_PER_NODE is ${nproc_per_node}" >&2
        exit 1
    fi
fi

echo cvd: ${cvd}
echo log_dir: "${log_dir}"
echo dataset: "${WILDREFER_DATASET}"
echo nms_pre: "${NMS_PRE}"
if [[ -n "${checkpoint_path}" ]]; then
    echo checkpoint_path: "${checkpoint_path}"
fi

master_port=$(auto_find_free_port "${MASTER_PORT_DEFAULT}" "${MASTER_PORT_MAX}")
if [[ $? -ne 0 || -z "${master_port}" ]]; then
    echo "Error: failed to find a free master_port in range ${MASTER_PORT_DEFAULT}-${MASTER_PORT_MAX}" >&2
    exit 1
fi

echo master_port: ${master_port}

if [[ -x "${TSP3D_ENV}/bin/torchrun" ]]; then
    dist_launch_cmd=("${TSP3D_ENV}/bin/torchrun")
else
    dist_launch_cmd=(python -m torch.distributed.launch)
fi

checkpoint_args=()
if [[ -n "${checkpoint_path}" ]]; then
    checkpoint_args=(--checkpoint_path "${checkpoint_path}")
fi
lr_decay_args=()
for epoch in ${LR_DECAY_EPOCHS}; do
    lr_decay_args+=("${epoch}")
done
prune_args=()
if [[ -n "${PRUNE_THRESHOLD_0}" ]]; then
    prune_args+=(--prune_threshold_0 "${PRUNE_THRESHOLD_0}")
fi
if [[ -n "${PRUNE_THRESHOLD_1}" ]]; then
    prune_args+=(--prune_threshold_1 "${PRUNE_THRESHOLD_1}")
fi
wildrefer_frame_args=(--wildrefer_frame_num "${WILDREFER_FRAME_NUM}")
if [[ "${WILDREFER_FUSE_FRAMES}" == "1" || "${WILDREFER_FUSE_FRAMES}" == "true" || "${WILDREFER_FUSE_FRAMES}" == "TRUE" ]]; then
    wildrefer_frame_args+=(--wildrefer_fuse_frames)
fi
wildrefer_extra_args=()
if [[ -n "${WILDREFER_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    wildrefer_extra_args=(${WILDREFER_EXTRA_ARGS})
fi

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=${cvd} "${dist_launch_cmd[@]}" \
    --nproc_per_node ${nproc_per_node} --master_port ${master_port} \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ${data_root}/ \
    --val_freq 3 --batch_size ${BS_PER_GPU} --save_freq 3 --print_freq 500 \
    --max_epoch ${MAX_EPOCH} \
    --lr=$(lr_scale "${BASE_LR}") \
    --keep_trans_lr=$(lr_scale "${BASE_KEEP_TRANS_LR}") \
    --text_encoder_lr=$(lr_scale "${BASE_TEXT_ENCODER_LR}") \
    --box_select_lr=$(lr_scale "${BASE_BOX_SELECT_LR}") \
    --seg_lr=$(lr_scale "${BASE_SEG_LR}") \
    --voxel_size=0.01 --num_workers=8 \
    --dataset ${WILDREFER_DATASET} --test_dataset ${WILDREFER_DATASET} \
    --detect_intermediate \
    --log_dir "${log_dir}" \
    --lr_decay_epochs "${lr_decay_args[@]}" \
    --tf32_matmul ${TF32_MATMUL} \
    --tf32_cudnn ${TF32_CUDNN} \
    --nms_pre ${NMS_PRE} \
    --nms_iou_thr ${NMS_IOU_THR} \
    --nms_score_thr ${NMS_SCORE_THR} \
    "${wildrefer_frame_args[@]}" \
    "${wildrefer_extra_args[@]}" \
    "${prune_args[@]}" \
    "${train_extra_args[@]}" \
    --rng_seed ${RNG_SEED} \
    "${checkpoint_args[@]}"

if [[ "${OCCUPY_GPU_AFTER_TRAIN:-0}" == "1" ]]; then
    echo "Post-train GPU occupy enabled (OCCUPY_GPU_AFTER_TRAIN=1)."
    sleep infinity
else
    echo "Skip post-train GPU occupy (set OCCUPY_GPU_AFTER_TRAIN=1 to enable)."
fi
