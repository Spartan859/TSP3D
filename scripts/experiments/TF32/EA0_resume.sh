# Resume EA0 training from epoch 96, using modern script template
TSP3D_ENV="/mnt/share/micromamba/root/envs/TSP3D"
export PATH="${TSP3D_ENV}/bin:${PATH}"
export CONDA_PREFIX="${TSP3D_ENV}"
export LD_LIBRARY_PATH="${TSP3D_ENV}/lib:${LD_LIBRARY_PATH}"
unset PYTHONHOME PYTHONPATH
which python
python -c "import sys; print(sys.path)"

# Configurable parameters
NPROC_PER_NODE=8
BASE_NPROC_PER_NODE=8
BS_PER_GPU=4
BASE_BS_PER_GPU=4
BASE_LR=5e-4
BASE_KEEP_TRANS_LR=5e-4
BASE_TEXT_ENCODER_LR=1e-5
BASE_BOX_SELECT_LR=4e-4
BASE_SEG_LR=1e-4
RNG_SEED=42
RNG_SEED=42
MASTER_PORT_DEFAULT=11022
MASTER_PORT_MAX=12022
GPU_FREE_MEM_THRESHOLD=1024
GPU_FREE_UTIL_THRESHOLD=10
TF32_MATMUL=default
TF32_CUDNN=default
CVD=""

data_root="${PWD}/data"

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

if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="$(basename "${0}" .sh)"
fi

log_dir="$(dirname "$(readlink -f "$0")")/${EXP_NAME}"
mkdir -p "${log_dir}"
scripts_dir="$(readlink -f "$(dirname "$(readlink -f "$0")")/../..")"
find_free_gpus_script="${scripts_dir}/find_free_gpus.sh"
switch_dataset_mode_script="${scripts_dir}/switch_dataset_mode.sh"

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

if ! bash "${switch_dataset_mode_script}" "${mode}" "${data_root}"; then
    echo "Error: failed to switch dataset mode to '${mode}'" >&2
    exit 1
fi

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

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=${cvd} "${dist_launch_cmd[@]}" \
    --nproc_per_node ${nproc_per_node} --master_port ${master_port} \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ${data_root}/ \
    --val_freq 3 --batch_size ${BS_PER_GPU} --save_freq 3 --print_freq 500 \
    --lr=$(lr_scale "${BASE_LR}") \
    --keep_trans_lr=$(lr_scale "${BASE_KEEP_TRANS_LR}") \
    --text_encoder_lr=$(lr_scale "${BASE_TEXT_ENCODER_LR}") \
    --box_select_lr=$(lr_scale "${BASE_BOX_SELECT_LR}") \
    --seg_lr=$(lr_scale "${BASE_SEG_LR}") \
    --voxel_size=0.01 --num_workers=8 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --log_dir "${log_dir}" \
    --augment_det \
    --lr_decay_epochs 50 75\
    --load_optimizer \
    --load_scheduler \
    --tf32_matmul ${TF32_MATMUL} \
    --tf32_cudnn ${TF32_CUDNN} \
    "${train_extra_args[@]}" \
    --use_external_attn_bi_layer 0\
    --use_text_guided_external_attn_bi_layer 0\
    --use_film_text_guided_external_attn_bi_layer 0\
    --rng_seed ${RNG_SEED}\
    --external_attn_k_keep0 64\
    --external_attn_k_keep1 64\
    --external_attn_k_com 64\
    --external_attn_k_seg128 64\
    --external_attn_k_seg64 64\
    --checkpoint_path ${PWD}/scripts/experiments/TF32/EA0/scanrefer/2026-04-18_17-45-08/ckpt_epoch_96.pth \
    # --com_threshold 0.15\
    # --num_samples_com 1800\
    # --use_refine \
    # --use_seg \
    # --use_seg_external_self_attn \
    
if [[ "${OCCUPY_GPU_AFTER_TRAIN:-0}" == "1" ]]; then
    echo "Post-train GPU occupy enabled (OCCUPY_GPU_AFTER_TRAIN=1)."
    # TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=${cvd} "${dist_launch_cmd[@]}" --nproc_per_node=$nproc_per_node ${PWD}/occupy_GPU_cal.py
    sleep infinity
else
    echo "Skip post-train GPU occupy (set OCCUPY_GPU_AFTER_TRAIN=1 to enable)."
fi
