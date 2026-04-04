export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

mode="large"
single_mode=0
for arg in "$@"; do
    case "$arg" in
        -m|--mini)
            mode="mini"
            ;;
        -s|--single)
            single_mode=1
            ;;
    esac
done

custom_cvd="${CVD:-}"
custom_master_port="${MASTER_PORT:-}"
train_extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cvd)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --cvd requires a value like 0,1,2"
                exit 1
            fi
            custom_cvd="$2"
            shift 2
            ;;
        --cvd=*)
            custom_cvd="${1#*=}"
            shift
            ;;
        -c)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -c requires a value like 0,1,2"
                exit 1
            fi
            custom_cvd="$2"
            shift 2
            ;;
        --master_port)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --master_port requires a value like 11022"
                exit 1
            fi
            custom_master_port="$2"
            shift 2
            ;;
        --master_port=*)
            custom_master_port="${1#*=}"
            shift
            ;;
        *)
            train_extra_args+=("$1")
            shift
            ;;
    esac
done

data_root="/root/lxy/TSP3D/data"

if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="$(basename "${0}" .sh)"
fi

log_dir="$(dirname "$(readlink -f "$0")")/${EXP_NAME}"
mkdir -p "${log_dir}"

NPROC_PER_NODE=4
CVD_START=0
BASE_BS=28
CUR_BS=28
BASE_LR=5e-4
BASE_KEEP_TRANS_LR=5e-4
BASE_TEXT_ENCODER_LR=1e-5
BASE_BOX_SELECT_LR=4e-4
# BASE_LR=5e-5
# BASE_KEEP_TRANS_LR=5e-5
# BASE_TEXT_ENCODER_LR=1e-6
# BASE_BOX_SELECT_LR=4e-5
BASE_SEG_LR=1e-4
MASTER_PORT_DEFAULT=11022
ENABLE_TF32="${ENABLE_TF32:-1}"

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

ln -sf ${data_root}/ScanRefer/ScanRefer_filtered_train_${mode}.txt \
    ${data_root}/ScanRefer/ScanRefer_filtered_train.txt
ln -sf ${data_root}/ScanRefer/ScanRefer_filtered_val_${mode}.txt \
    ${data_root}/ScanRefer/ScanRefer_filtered_val.txt
ln -sf ${data_root}/train_v3scans_${mode}.pkl \
    ${data_root}/train_v3scans.pkl
ln -sf ${data_root}/val_v3scans_${mode}.pkl \
    ${data_root}/val_v3scans.pkl

nproc_per_node=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
if [[ ${single_mode} -eq 1 ]]; then
    nproc_per_node=1
fi

if [[ -n "${custom_cvd}" ]]; then
    cvd="$(echo "${custom_cvd}" | tr -d '[:space:]')"
    if [[ ! "${cvd}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Error: invalid cvd format '${custom_cvd}', expected like 0,1,2"
        exit 1
    fi
    IFS=',' read -r -a cvd_arr <<< "${cvd}"
    nproc_per_node=${#cvd_arr[@]}
else
    # let cvd = CVD_START,...,CVD_START+nproc_per_node-1
    cvd=$(seq -s, ${CVD_START} $((CVD_START + nproc_per_node - 1)))
fi

echo cvd: ${cvd}
echo log_dir: "${log_dir}"

master_port="${custom_master_port:-${MASTER_PORT_DEFAULT}}"
echo master_port: ${master_port}
echo enable_tf32: ${ENABLE_TF32}

tf32_args=()
if [[ "${ENABLE_TF32}" == "1" ]]; then
    tf32_args+=(--enable_tf32)
fi

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=${cvd} torchrun \
    --nproc_per_node ${nproc_per_node} --master_port ${master_port} \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ${data_root}/ \
    --val_freq 3 --batch_size $(bs_per_gpu "${CUR_BS}" "${nproc_per_node}") --save_freq 3 --print_freq 500 \
    --lr=$(lr_scale "${BASE_LR}") \
    --keep_trans_lr=$(lr_scale "${BASE_KEEP_TRANS_LR}") \
    --text_encoder_lr=$(lr_scale "${BASE_TEXT_ENCODER_LR}") \
    --box_select_lr=$(lr_scale "${BASE_BOX_SELECT_LR}") \
    --seg_lr=$(lr_scale "${BASE_SEG_LR}") \
    --voxel_size=0.01 --num_workers=32 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --log_dir "${log_dir}" \
    --augment_det \
    --lr_decay_epochs 50 75\
    --checkpoint_path /root/lxy/TSP3D/scripts/experiments/A100/ori/scanrefer/2026-04-04_07-17-06/ckpt_epoch_24.pth \
    --load_optimizer \
    --load_scheduler \
    "${tf32_args[@]}" \
    "${train_extra_args[@]}" \
    # --use_external_attn_bi_layer 0 2\
    # --use_text_guided_external_attn_bi_layer 0 2\
    # --use_film_text_guided_external_attn_bi_layer 0 2\
    # --use_refine \
    # --use_seg \
    # --use_seg_external_self_attn \
    
if [[ "${OCCUPY_GPU_AFTER_TRAIN:-0}" == "1" ]]; then
    echo "Post-train GPU occupy enabled (OCCUPY_GPU_AFTER_TRAIN=1)."
    torchrun --nproc_per_node=$nproc_per_node ~/lxy/occupy_GPU_cal.py
else
    echo "Skip post-train GPU occupy (set OCCUPY_GPU_AFTER_TRAIN=1 to enable)."
fi