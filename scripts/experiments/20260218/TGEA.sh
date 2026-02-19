export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

if [[ "$@" == *"-m"* ]]; then
    mode="mini"
else
    mode="large"
fi

data_root="/root/lxy/TSP3D/data"

NPROC_PER_NODE=4
CVD_START=4
BASE_BS=28
CUR_BS=28
BASE_LR=5e-4
BASE_KEEP_TRANS_LR=5e-4
BASE_TEXT_ENCODER_LR=1e-5
BASE_BOX_SELECT_LR=4e-4

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
if [[ "$@" == *"-s"* ]]; then
    nproc_per_node=1
fi
# let cvd = CVD_START,...,CVD_START+nproc_per_node-1
cvd=$(seq -s, ${CVD_START} $((CVD_START + nproc_per_node - 1)))
echo cvd: ${cvd}
echo log_dir: "$(dirname "$(readlink -f "$0")")"

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=${cvd} torchrun \
    --nproc_per_node ${nproc_per_node} --master_port 12222 \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ${data_root}/ \
    --val_freq 3 --batch_size $(bs_per_gpu "${CUR_BS}" "${nproc_per_node}") --save_freq 3 --print_freq 500 \
    --lr=$(lr_scale "${BASE_LR}") \
    --keep_trans_lr=$(lr_scale "${BASE_KEEP_TRANS_LR}") \
    --text_encoder_lr=$(lr_scale "${BASE_TEXT_ENCODER_LR}") \
    --box_select_lr=$(lr_scale "${BASE_BOX_SELECT_LR}") \
    --voxel_size=0.01 --num_workers=8 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --log_dir "$(dirname "$(readlink -f "$0")")" \
    --augment_det \
    --lr_decay_epochs 50 75 \
    --use_external_attn_bi_layer0 \
    --use_text_guided_external_attn_bi_layer0 \
    --use_film_text_guided_external_attn_bi_layer0 \
    --checkpoint_path /root/lxy/TSP3D_ori/scripts/experiments/20260218/scanrefer/2026-02-18_07-03-44/ckpt_epoch_24.pth \
    # --checkpoint_path /root/lxy/TSP3D_ori/scripts/experiments/20260218/scanrefer/2026-02-18_13-15-37/ckpt_epoch_6.pth \
    # --use_seg \
    # --clip_norm 1.0 \
    # --window_size 5 \
    # --quant_size 4 \
    # --swin_layer_num 2 \
    # --use_Swin \
    # --swin_drop_path 0.3 \
    # --checkpoint_path /home/guowenxuan/lxy/TSP3D/scripts/experiments/20250924/scanrefer/2025-09-24_19-43-49/ckpt_epoch_36.pth \
    # --use_F3_CA \
    # --use_seg \
    # --use_Mq 2 \
    # --checkpoint_path /home/guowenxuan/lxy/TSP3D/outputs/ckpt_scanrefer.pth \
    # --checkpoint_path /home/guowenxuan/lxy/TSP3D/outputs/logs/scanrefer/2025-08-29_19-48-22/ckpt_epoch_81.pth \
    # --checkpoint_path /home/guowenxuan/lxy/TSP3D/outputs/logs/scanrefer/2025-08-21_00-44-56/ckpt_epoch_87.pth \
    # --checkpoint_path /home/guowenxuan/lxy/TSP3D/outputs/logs/scanrefer/2025-08-18_10-58-47/ckpt_epoch_81.pth \
    # --checkpoint_path /home/guowenxuan/lxy/TSP3D/outputs/ckpt_scanrefer.pth \
    # --checkpoint_path /home/gwx/lxy/TSP3D/outputs/logs/scanrefer/2025-07-10_10-27-52/ckpt_epoch_42.pth \
    # --checkpoint_path /home/gwx/lxy/TSP3D/outputs/logs/scanrefer/2025-07-03_18-03-55/ckpt_epoch_18.pth \
    # --checkpoint_path /home/gwx/gwx/3DVG/TSP3D/outputs/logs/scanrefer/2025-06-25_22-42-41/ckpt_epoch_51.pth \

torchrun --nproc_per_node=$nproc_per_node ~/lxy/occupy_GPU_cal.py