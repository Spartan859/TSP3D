#!/usr/bin/env bash

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

# 传入 -m 时使用 mini，否则默认 large
if [[ "$@" == *"-m"* ]]; then
    mode="mini"
else
    mode="large"
fi

data_root="/root/lxy/TSP3D_ori/data"

# 切换 ScanRefer 标注与 pkl
ln -sf ${data_root}/ScanRefer/ScanRefer_filtered_train_${mode}.txt \
    ${data_root}/ScanRefer/ScanRefer_filtered_train.txt
ln -sf ${data_root}/ScanRefer/ScanRefer_filtered_val_${mode}.txt \
    ${data_root}/ScanRefer/ScanRefer_filtered_val.txt
ln -sf ${data_root}/train_v3scans_${mode}.pkl \
    ${data_root}/train_v3scans.pkl
ln -sf ${data_root}/val_v3scans_${mode}.pkl \
    ${data_root}/val_v3scans.pkl

log_dir="/root/lxy/TSP3D_ori/output/logs/test"
mkdir -p "${log_dir}"

echo "mode: ${mode}"
echo "log_dir: ${log_dir}"

for epoch in $(seq 252 3 258); do
    ckpt="/root/lxy/TSP3D_ori/scripts/experiments/TGEA_convrefine/TGEA_refine/scanrefer/2026-03-15_13-30-43/ckpt_epoch_${epoch}.pth"
    
    echo "========================================="
    echo "Evaluating checkpoint: ${ckpt}"
    echo "========================================="

    TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
        --nproc_per_node 2 --master_port 25222 \
        train_dist_mod.py \
        --use_color \
        --weight_decay 0.0005 \
        --data_root ${data_root}/ \
        --val_freq 3 --batch_size 8 --save_freq 6 --print_freq 500 \
        --lr=5e-4 --keep_trans_lr=5e-4 --voxel_size=0.01 --num_workers=8 \
        --dataset scanrefer --test_dataset scanrefer \
        --detect_intermediate \
        --log_dir "${log_dir}" \
        --lr_decay_epochs 50 75 \
        --augment_det \
        --use_external_attn_bi_layer 0 \
        --use_text_guided_external_attn_bi_layer 0 \
        --use_film_text_guided_external_attn_bi_layer 0 \
        --checkpoint_path ${ckpt} \
        --eval \
        --use_seg \
        --use_refine \
        --use_seg_external_self_attn
done