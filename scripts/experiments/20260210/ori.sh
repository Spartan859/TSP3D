export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

if [[ "$@" == *"-m"* ]]; then
    mode="mini"
else
    mode="large"
fi

data_root="/root/lxy/TSP3D/data"

ln -sf ${data_root}/ScanRefer/ScanRefer_filtered_train_${mode}.txt \
    ${data_root}/ScanRefer/ScanRefer_filtered_train.txt
ln -sf ${data_root}/ScanRefer/ScanRefer_filtered_val_${mode}.txt \
    ${data_root}/ScanRefer/ScanRefer_filtered_val.txt
ln -sf ${data_root}/train_v3scans_${mode}.pkl \
    ${data_root}/train_v3scans.pkl
ln -sf ${data_root}/val_v3scans_${mode}.pkl \
    ${data_root}/val_v3scans.pkl

nproc_per_node=$(nvidia-smi -L | wc -l)
if [[ "$@" == *"-s"* ]]; then
    nproc_per_node=1
fi
# let cvd = 0,1,2,...,nproc_per_node-1
cvd=$(seq -s, 0 $((nproc_per_node-1)))
echo cvd: ${cvd}
echo log_dir: "$(dirname "$(readlink -f "$0")")"

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=${cvd} torchrun \
    --nproc_per_node ${nproc_per_node} --master_port 12222 \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ${data_root}/ \
    --val_freq 3 --batch_size 7 --save_freq 3 --print_freq 500 \
    --lr=5e-4 \
    --keep_trans_lr=5e-4 \
    --voxel_size=0.01 --num_workers=8 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --log_dir "$(dirname "$(readlink -f "$0")")" \
    --augment_det \
    --lr_decay_epochs 50 75 \
    # --use_external_attn_bi_layer0 \
    # --use_text_guided_external_attn_bi_layer0 \
    # --use_film_text_guided_external_attn_bi_layer0 \
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