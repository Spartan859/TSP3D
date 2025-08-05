if [[ "$@" == *"-m"* ]]; then
    mode="mini"
else
    mode="large"
fi

data_root="/home/guowenxuan/lxy/TSP3D/data"

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

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=${cvd} python -m torch.distributed.launch \
    --nproc_per_node ${nproc_per_node} --master_port 12222 \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /home/guowenxuan/lxy/TSP3D/data/ \
    --val_freq 3 --batch_size 8 --save_freq 3 --print_freq 500 \
    --lr=5e-3 \
    --keep_trans_lr=5e-2 \
    --text_encoder_lr=1e-3 \
    --box_select_lr=4e-2 \
    --seg_lr=5e-2 \
    --voxel_size=0.02 --num_workers=1 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --log_dir /home/guowenxuan/lxy/TSP3D/outputs/logs \
    --lr_decay_epochs 30 45 100 120\
    --augment_det \
    --checkpoint_path /home/guowenxuan/lxy/TSP3D/outputs/ckpt_scanrefer.pth \
    # --checkpoint_path /home/gwx/lxy/TSP3D/outputs/logs/scanrefer/2025-07-10_10-27-52/ckpt_epoch_42.pth \
    # --checkpoint_path /home/gwx/lxy/TSP3D/outputs/logs/scanrefer/2025-07-03_18-03-55/ckpt_epoch_18.pth \
    # --checkpoint_path /home/gwx/gwx/3DVG/TSP3D/outputs/logs/scanrefer/2025-06-25_22-42-41/ckpt_epoch_51.pth \

torchrun --nproc_per_node=$nproc_per_node ~/lxy/occupy_GPU_cal.py