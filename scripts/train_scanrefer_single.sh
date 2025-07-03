TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch \
    --nproc_per_node 2 --master_port 12222 \
    train_dist_mod.py \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /home/gwx/lxy/TSP3D/data/ \
    --val_freq 3 --batch_size 8 --save_freq 3 --print_freq 500 \
    --lr=5e-4 --keep_trans_lr=5e-4 --voxel_size=0.02 --num_workers=8 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --log_dir /home/gwx/lxy/TSP3D/outputs/logs \
    --lr_decay_epochs 30 45 \
    --augment_det \
    # --checkpoint_path /home/gwx/gwx/3DVG/TSP3D/outputs/logs/scanrefer/2025-06-25_22-42-41/ckpt_epoch_51.pth \