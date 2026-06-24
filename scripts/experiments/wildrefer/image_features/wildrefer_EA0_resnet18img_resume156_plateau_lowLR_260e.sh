# Resume the best ResNet18 image-feature STRefer checkpoint with low LR and plateau decay.
# With NPROC_PER_NODE=8 and BS_PER_GPU=7, these BASE_* values scale to
# lr_base=2.5e-5, lr_text=5e-7, lr_select=2e-5, lr_seg=5e-6.
script_dir="$(dirname "$(readlink -f "$0")")"

export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PWD}/scripts/experiments/wildrefer/image_features/wildrefer_EA0_resnet18img_prune035_400e_strefer/strefer/2026-06-24_01-05-41/ckpt_epoch_156.pth}"
export EXP_NAME="${EXP_NAME:-wildrefer_EA0_resnet18img_resume156_plateau_lowLR_260e_${WILDREFER_DATASET:-strefer}}"

export BASE_LR="${BASE_LR:-1.25e-5}"
export BASE_KEEP_TRANS_LR="${BASE_KEEP_TRANS_LR:-1.25e-5}"
export BASE_TEXT_ENCODER_LR="${BASE_TEXT_ENCODER_LR:-2.5e-7}"
export BASE_BOX_SELECT_LR="${BASE_BOX_SELECT_LR:-1e-5}"
export BASE_SEG_LR="${BASE_SEG_LR:-2.5e-6}"
export MAX_EPOCH="${MAX_EPOCH:-260}"

export WILDREFER_EXTRA_ARGS="${WILDREFER_EXTRA_ARGS:---use_external_attn_bi_layer 0 --use_text_guided_external_attn_bi_layer 0 --use_film_text_guided_external_attn_bi_layer 0 --lr-scheduler plateau --plateau_metric official_acc0.50 --plateau_factor 0.5 --plateau_patience 5 --plateau_threshold 1e-4 --plateau_cooldown 1 --plateau_min_lr 1e-7}"

bash "${script_dir}/wildrefer_EA0_resnet18img_prune035_400e.sh" "$@"
