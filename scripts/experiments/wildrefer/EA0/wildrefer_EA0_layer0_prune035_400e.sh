# WildRefer EA0 reproduction: layer0 external attention, prune=0.35, nms_pre=1, 400 epochs.
script_dir="$(dirname "$(readlink -f "$0")")"

export BASE_LR="${BASE_LR:-2.5e-4}"
export BASE_KEEP_TRANS_LR="${BASE_KEEP_TRANS_LR:-2.5e-4}"
export BASE_TEXT_ENCODER_LR="${BASE_TEXT_ENCODER_LR:-5e-6}"
export BASE_BOX_SELECT_LR="${BASE_BOX_SELECT_LR:-2e-4}"
export BASE_SEG_LR="${BASE_SEG_LR:-5e-5}"
export NMS_PRE="${NMS_PRE:-1}"
export MAX_EPOCH="${MAX_EPOCH:-400}"
export LR_DECAY_EPOCHS="${LR_DECAY_EPOCHS:-400}"
export PRUNE_THRESHOLD_0="${PRUNE_THRESHOLD_0:-0.35}"
export PRUNE_THRESHOLD_1="${PRUNE_THRESHOLD_1:-0.35}"
export WILDREFER_LOG_ROOT="${WILDREFER_LOG_ROOT:-${script_dir}}"
if [[ -z "${WILDREFER_EXTRA_ARGS}" ]]; then
    export WILDREFER_EXTRA_ARGS="--use_external_attn_bi_layer 0 --use_text_guided_external_attn_bi_layer 0 --use_film_text_guided_external_attn_bi_layer 0"
fi

if [[ -z "${EXP_NAME}" ]]; then
    export EXP_NAME="$(basename "$0" .sh)_${WILDREFER_DATASET:-strefer}"
fi

bash "${script_dir}/../ori/wildrefer_ori.sh" "$@"
