# WildRefer baseline with conservative LR, nms_pre=1, and 400 epochs.
script_dir="$(dirname "$(readlink -f "$0")")"

export BASE_LR="${BASE_LR:-2.5e-4}"
export BASE_KEEP_TRANS_LR="${BASE_KEEP_TRANS_LR:-2.5e-4}"
export BASE_TEXT_ENCODER_LR="${BASE_TEXT_ENCODER_LR:-5e-6}"
export BASE_BOX_SELECT_LR="${BASE_BOX_SELECT_LR:-2e-4}"
export BASE_SEG_LR="${BASE_SEG_LR:-5e-5}"
export NMS_PRE="${NMS_PRE:-1}"
export MAX_EPOCH="${MAX_EPOCH:-400}"

if [[ -z "${EXP_NAME}" ]]; then
    export EXP_NAME="$(basename "$0" .sh)_${WILDREFER_DATASET:-strefer}"
fi

bash "${script_dir}/wildrefer_ori.sh" "$@"
