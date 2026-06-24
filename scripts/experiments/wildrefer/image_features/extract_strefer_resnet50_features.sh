# Extract STRefer ResNet50 layer3 image feature maps for projected point fusion.
TSP3D_ENV="/mnt/share/micromamba/root/envs/TSP3D"
export PATH="${TSP3D_ENV}/bin:${PATH}"
export CONDA_PREFIX="${TSP3D_ENV}"
export LD_LIBRARY_PATH="${TSP3D_ENV}/lib:${LD_LIBRARY_PATH}"
unset PYTHONHOME PYTHONPATH

OUTPUT="${OUTPUT:-${PWD}/data/WildRefer/strefer_resnet50_layer3_448.h5}"
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHECKPOINT_ARG=()
if [[ -n "${WILDREFER_IMAGE_CKPT:-}" ]]; then
    CHECKPOINT_ARG=(--checkpoint "${WILDREFER_IMAGE_CKPT}")
fi
PRETRAINED_ARG=()
if [[ "${WILDREFER_IMAGE_PRETRAINED:-0}" == "1" || "${WILDREFER_IMAGE_PRETRAINED:-0}" == "true" || "${WILDREFER_IMAGE_PRETRAINED:-0}" == "TRUE" ]]; then
    PRETRAINED_ARG=(--pretrained)
fi

python scripts/extract_wildrefer_image_features.py \
    --data_root data/ \
    --dataset strefer \
    --splits train test \
    --output "${OUTPUT}" \
    --arch resnet50 \
    --device "${DEVICE}" \
    --image_size "${IMAGE_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --overwrite \
    "${PRETRAINED_ARG[@]}" \
    "${CHECKPOINT_ARG[@]}"
