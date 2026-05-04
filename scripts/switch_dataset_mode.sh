#!/usr/bin/env bash
# Switch the ScanRefer / v3scans symlinks to point at the "large" or "mini" split.
#
# Usage:
#   bash scripts/switch_dataset_mode.sh large /root/lxy/TSP3D/data
#   bash scripts/switch_dataset_mode.sh mini  /root/lxy/TSP3D/data
#
# Arguments:
#   $1  mode       "large" (default) or "mini"
#   $2  data_root  path to data directory (default: /root/lxy/TSP3D/data)

MODE="${1:-large}"
DATA_ROOT="${2:-/root/lxy/TSP3D/data}"

if [[ "${MODE}" != "large" && "${MODE}" != "mini" ]]; then
    echo "Error: mode must be 'large' or 'mini', got '${MODE}'" >&2
    exit 1
fi

ln -sf "${DATA_ROOT}/ScanRefer/ScanRefer_filtered_train_${MODE}.txt" \
       "${DATA_ROOT}/ScanRefer/ScanRefer_filtered_train.txt"
ln -sf "${DATA_ROOT}/ScanRefer/ScanRefer_filtered_val_${MODE}.txt" \
       "${DATA_ROOT}/ScanRefer/ScanRefer_filtered_val.txt"
ln -sf "${DATA_ROOT}/train_v3scans_${MODE}.pkl" \
       "${DATA_ROOT}/train_v3scans.pkl"
ln -sf "${DATA_ROOT}/val_v3scans_${MODE}.pkl" \
       "${DATA_ROOT}/val_v3scans.pkl"

echo "Dataset mode switched to: ${MODE} (data_root=${DATA_ROOT})"
