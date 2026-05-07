#!/usr/bin/env bash
# Switch dataset symlinks to point at the "large" or "mini" split.
#
# Usage:
#   bash scripts/switch_dataset_mode.sh large /root/lxy/TSP3D/data
#   bash scripts/switch_dataset_mode.sh mini  /root/lxy/TSP3D/data sr3d
#   bash scripts/switch_dataset_mode.sh mini  /root/lxy/TSP3D/data nr3d
#   bash scripts/switch_dataset_mode.sh mini  /root/lxy/TSP3D/data all
#
# Arguments:
#   $1  mode       "large" (default) or "mini"
#   $2  data_root  path to data directory (default: /root/lxy/TSP3D/data)
#   $3  dataset    "all" (default), "scanrefer", "sr3d", or "nr3d"
#
# Naming convention expected for mini/large split files:
#   ScanRefer/ScanRefer_filtered_train_${MODE}.txt
#   ScanRefer/ScanRefer_filtered_val_${MODE}.txt
#   meta_data/sr3d_train_scans_${MODE}.txt
#   meta_data/sr3d_test_scans_${MODE}.txt
#   meta_data/nr3d_train_scans_${MODE}.txt
#   meta_data/nr3d_test_scans_${MODE}.txt

set -euo pipefail

MODE="${1:-large}"
DATA_ROOT="${2:-/root/lxy/TSP3D/data}"
DATASET="${3:-all}"

if [[ "${MODE}" != "large" && "${MODE}" != "mini" ]]; then
    echo "Error: mode must be 'large' or 'mini', got '${MODE}'" >&2
    exit 1
fi

if [[ "${DATASET}" != "scanrefer" && "${DATASET}" != "sr3d" && "${DATASET}" != "nr3d" && "${DATASET}" != "all" ]]; then
    echo "Error: dataset must be one of 'scanrefer' | 'sr3d' | 'nr3d' | 'all', got '${DATASET}'" >&2
    exit 1
fi

link_required() {
    local src="$1"
    local dst="$2"
    if [[ ! -e "${src}" ]]; then
        echo "Error: required file not found: ${src}" >&2
        exit 1
    fi
    ln -sf "${src}" "${dst}"
}

switch_scanrefer() {
    link_required "${DATA_ROOT}/ScanRefer/ScanRefer_filtered_train_${MODE}.txt" \
                  "${DATA_ROOT}/ScanRefer/ScanRefer_filtered_train.txt"
    link_required "${DATA_ROOT}/ScanRefer/ScanRefer_filtered_val_${MODE}.txt" \
                  "${DATA_ROOT}/ScanRefer/ScanRefer_filtered_val.txt"
}

switch_sr3d() {
    link_required "${DATA_ROOT}/meta_data/sr3d_train_scans_${MODE}.txt" \
                  "${DATA_ROOT}/meta_data/sr3d_train_scans.txt"
    link_required "${DATA_ROOT}/meta_data/sr3d_test_scans_${MODE}.txt" \
                  "${DATA_ROOT}/meta_data/sr3d_test_scans.txt"
}

switch_nr3d() {
    link_required "${DATA_ROOT}/meta_data/nr3d_train_scans_${MODE}.txt" \
                  "${DATA_ROOT}/meta_data/nr3d_train_scans.txt"
    link_required "${DATA_ROOT}/meta_data/nr3d_test_scans_${MODE}.txt" \
                  "${DATA_ROOT}/meta_data/nr3d_test_scans.txt"
}

switch_v3scans() {
    link_required "${DATA_ROOT}/train_v3scans_${MODE}.pkl" \
                  "${DATA_ROOT}/train_v3scans.pkl"
    link_required "${DATA_ROOT}/val_v3scans_${MODE}.pkl" \
                  "${DATA_ROOT}/val_v3scans.pkl"
}

case "${DATASET}" in
    scanrefer)
        switch_scanrefer
        switch_v3scans
        ;;
    sr3d)
        switch_sr3d
        switch_v3scans
        ;;
    nr3d)
        switch_nr3d
        switch_v3scans
        ;;
    all)
        switch_scanrefer
        switch_sr3d
        switch_nr3d
        switch_v3scans
        ;;
esac

echo "Dataset mode switched: mode=${MODE}, dataset=${DATASET}, data_root=${DATA_ROOT}"
