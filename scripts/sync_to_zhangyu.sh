#!/usr/bin/env bash
set -euo pipefail

LOCAL_BASE="/share/lxy/TSP3D_ori"
REMOTE_USER="root"
REMOTE_HOST="120.48.163.20"
REMOTE_PORT=8551
REMOTE_BASE="/mnt/share/algorithm/kimi/cache/lxy/TSP3D"

usage() {
  echo "Usage: $0 <relative-path>" >&2
  echo "Example: $0 scripts/experiments/20260216/.../ckpt_epoch_204.pth" >&2
  exit 2
}

if [ "$#" -ne 1 ]; then
  usage
fi

REL="$1"
REL="${REL#/}"

SRC="$LOCAL_BASE/$REL"
if [ ! -e "$SRC" ]; then
  echo "Error: source '$SRC' does not exist" >&2
  exit 1
fi

PARENT="$(dirname "$REL")"
if [ "$PARENT" = "." ]; then
  REMOTE_PARENT="$REMOTE_BASE"
else
  REMOTE_PARENT="$REMOTE_BASE/$PARENT"
fi

echo "Ensuring remote directory exists: $REMOTE_PARENT"
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_PARENT'"

echo "Rsyncing $SRC -> $REMOTE_USER@$REMOTE_HOST:$REMOTE_PARENT/"
rsync -avz --progress -e "ssh -p $REMOTE_PORT" "$SRC" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PARENT/"

echo "Done."
