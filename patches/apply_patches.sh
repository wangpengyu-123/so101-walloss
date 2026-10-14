#!/usr/bin/env bash
# Apply the Plan A (round 2) patches to a wall-x source tree.
#
# Default target: the patched round-1 source actually used on the cloud
# (identical to wallx_patched.tar.gz). Override with WALLX_SRC.
#
#   bash patches/apply_patches.sh [WALLX_SRC]
#
# The patches are unified diffs (git-style, -p1) generated against
# /home/ubuntu22/so101_walloss/wall-x. They only touch 4 files:
#   1. wall_x/config/model_config.py          (new config fields)
#   2. wall_x/model/core/vla_mixin.py         (add_lora wraps action head)
#   3. wall_x/trainer/fsdp_trainer/fsdp_trainer.py (pass new args)
#   4. wall_x/trainer/fsdp_trainer/checkpoint_io.py (save all trainable)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-/home/ubuntu22/so101_walloss/wall-x}"

if [ ! -d "$TARGET/wall_x" ]; then
    echo "ERROR: $TARGET does not look like a wall-x source tree (no wall_x/)" >&2
    exit 1
fi

for p in "$HERE"/*.patch; do
    echo "==> applying $(basename "$p")"
    patch -p1 -d "$TARGET" < "$p"
done

echo "done. verify with: git -C $TARGET diff --stat (if a git repo)"
