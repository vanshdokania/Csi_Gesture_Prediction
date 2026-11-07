#!/usr/bin/env bash
# run_pose_prepare.sh - stage 1 of the pose pipeline: video -> pose labels, CSI -> features.
#
#     bash scripts/run_pose_prepare.sh
#
# Runs prepare_percom.py on the GPU server (the labelling needs Mask R-CNN and Keypoint R-CNN, which take
# seconds per frame on a CPU). The converted CSI (.npz files from csi_to_npz.py) have to be in $NPZ_DIR already.
# Used for the report: 2 labelled frames per second, 480 px frame height.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SIWIS_DIR"

$RUN python prepare_percom.py --src "$SSD" --out "$POSE_DATA" --npz_dir "$NPZ_DIR" --label_fps 2 --max_h 480

# A quick look at the labels: how often is each wrist above its elbow, per gesture?
python "$REPO/tools/label_check.py" "$POSE_DATA"
