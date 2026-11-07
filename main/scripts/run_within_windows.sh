#!/usr/bin/env bash
# run_within_windows.sh - within-person experiment on fixed 5-second windows ("Now move to within person tests").
#
#     bash scripts/run_within_windows.sh
#
# For each person: pose model on the first 80 % of every recording (60 epochs, out-of-fold poses for the training part),
# then an SNN on windows of 10 frames, three seeds, joints / outline / both, always with speed.
# Takes roughly 1 to 1.5 hours on one H200.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SIWIS_DIR"

$RUN python within_pipeline.py --data "$POSE_DATA" --npz_dir "$NPZ_DIR" --out "$PERCOM/siwis_results_within_full"
