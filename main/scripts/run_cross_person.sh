#!/usr/bin/env bash
# run_cross_person.sh - the cross-person experiment: train on three people, test on the fourth.
#
#     bash scripts/run_cross_person.sh
#
# A. earlier pose model (2 channels, repository loss)             -> $PERCOM/siwis_results
# B. final pose model (10 channels, balanced loss, AdamW)         -> $PERCOM/siwis_results_final
# C. SNN on the predicted poses, with and without speed           (report: "SNN settings", table "Model results")
# D. control: SNN on poses taken from the video                   (report: "Video poses", 67-80 %)
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SIWIS_DIR"
OLD="$PERCOM/siwis_results"
FINAL="$PERCOM/siwis_results_final"

# --- A. earlier model: signal-strength spread only, repository loss, 50 epochs
$RUN python train_percom.py --data "$POSE_DATA" --split person --loso --out "$OLD" --epochs 50

# --- B. final model: all 10 channels, mask loss / pixels + 500 x joint loss, 80 epochs
$RUN python train_percom_v2.py --data "$POSE_DATA" --npz_dir "$NPZ_DIR" --out "$FINAL" --epochs 80

# --- C. SNN on what the pose model predicted
python pose_snn.py --results "$FINAL" --data "$POSE_DATA"                              --feat mask   # outline, no speed
python pose_snn.py --results "$FINAL" --data "$POSE_DATA" --feat kp   --velocity                     # arm joints + speed
python pose_snn.py --results "$FINAL" --data "$POSE_DATA" --feat mask --velocity                     # outline + speed
python pose_snn.py --results "$FINAL" --data "$POSE_DATA" --feat both --velocity                     # outline + joints + speed

# --- D. control: the same SNN, but fed with poses extracted from the video
python pose_snn.py --results "$FINAL" --data "$POSE_DATA" --source gt --feat mask
python pose_snn.py --results "$FINAL" --data "$POSE_DATA" --source gt --feat mask --velocity
