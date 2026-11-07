#!/usr/bin/env bash
# run_segments.sh - the final experiment: one complete gesture at a time, within person.
#
#     bash scripts/run_segments.sh
#
# 1. video_segments.py    cut every recording into single gestures, using the video (hand height curve)
# 2. check the plots      $POSE_DATA/segments_ai_plots/<recording>.png   <- please look at some of them
# 3. segment_pipeline.py  pose model (out-of-fold), SNN with 3 seeds, plus the control run with video poses
#
# The control run (--control) feeds the SNN with poses taken from the video. It is not a result, it only shows
# that the SNN itself can solve the task (99-100 % in the report).
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SIWIS_DIR"

$RUN python video_segments.py --src "$SSD" --data "$POSE_DATA"

echo
echo "Segments written to $POSE_DATA/segments_ai.json. Look at the plots in $POSE_DATA/segments_ai_plots first."
read -r -p "Use these cuts for the experiment? [y/N] " yn
[ "$yn" = "y" ] || [ "$yn" = "Y" ] || { echo "Stopped. Adjust the settings of video_segments.py and run again."; exit 0; }
cp "$POSE_DATA/segments_ai.json" "$POSE_DATA/segments.json"

$RUN python segment_pipeline.py --data "$POSE_DATA" --npz_dir "$NPZ_DIR" --out "$PERCOM/siwis_results_seg" --control
