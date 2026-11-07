#!/usr/bin/env bash
# run_doppler.sh - the Doppler-heatmap experiment (report, "Reading the gesture directly from CSI").
#
#     bash scripts/run_doppler.sh
#
# 1. make_dataset.py   .csi -> .npz -> Doppler heatmap -> labelled 1 s frames (needs the PicoScenes environment,
#                      so run this part on the laptop; skipped for recordings that are already converted)
# 2. train_snn.py      SNN trained on 2 people, validated on 1, tested on the 4th, repeated for every person.
#                      This is the "train on 3 people and test on the 4th" run: about 35 % per frame, 44 % per recording.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"

"$PICO_PY" doppler_snn/make_dataset.py --src "$SSD" --out "$DOPPLER_DATA" --work "$PERCOM/work"

$RUN python3 doppler_snn/train_snn.py --mode loso --dataset "$DOPPLER_DATA" --pre "$PRE_DIR" --out "$PERCOM/snn_results"
