#!/usr/bin/env bash
# env.sh - the folders and settings that all run_*.sh scripts share. Source it, do not run it:
#
#     source scripts/env.sh
#
# Everything can be overridden from the shell before you run a script, for example
#     SSD="/media/me/Disk/percom" RUN=gpurun bash scripts/run_cross_person.sh

# The repository (this folder's parent), found automatically.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Working folders. They are created on the machine that runs the step.
PERCOM="${PERCOM:-$HOME/percom}"
NPZ_DIR="${NPZ_DIR:-$PERCOM/work/npz}"          # converted CSI (output of csi_to_npz.py)
PRE_DIR="${PRE_DIR:-$PERCOM/work/output}"       # Doppler heatmaps (output of preprocess_csi.py)
DOPPLER_DATA="${DOPPLER_DATA:-$PERCOM/dataset}" # labelled Doppler frames (output of make_dataset.py)
POSE_DATA="${POSE_DATA:-$PERCOM/siwis_data}"    # pose labels + CSI features (output of prepare_percom.py)

# Where the raw recordings live: <SSD>/<ACT>/<ACT>_S<n>_<Person>/ with video.mp4, timing.txt and rx_*.csi.
SSD="${SSD:-/media/${USER:-$(whoami)}/My Passport/percom}"

# The instructor's SiWiS repository. scripts/setup_siwis.sh clones it here and copies our files into it.
SIWIS_DIR="${SIWIS_DIR:-$REPO/external/siwis-pose-estimation}"

# Python used for the CSI conversion. It has to be the PicoScenes toolbox environment, because only that
# one can read .csi files. (Only the laptop needs it.)
PICO_PY="${PICO_PY:-$HOME/PicoscenesToolbox/.venv/bin/python}"

# Prefix for heavy jobs. On our GPU server every job is started through "gpurun" inside a tmux session:
#     tmux new -s seg
#     RUN=gpurun bash scripts/run_segments.sh
# On a machine without such a wrapper leave it empty.
RUN="${RUN:-}"
