#!/usr/bin/env bash
# setup_siwis.sh - get the instructor's SiWiS repository and put our scripts next to its files.
#
#     bash scripts/setup_siwis.sh
#
# Our pose scripts import the repository's config.py, models/, losses/ and data/dataset.py. We did not change
# any of those files. Instead of keeping a fork, this script
#   1. clones https://github.com/KateIronic/siwis-pose-estimation into external/ (it is not part of this repository),
#   2. copies our siwis_pose/*.py and csi_tools/*.py into the clone, so everything can be run from there.
# (Copies, not links: Python puts the real location of a script on its import path, and a linked script
# would then no longer find the repository's config.py.)
# Re-running the script is safe: an existing clone is kept and the copies are refreshed. Run it again
# whenever you change one of our files.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ ! -d "$SIWIS_DIR/.git" ]; then
    mkdir -p "$(dirname "$SIWIS_DIR")"
    git clone https://github.com/KateIronic/siwis-pose-estimation "$SIWIS_DIR"
else
    echo "SiWiS repository already cloned in $SIWIS_DIR"
fi

for f in "$REPO"/siwis_pose/*.py "$REPO"/csi_tools/*.py; do
    rm -f "$SIWIS_DIR/$(basename "$f")"      # also removes an old link, if there is one
    cp "$f" "$SIWIS_DIR/$(basename "$f")"
done
echo "Copied $(ls "$REPO"/siwis_pose/*.py "$REPO"/csi_tools/*.py | wc -l) files into $SIWIS_DIR"
echo "Now run the pose scripts from there:  cd $SIWIS_DIR && python prepare_percom.py --help"
