#!/usr/bin/env bash
# add_person_laptop.sh - run this on the laptop to add one more person to the dataset.
#
#   PERSON=Nikita bash scripts/add_person_laptop.sh
#
# We used it when Nikita joined late: her recordings were still sitting in ~/percom as the plain
# folders that record.sh creates, named like  nikita_bhu_1, nikita_bhu_2, nikita_lhu_1, ...
# The script
#   1. checks that every folder has video.mp4, timing.txt and a .csi file,
#   2. copies each folder to the layout the pipeline expects,  <SSD>/<ACT>/<ACT>_S<n>_<Person>/
#      (the originals are not touched, delete them yourself when you are happy),
#   3. converts the CSI to .npz (slow, roughly a minute per 1 GB file),
#   4. optionally sends the small files (video, timing, npz, but not the huge .csi files) to the GPU server.
#
# Settings you can override from the command line:
#   PERSON      name of the person (required), e.g. PERSON=Nikita
#   SESSIONS    how many sessions per gesture were recorded (default 3)
#   SSD         where the dataset lives (default /media/$USER/My Passport/percom)
#   PICO_PY     python of the PicoScenes toolbox virtual environment, needed for the conversion
#   SERVER      user@host of the GPU server, only needed for step 4
set -u

PERSON="${PERSON:-}"
[ -n "$PERSON" ] || { echo "Please say who to add, for example:  PERSON=Nikita bash $0"; exit 1; }
SESSIONS="${SESSIONS:-3}"
SSD="${SSD:-/media/$USER/My Passport/percom}"
PICO_PY="${PICO_PY:-$HOME/PicoscenesToolbox/.venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONVERTER="$HERE/../csi_tools/csi_to_npz.py"
SRC="$HOME/percom"
NPZ="$HOME/percom/work/npz"
LOWER="$(echo "$PERSON" | tr 'A-Z' 'a-z')"

[ -d "$SSD" ] || { echo "Error: $SSD not found. Is the SSD plugged in? (or run: SSD='/path/to/percom' bash $0)"; exit 1; }
[ -x "$PICO_PY" ] || { echo "Error: $PICO_PY not found. Set PICO_PY to the python of the PicoScenes toolbox environment"; exit 1; }
mkdir -p "$NPZ"

echo "== 1. checking and copying =="
ok=0; bad=0; good=""
for act in bhu lhu rhu; do
  for n in $(seq 1 "$SESSIONS"); do
    # the folder may be called nikita_bhu_1 or Nikita_bhu_1, so the search ignores the case
    f=$(find "$SRC" -maxdepth 1 -type d -iname "${LOWER}_${act}_${n}" | head -1)
    ACT=$(echo "$act" | tr 'a-z' 'A-Z')
    if [ -z "$f" ]; then echo "  missing  ${LOWER}_${act}_${n}"; bad=$((bad+1)); continue; fi
    miss=""
    [ -s "$f/video.mp4" ]  || miss="$miss video.mp4"
    [ -s "$f/timing.txt" ] || miss="$miss timing.txt"
    find "$f" -name '*.csi' 2>/dev/null | grep -q . || miss="$miss .csi"
    if [ -n "$miss" ]; then echo "  problem  $f is missing:$miss"; bad=$((bad+1)); continue; fi
    dest="$SSD/$ACT/${ACT}_S${n}_${PERSON}"
    mkdir -p "$dest"
    echo "  copying $f -> $dest   (can take a few minutes)"
    cp -a "$f"/. "$dest"/ || { echo "  copy failed"; bad=$((bad+1)); continue; }
    echo "$act" > "$dest/activity.txt"
    good="$good $dest"
    ok=$((ok+1))
  done
done
echo "   $ok recordings copied, $bad problems"
[ "$bad" -gt 0 ] && echo "   (fix the problems above and run the script again; copies that were already made are simply overwritten)"

echo; echo "== 2. converting the CSI to npz (slow: about a minute per 1 GB file) =="
for d in $good; do
  name=$(basename "$d")
  if ls "$NPZ"/*__"${name}"_rx_*.npz >/dev/null 2>&1; then echo "  already converted: $name"; continue; fi
  echo "  $name"
  "$PICO_PY" "$CONVERTER" "$d" --out "$NPZ" 2>&1 | tail -2
done
echo; ls -lh "$NPZ" | grep -i "$PERSON" || echo "  (no $PERSON npz found, so the conversion failed. See the messages above)"

echo; echo "== 3. send the small files to the GPU server (video, timing, npz, but not the 1 GB .csi files) =="
SERVER="${SERVER:-}"
if [ -z "$SERVER" ]; then
  echo "SERVER is not set (for example SERVER=user@gpu-server), so nothing was sent."
  echo "Copy ~/percom/${LOWER}_src and the $PERSON npz files to the server yourself, or run the script again with SERVER set."
  exit 0
fi
read -r -p "Send to $SERVER now? [y/N] " yn
if [ "$yn" = "y" ] || [ "$yn" = "Y" ]; then
  ssh "$SERVER" "mkdir -p ~/percom/${LOWER}_src ~/percom/work/npz" || exit 1
  for d in $good; do
    rel="${d#"$SSD"/}"
    ssh "$SERVER" "mkdir -p ~/percom/${LOWER}_src/$rel"
    scp "$d/video.mp4" "$d/timing.txt" "$d/activity.txt" "$SERVER":~/percom/${LOWER}_src/"$rel"/
  done
  scp "$NPZ"/*"$PERSON"*.npz "$SERVER":~/percom/work/npz/
  echo "Done. Next, on the server, run prepare_percom.py for $PERSON (see the README, stage 3)."
else
  echo "Skipped. Run the script again later, or copy the files to the server yourself."
fi
