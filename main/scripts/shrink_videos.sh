#!/usr/bin/env bash
# shrink_videos.sh - run on the laptop. The original videos are 1080p and far too big to send around, and
# the pose labelling only needs 480p. This script shrinks every recording video to 480 px height at 10 fps
# and copies it, together with its timing.txt, to the GPU server.
#
#   SERVER=user@gpu-server bash scripts/shrink_videos.sh
#
# Settings you can override: SSD (where the recordings are), SERVER (user@host), OUT (local folder for the small videos).
# Divij and Nikita are skipped because they are not part of the final experiments.
SSD="${SSD:-/media/$USER/My Passport/percom}"
SERVER="${SERVER:-}"
OUT="${OUT:-$HOME/percom/video_small}"
[ -d "$SSD" ] || { echo "Error: $SSD not found. Is the SSD mounted? (or run: SSD=/path/to/percom bash $0)"; exit 1; }
mkdir -p "$OUT"
n=0
for d in "$SSD"/*/*/; do
  d="${d%/}"; name=$(basename "$d"); act=$(basename "$(dirname "$d")")
  case "$name" in *ivij*|*ikita*) continue;; esac
  [ -s "$d/video.mp4" ] || { echo "skip $name (no video.mp4)"; continue; }
  mkdir -p "$OUT/$act/$name"
  cp "$d/timing.txt" "$OUT/$act/$name/" 2>/dev/null || echo "warning: $name has no timing.txt"
  if [ ! -s "$OUT/$act/$name/video.mp4" ]; then
    echo "shrinking $name"
    ffmpeg -v error -nostdin -y -i "$d/video.mp4" -vf "fps=10,scale=-2:480" -an -c:v libx264 -crf 26 -preset veryfast "$OUT/$act/$name/video.mp4" \
      || { echo "failed: $name"; rm -f "$OUT/$act/$name/video.mp4"; continue; }
  fi
  n=$((n+1))
done
echo "$n videos ready in $OUT ($(du -sh "$OUT" | cut -f1))"
[ -n "$SERVER" ] || { echo "SERVER is not set, so nothing was sent. Run again with SERVER=user@host to upload."; exit 0; }
echo "sending to the server ..."
ssh "$SERVER" 'mkdir -p ~/percom/video_small' || exit 1
scp -r "$OUT"/* "$SERVER":~/percom/video_small/ || exit 1
echo "Done. On the server the videos are in ~/percom/video_small"
