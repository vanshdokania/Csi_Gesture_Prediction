#!/usr/bin/env bash
# record.sh - record one trial: Wi-Fi CSI (PicoScenes) + video from the EMEET SmartCam S600.
#
# Every trial gets its own folder (~/percom/experiment_<date>_<time>).
#
# Usage:   cd ~/percom && ./record.sh
#   t = 0 s    PicoScenes starts logging CSI
#   t = 10 s   the webcam starts recording -> perform the gesture now
#   Ctrl+C     ffmpeg is stopped first, then PicoScenes, and both files are kept.
#
# The exact start times of CSI and video are written to timing.txt. The later
# synchronisation step (csi_tools/csi_video_sync.py) starts from those two numbers.

set -u
set -m   # every background job gets its own process group. That way Ctrl+C reaches
         # only this script, and the script shuts the children down itself, in order.

# ----------------------------- settings --------------------------------------
BASE_DIR="$HOME/percom"
# Change this to your own camera (ls -l /dev/v4l/by-id/ shows the name).
CAM="/dev/v4l/by-id/usb-EMEET_EMEET_SmartCam_S600_A250510000904236-video-index0"
WIDTH=1920
HEIGHT=1080
FPS=30
SETUP_DELAY=10
PICO_CMD=(PicoScenes -d debug -i 2 --mode logger --plot)
# PicoScenes only logs a packet when one arrives, so without traffic the .csi file stays
# empty. We keep pinging the router for the whole recording, which is the same as running
#   sudo ping -i 0.002 192.168.0.1
# by hand. Leave ROUTER_IP empty if something else already generates the traffic.
ROUTER_IP="192.168.0.1"
PING_INTERVAL="0.002"
# ------------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }

# ----------------------------- preflight --------------------------------------
for c in ffmpeg ffprobe v4l2-ctl PicoScenes; do
    command -v "$c" >/dev/null 2>&1 || die "'$c' not found in PATH"
done
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
    || die "this ffmpeg has no libx264 encoder"

[ -e "$CAM" ] || die "EMEET camera not found at:
  $CAM
Is it plugged in?  (ls -l /dev/v4l/by-id/)"

CAM_REAL="$(readlink -f "$CAM")"
echo "Camera: $CAM -> $CAM_REAL"

# the by-id path must point to a real capture node, not the metadata node
v4l2-ctl -d "$CAM" --all 2>/dev/null | sed -n '/Device Caps/,$p' | grep -q 'Video Capture' \
    || die "$CAM_REAL is not a Video Capture node"

# the camera has to offer MJPEG at WIDTHxHEIGHT and FPS
if ! v4l2-ctl -d "$CAM" --list-formats-ext 2>/dev/null | awk -v s="${WIDTH}x${HEIGHT}" -v f="$FPS" '
        /^\t?\[[0-9]+\]:/            { fmt = $0 }
        /Size: Discrete/             { sz = $3 }
        /Interval/ && fmt ~ /MJPG/ && sz == s && index($0, "(" f ".000 fps)") { ok = 1 }
        END { exit !ok }'; then
    echo "Camera does not list MJPEG ${WIDTH}x${HEIGHT} @ ${FPS}fps. It offers:" >&2
    v4l2-ctl -d "$CAM" --list-formats-ext >&2
    die "change WIDTH/HEIGHT/FPS at the top of record.sh"
fi

# ----------------------------- experiment folder -----------------------------
EXP_DIR="$BASE_DIR/experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXP_DIR" || die "cannot create $EXP_DIR"
VIDEO="$EXP_DIR/video.mp4"
MARKER="$EXP_DIR/.start_marker"
touch "$MARKER"
echo "Experiment folder: $EXP_DIR"

PICO_PID=""
FFMPEG_PID=""
PING_PID=""
CLEANED=0

# stop_group <name> <pid> <seconds>
# Stops a background job politely: SIGINT first, then SIGTERM, then SIGKILL, always on
# the whole process group of the job.
stop_group() {
    local name=$1 pid=$2 secs=$3 i
    [ -n "$pid" ] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    echo "Stopping $name (waiting up to ${secs}s for a clean exit)..."
    kill -INT -- "-$pid" 2>/dev/null
    for ((i = 0; i < secs * 10; i++)); do
        kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null; return 0; }
        sleep 0.1
    done
    echo "  $name did not exit on SIGINT, sending SIGTERM"
    kill -TERM -- "-$pid" 2>/dev/null
    for ((i = 0; i < 50; i++)); do
        kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null; return 0; }
        sleep 0.1
    done
    echo "  $name still alive, sending SIGKILL"
    kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
}

cleanup() {
    [ "$CLEANED" -eq 1 ] && return
    CLEANED=1
    trap '' INT TERM          # ignore further Ctrl+C while finalising files
    echo
    echo "=== Stopping the trial - please wait and do not press Ctrl+C again ==="

    stop_group "ffmpeg (video)" "$FFMPEG_PID" 20     # first: finalise video.mp4
    stop_group "PicoScenes (CSI)" "$PICO_PID" 30     # then: finalise CSI
    stop_group "ping (Wi-Fi traffic)" "$PING_PID" 10 # last: stop the traffic
    sync

    # PicoScenes normally writes into its working directory (= EXP_DIR). If a .csi
    # file from this run ended up somewhere else under $HOME, move it back here.
    if ! find "$EXP_DIR" -maxdepth 1 -name '*.csi' | grep -q .; then
        find "$HOME" -maxdepth 4 -name '*.csi' -newer "$MARKER" \
             -not -path "$EXP_DIR/*" -print0 2>/dev/null |
        while IFS= read -r -d '' f; do
            echo "Moving stray CSI file into experiment folder: $f"
            mv -- "$f" "$EXP_DIR/"
        done
    fi

    echo
    echo "=== Result: $EXP_DIR ==="
    if [ -s "$VIDEO" ] && ffprobe -v error -select_streams v:0 \
            -show_entries stream=codec_name,width,height,avg_frame_rate \
            -show_entries format=duration -of default=nw=1 "$VIDEO" 2>/dev/null; then
        echo "video.mp4 : OK ($(du -h "$VIDEO" | cut -f1))"
    else
        echo "video.mp4 : PROBLEM - missing or unreadable. Check $EXP_DIR/ffmpeg.log"
    fi

    local n
    n=$(find "$EXP_DIR" -maxdepth 1 -name '*.csi' | wc -l)
    echo "CSI files : $n"
    find "$EXP_DIR" -maxdepth 1 -name '*.csi' -exec ls -lh {} \;
    [ "$n" -eq 0 ] && echo "  (none found - check $EXP_DIR/picoscenes.log)"
    [ "$n" -gt 1 ] && echo "  (PicoScenes produced several files in this one run; they all belong to this trial)"
    rm -f "$MARKER"
}

trap 'cleanup; exit 0' INT TERM
trap 'cleanup' EXIT

# ----------------------------- 0. start Wi-Fi traffic ---------------------------
if [ -n "$ROUTER_IP" ]; then
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        SUDO="sudo -n"
        echo "Pinging every $PING_INTERVAL s needs root - enter your sudo password once:"
        sudo -v || die "sudo failed"
    fi
    echo "Starting traffic: ping -i $PING_INTERVAL $ROUTER_IP"
    $SUDO ping -i "$PING_INTERVAL" -q "$ROUTER_IP" >"$EXP_DIR/ping.log" 2>&1 </dev/null &
    PING_PID=$!
    sleep 1
    kill -0 "$PING_PID" 2>/dev/null || { cat "$EXP_DIR/ping.log" >&2; die "ping did not start"; }
fi

# ----------------------------- 1. start PicoScenes ----------------------------
echo "Starting PicoScenes: ${PICO_CMD[*]}"
( cd "$EXP_DIR" && exec "${PICO_CMD[@]}" ) >"$EXP_DIR/picoscenes.log" 2>&1 </dev/null &
PICO_PID=$!
T_PICO=$(date +%s.%N)

# ----------------------------- 2. setup delay ---------------------------------
for ((s = SETUP_DELAY; s > 0; s--)); do
    kill -0 "$PICO_PID" 2>/dev/null || { tail -n 20 "$EXP_DIR/picoscenes.log"; die "PicoScenes exited during setup"; }
    printf '\rVideo starts in %2d s ... ' "$s"
    sleep 1
done
printf '\r%40s\r' ''

# ----------------------------- 3. start video ---------------------------------
echo "Starting EMEET recording: ${WIDTH}x${HEIGHT} @ ${FPS}fps MJPEG -> $VIDEO"
# -movflags frag_keyframe+empty_moov : MP4 stays playable even if ffmpeg is ever killed
ffmpeg -nostdin -hide_banner -loglevel info \
    -f v4l2 -input_format mjpeg -video_size "${WIDTH}x${HEIGHT}" -framerate "$FPS" \
    -i "$CAM" \
    -c:v libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p \
    -movflags +frag_keyframe+empty_moov+default_base_moof \
    "$VIDEO" >"$EXP_DIR/ffmpeg.log" 2>&1 </dev/null &
FFMPEG_PID=$!
T_VIDEO=$(date +%s.%N)

# make sure frames are really arriving before we tell the person to start
sleep 3
if ! kill -0 "$FFMPEG_PID" 2>/dev/null || ! [ -s "$VIDEO" ]; then
    echo "---- ffmpeg.log ----" >&2
    tail -n 25 "$EXP_DIR/ffmpeg.log" >&2
    die "ffmpeg is not recording (see log above)"
fi

{
    echo "picoscenes_start_epoch=$T_PICO"
    echo "video_start_epoch=$T_VIDEO"
    echo "camera=$CAM_REAL"
    echo "video=${WIDTH}x${HEIGHT}@${FPS}"
} > "$EXP_DIR/timing.txt"

echo
echo ">>> RECORDING (CSI + video). Perform the gesture now."
echo ">>> Press Ctrl+C once when you are finished."

# ----------------------------- 4. wait for Ctrl+C -----------------------------
while kill -0 "$FFMPEG_PID" 2>/dev/null && kill -0 "$PICO_PID" 2>/dev/null; do
    sleep 1
done

echo
kill -0 "$FFMPEG_PID" 2>/dev/null || echo "WARNING: ffmpeg stopped by itself - see $EXP_DIR/ffmpeg.log"
kill -0 "$PICO_PID"  2>/dev/null || echo "WARNING: PicoScenes stopped by itself - see $EXP_DIR/picoscenes.log"
exit 1