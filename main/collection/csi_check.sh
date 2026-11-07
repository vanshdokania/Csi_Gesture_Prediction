#!/usr/bin/env bash
# csi_check.sh - debugging helper for the days when PicoScenes does not write a .csi file.
# It checks the usual suspects and, if the channel is the problem, re-tunes the card.
#
# Usage:   cd ~/percom && ./csi_check.sh [router_ip]        (default router_ip = 192.168.0.1)
#
# What it does (about 40 seconds, asks for your sudo password once):
#   1. shows whether the AX200 is connected to the router's Wi-Fi, and on which channel / width
#   2. checks that the router answers pings over that Wi-Fi link
#   3. runs a short PicoScenes test with traffic and checks whether a .csi file appears
#   4. if not: tunes PicoScenes to the router's channel (array_prepare_for_picoscenes) and tests again
#
# The most common cause in our setup was a channel mismatch: the router sat on a 5 GHz channel
# but PicoScenes was listening on a different one, so it never saw a packet.

set -u
set -m     # background jobs get their own process group, so signals reach the right process

ROUTER_IP="${1:-192.168.0.1}"
NIC_ID="${NIC_ID:-2}"                    # PicoScenes PhyPath of the AX200 (array_status)
TEST_SECONDS="${TEST_SECONDS:-15}"
PING_INTERVAL="${PING_INTERVAL:-0.01}"

say()  { printf '%s\n' "$*"; }
head_() { printf '\n== %s ==\n' "$*"; }

# --------------------------------------------------------------------------- helpers
stop_group() {   # <pid> <seconds>: SIGINT, then TERM, then KILL on the job's process group
    local pid=$1 secs=$2 i
    [ -n "$pid" ] || return 0
    [ -d "/proc/$pid" ] || return 0
    kill -INT -- "-$pid" 2>/dev/null
    for ((i = 0; i < secs * 10; i++)); do [ -d "/proc/$pid" ] || { wait "$pid" 2>/dev/null; return 0; }; sleep 0.1; done
    kill -TERM -- "-$pid" 2>/dev/null; sleep 2
    [ -d "/proc/$pid" ] && kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
}

ps_channel_string() {   # <control MHz> <width MHz> <center MHz>  ->  PicoScenes channel string
    case "$2" in
        20)     echo "$1 HT20" ;;
        40)     if [ "$3" -gt "$1" ]; then echo "$1 HT40+"; else echo "$1 HT40-"; fi ;;
        80|160) echo "$1 $2 $3" ;;
        *)      echo "" ;;
    esac
}

PICO_PID=""; PING_PID=""
cleanup() { stop_group "$PING_PID" 5; stop_group "$PICO_PID" 20; }
trap 'cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

# --------------------------------------------------------------------------- preflight
for c in iw ip ping PicoScenes; do
    command -v "$c" >/dev/null 2>&1 || { say "ERROR: '$c' not found (sudo apt install iw iproute2 iputils-ping)"; exit 1; }
done
if pgrep -x PicoScenes >/dev/null 2>&1; then
    say "ERROR: PicoScenes is already running (PID $(pgrep -x PicoScenes | tr '\n' ' ')). It owns the Wi-Fi card."
    say "       Stop it first:  pkill -INT PicoScenes ; sleep 3 ; pgrep -x PicoScenes || echo stopped"
    exit 1
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo -n"
    say "Pinging fast needs root - enter your sudo password once:"
    sudo -v || { say "ERROR: sudo failed"; exit 1; }
fi

# --------------------------------------------------------------------------- 1. Wi-Fi state
head_ "1. Wi-Fi connection of this laptop"
IFACES=(); declare -A TYPE SSID FREQ WIDTH C1
iface=""
while IFS= read -r line; do
    if   [[ $line =~ ^[[:space:]]*Interface[[:space:]]+(.+)$ ]]; then iface="${BASH_REMATCH[1]}"; IFACES+=("$iface")
    elif [[ $line =~ ^[[:space:]]*type[[:space:]]+(.+)$ ]];      then TYPE[$iface]="${BASH_REMATCH[1]}"
    elif [[ $line =~ ^[[:space:]]*ssid[[:space:]]+(.+)$ ]];      then SSID[$iface]="${BASH_REMATCH[1]}"
    elif [[ $line =~ channel[[:space:]]+[0-9]+[[:space:]]+\(([0-9]+)[[:space:]]+MHz\),[[:space:]]+width:[[:space:]]+([0-9]+)[[:space:]]+MHz(,[[:space:]]+center1:[[:space:]]+([0-9]+)[[:space:]]+MHz)? ]]; then
        FREQ[$iface]="${BASH_REMATCH[1]}"; WIDTH[$iface]="${BASH_REMATCH[2]}"
        C1[$iface]="${BASH_REMATCH[4]:-${BASH_REMATCH[1]}}"
    fi
done < <(iw dev 2>/dev/null)

WIF=""
for i in "${IFACES[@]}"; do
    say "  interface $i: type=${TYPE[$i]:-?}  ssid=${SSID[$i]:--}  channel=${FREQ[$i]:--} MHz  width=${WIDTH[$i]:--} MHz  center=${C1[$i]:--} MHz"
    [ -z "$WIF" ] && [ -n "${SSID[$i]:-}" ] && [ -n "${FREQ[$i]:-}" ] && WIF="$i"
done
if [ -z "$WIF" ]; then
    say
    say "PROBLEM: the laptop is NOT connected to any Wi-Fi network (no interface has an SSID + channel)."
    say "         The ping cannot reach the router, so PicoScenes has nothing to measure."
    say "To fix : connect to the TP-Link Wi-Fi (the 5 GHz network is preferred) and run this check again:"
    say "           nmcli dev wifi list"
    say "           nmcli dev wifi connect \"<SSID>\" password \"<password>\""
    exit 1
fi
WIFI_STR="$(ps_channel_string "${FREQ[$WIF]}" "${WIDTH[$WIF]}" "${C1[$WIF]}")"
say
say "  -> connected to '${SSID[$WIF]}' on ${FREQ[$WIF]} MHz, ${WIDTH[$WIF]} MHz wide (centre ${C1[$WIF]} MHz)"
say "  -> PicoScenes channel string for this network:  \"$WIFI_STR\""

# --------------------------------------------------------------------------- 2. router reachable over Wi-Fi?
head_ "2. Can the router be reached over this Wi-Fi link?"
GW="$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')"
DEV="$(ip route get "$ROUTER_IP" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -n1)"
say "  default gateway : ${GW:-none}"
say "  route to $ROUTER_IP goes out via: ${DEV:-unknown}"
if [ -n "$DEV" ] && [ "$DEV" != "$WIF" ]; then
    say "  PROBLEM: traffic to $ROUTER_IP leaves through '$DEV', not the Wi-Fi card '$WIF' -> no CSI."
    say "           Unplug the Ethernet cable / other network, then run this again."
    exit 1
fi
if [ -n "$GW" ] && [ "$GW" != "$ROUTER_IP" ]; then
    say "  NOTE: your default gateway is $GW, but you are pinging $ROUTER_IP."
    say "        If the router's address is really $GW, run:  ./csi_check.sh $GW   (and set ROUTER_IP in record.sh)"
fi
PING_OUT="$(ping -c 5 -i 0.2 -W 1 "$ROUTER_IP" 2>&1 | tail -n 3)"
say "$(printf '%s\n' "$PING_OUT" | sed 's/^/  /')"
if printf '%s' "$PING_OUT" | grep -q ' 0 received\|100% packet loss'; then
    say "  PROBLEM: the router does not answer pings. Wrong IP address, or you are on a different network."
    exit 1
fi
say "  OK: the router answers."

# --------------------------------------------------------------------------- 3/4. live PicoScenes test
run_test() {   # <label>  -> sets TEST_BYTES, TEST_CH_LINE, TEST_DIR
    local label=$1
    TEST_DIR="$(mktemp -d /tmp/csi_check.XXXXXX)"
    say "  [$label] running PicoScenes + ping for ${TEST_SECONDS}s ..."
    $SUDO ping -i "$PING_INTERVAL" -q "$ROUTER_IP" >"$TEST_DIR/ping.log" 2>&1 </dev/null &
    PING_PID=$!
    ( cd "$TEST_DIR" && exec PicoScenes -d debug -i "$NIC_ID" --mode logger ) >"$TEST_DIR/ps.log" 2>&1 </dev/null &
    PICO_PID=$!
    sleep "$TEST_SECONDS"
    stop_group "$PICO_PID" 20; PICO_PID=""
    stop_group "$PING_PID" 5;  PING_PID=""
    TEST_BYTES="$(find "$TEST_DIR" -maxdepth 1 -name '*.csi' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')"
    TEST_CH_LINE="$(grep -m1 -o 'center=[0-9.]*MHz, control=[0-9.]*MHz, RxCBW=[0-9.]*MHz' "$TEST_DIR/ps.log" 2>/dev/null)"
    say "  [$label] PicoScenes listened on: ${TEST_CH_LINE:-(not found in log)}"
    say "  [$label] .csi data written     : $TEST_BYTES bytes   (log: $TEST_DIR/ps.log)"
}

head_ "3. PicoScenes test with the card as it is now"
run_test "test 1"
if [ "$TEST_BYTES" -gt 0 ]; then
    say
    say "RESULT: CSI is being produced. record.sh should work as it is."
    say "        (~$((TEST_BYTES / TEST_SECONDS / 1024)) KB/s of CSI at ping interval $PING_INTERVAL s)"
    exit 0
fi

# compare channel PicoScenes used with the connected network's channel
PS_CENTER="$(printf '%s' "$TEST_CH_LINE" | sed -n 's/.*center=\([0-9]*\).*/\1/p')"
PS_CONTROL="$(printf '%s' "$TEST_CH_LINE" | sed -n 's/.*control=\([0-9]*\).*/\1/p')"
MISMATCH=0
if [ -n "$PS_CONTROL" ] && { [ "$PS_CONTROL" != "${FREQ[$WIF]}" ] || [ "$PS_CENTER" != "${C1[$WIF]}" ]; }; then MISMATCH=1; fi

if [ "$MISMATCH" -eq 1 ]; then
    say
    say "Found it: PicoScenes listens on control $PS_CONTROL / centre $PS_CENTER MHz,"
    say "          but your router is on control ${FREQ[$WIF]} / centre ${C1[$WIF]} MHz.  -> wrong channel, so no packets."
else
    say
    say "NOTE: PicoScenes already listens on the router's channel, yet no CSI arrived. Trying a clean re-tune anyway."
fi

head_ "4. Tuning PicoScenes to the router's channel and testing again"
command -v array_prepare_for_picoscenes >/dev/null 2>&1 || { say "ERROR: array_prepare_for_picoscenes not found"; exit 1; }
say "  running: array_prepare_for_picoscenes $NIC_ID \"$WIFI_STR\""
array_prepare_for_picoscenes "$NIC_ID" "$WIFI_STR" || { say "ERROR: array_prepare_for_picoscenes failed"; exit 1; }
sleep 2
# did the Wi-Fi connection survive?
if ! iw dev 2>/dev/null | grep -q 'ssid '; then
    say "  WARNING: the Wi-Fi connection dropped after re-tuning (no SSID any more)."
    say "           A ping cannot work then; reconnect with:  nmcli dev wifi connect \"${SSID[$WIF]}\""
fi
run_test "test 2"
if [ "$TEST_BYTES" -gt 0 ]; then
    say
    say "RESULT: fixed. The card is now tuned to the router and CSI is produced."
    say "        You can run ./record.sh now. If you later change the router/channel, run this check again."
    exit 0
fi

say
say "RESULT: still no CSI. Keep these for debugging (or send them to whoever helps you):"
say "   - this whole output"
say "   - $TEST_DIR/ps.log"
say "   - the output of:  array_status    and    sudo dmesg | grep -i -E 'iwlwifi|picoscenes' | tail -n 20"
exit 1
