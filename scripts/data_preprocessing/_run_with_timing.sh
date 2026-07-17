#!/usr/bin/env bash
# Shared progress/timing wrapper for camera-LMDB launchers.
# Call run_with_timing <command> [args...].  The wrapped builders update
# __count__ in their LMDB shards, which lets us estimate ETA while running.

_timing_human_seconds() {
    local seconds=${1:-0}
    printf '%02dh:%02dm:%02ds' $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
}

_timing_done_count() {
    local output_dir=${1:-}
    [[ -n "${output_dir}" && -d "${output_dir}" ]] || { echo 0; return; }
    python - "${output_dir}" <<'PY' 2>/dev/null || echo 0
import glob
import sys

try:
    import lmdb
except Exception:
    print(0)
    raise SystemExit

import os
root = sys.argv[1]
final = f"{root}/data"
rank_paths = sorted(glob.glob(f"{root}/.rank_*"))

def read_count(path):
    try:
        env = lmdb.open(path, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            value = txn.get(b"__count__")
        env.close()
        return int(value.decode()) if value is not None else 0
    except Exception:
        return 0

final_count = read_count(final) if os.path.isdir(final) else 0
rank_count = 0
newest_rank_mtime = 0.0
for path in rank_paths:
    rank_count += read_count(path)
    try:
        newest_rank_mtime = max(newest_rank_mtime, os.path.getmtime(path))
    except Exception:
        pass

# Before pre-merge, final_count is the old merged total and rank_count is new
# work. During pre-merge, final_count has just been updated while rank shards
# still exist; use the newer final LMDB to avoid counting those records twice.
final_mtime = os.path.getmtime(final) if os.path.isdir(final) else 0.0
print(final_count if final_mtime >= newest_rank_mtime else final_count + rank_count)
PY
}

run_with_timing() {
    local output_dir=${TIMER_OUTPUT_DIR:-${OUTPUT_DIR:-}}
    local total_items=${TIMER_TOTAL_ITEMS:-0}
    local interval=${TIMER_INTERVAL:-60}
    local start_epoch now_epoch elapsed done eta monitor_pid child_pid exit_code

    start_epoch=$(date +%s)
    echo "[timing] started: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "[timing] output=${output_dir:-<unknown>} total=${total_items:-unknown}"

    "$@" &
    child_pid=$!
    (
        while kill -0 "${child_pid}" 2>/dev/null; do
            sleep "${interval}"
            kill -0 "${child_pid}" 2>/dev/null || break
            now_epoch=$(date +%s)
            elapsed=$((now_epoch - start_epoch))
            done=$(_timing_done_count "${output_dir}")
            if [[ "${total_items}" =~ ^[0-9]+$ ]] && (( total_items > 0 && done >= total_items )); then
                printf '[timing] now=%s elapsed=%s progress=%s/%s ETA=%s\n' \
                    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$(_timing_human_seconds "${elapsed}")" \
                    "${done}" "${total_items}" "$(_timing_human_seconds 0)"
            elif [[ "${total_items}" =~ ^[0-9]+$ ]] && (( total_items > 0 && done > 0 )); then
                eta=$((elapsed * (total_items - done) / done))
                printf '[timing] now=%s elapsed=%s progress=%s/%s ETA=%s\n' \
                    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$(_timing_human_seconds "${elapsed}")" \
                    "${done}" "${total_items}" "$(_timing_human_seconds "${eta}")"
            else
                printf '[timing] now=%s elapsed=%s progress=%s/%s ETA=calculating\n' \
                    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$(_timing_human_seconds "${elapsed}")" \
                    "${done}" "${total_items:-unknown}"
            fi
        done
    ) &
    monitor_pid=$!

    if wait "${child_pid}"; then
        exit_code=0
    else
        exit_code=$?
    fi
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true

    now_epoch=$(date +%s)
    elapsed=$((now_epoch - start_epoch))
    done=$(_timing_done_count "${output_dir}")
    printf '[timing] finished: %s elapsed=%s progress=%s/%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$(_timing_human_seconds "${elapsed}")" \
        "${done}" "${total_items:-unknown}"
    return "${exit_code}"
}
