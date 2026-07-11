#!/usr/bin/env bash
# Fail fast if the mounted /data volume is below the minimum free threshold,
# before a worker attempts a (potentially multi-GB) weight download. Protects
# the shared, near-full lab volume from being exhausted.
set -euo pipefail

MIN_GB="${DATA_DISK_MIN_GB:-5}"
TARGET="${1:-/data/hf_cache}"

avail_kb="$(df -Pk "$TARGET" | awk 'NR==2 {print $4}')"
avail_gb=$(( avail_kb / 1024 / 1024 ))

if (( avail_gb < MIN_GB )); then
    echo "FATAL: only ${avail_gb}GB free on ${TARGET} (min ${MIN_GB}GB required)." >&2
    echo "Refusing to start to avoid exhausting the shared volume." >&2
    exit 1
fi
echo "Disk check OK: ${avail_gb}GB free on ${TARGET} (min ${MIN_GB}GB)."
