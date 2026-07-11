#!/usr/bin/env bash
set -euo pipefail
/app/check_data_disk.sh
exec python3 /app/worker.py
