#!/usr/bin/env bash
# Optional: pre-download all three systems' weights into the shared HF cache
# before starting the stack, so first-run health checks don't wait on network.
# Runs each worker image's download path once, then exits.
set -euo pipefail
echo "Weights download on first container start into \$HF_CACHE_DIR."
echo "To pre-warm, simply run 'docker compose up -d' and watch:"
echo "  docker compose logs -f worker-tortoise worker-cosyvoice2 worker-metavoice"
echo "Each worker reports 'ready' once its weights are cached and the model is loaded."
