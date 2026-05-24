#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    exec "${SCRIPT_DIR}/train_bidirectional.sh" rcm "$@"
else
    exec "${SCRIPT_DIR}/train_bidirectional.sh" rcm configs/stage1_bidirectional_rcm.yaml "$@"
fi
