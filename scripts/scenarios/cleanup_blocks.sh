#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

trap - EXIT
cleanup_blocks
show_blocks
log "Cleanup complete"
