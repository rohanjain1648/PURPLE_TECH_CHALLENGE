#!/usr/bin/env bash
# One command to process all clips for a store and feed events into the API.
# Usage: bash pipeline/run.sh <clips_dir> <store_id> [api_url]

set -euo pipefail

CLIPS_DIR="${1:?Usage: run.sh <clips_dir> <store_id> [api_url]}"
STORE_ID="${2:?Usage: run.sh <clips_dir> <store_id> [api_url]}"
API_URL="${3:-http://localhost:8000}"
LAYOUT="data/store_layout.json"

echo "=== Store Intelligence Detection Pipeline ==="
echo "Store:  $STORE_ID"
echo "Clips:  $CLIPS_DIR"
echo "API:    $API_URL"
echo ""

# Wait for API to be ready
echo "Waiting for API..."
until curl -sf "$API_URL/health" > /dev/null; do sleep 2; done
echo "API ready."

python -m pipeline.detect \
  --clips "$CLIPS_DIR" \
  --layout "$LAYOUT" \
  --store-id "$STORE_ID" \
  --api-url "$API_URL" \
  --output "events_${STORE_ID}.jsonl" \
  --model yolov8s.pt \
  --conf 0.40 \
  --frame-skip 3

echo ""
echo "Detection complete. Events written to events_${STORE_ID}.jsonl"
echo "Metrics: $API_URL/stores/$STORE_ID/metrics"
