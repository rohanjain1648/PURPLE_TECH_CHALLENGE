#!/usr/bin/env bash
# Process the actual Purplle CCTV clips using clips_config.json.
# Usage: bash pipeline/run.sh [clips_dir] [api_url] [output_jsonl]

set -euo pipefail

CLIPS_DIR="${1:-CCTV Footage}"
API_URL="${2:-http://localhost:8000}"
OUTPUT="${3:-events_STORE_PRP_001.jsonl}"
LAYOUT="data/store_layout.json"
CLIPS_CONFIG="data/clips_config.json"

echo "=== Purplle Store Intelligence — Detection Pipeline ==="
echo "Clips : $CLIPS_DIR"
echo "Config: $CLIPS_CONFIG"
echo "API   : $API_URL"
echo "Output: $OUTPUT"
echo ""

# Wait for API to be ready
echo "Waiting for API..."
until curl -sf "$API_URL/health" > /dev/null 2>&1; do
    sleep 2
done
echo "API ready."
echo ""

# Run YOLOv8s detection over all 5 clips via clips_config.json
python -m pipeline.detect \
    --clips-config "$CLIPS_CONFIG" \
    --clips-dir "$CLIPS_DIR" \
    --layout "$LAYOUT" \
    --api-url "$API_URL" \
    --output "$OUTPUT" \
    --model yolov8s.pt \
    --conf 0.35

echo ""
echo "Detection complete."
echo "Events JSONL: $OUTPUT"
echo "Live metrics: $API_URL/stores/STORE_PRP_001/metrics"
echo "Funnel      : $API_URL/stores/STORE_PRP_001/funnel"
echo "Heatmap     : $API_URL/stores/STORE_PRP_001/heatmap"
echo "Anomalies   : $API_URL/stores/STORE_PRP_001/anomalies"
