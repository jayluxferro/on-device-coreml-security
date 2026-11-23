#!/bin/bash
# PoC: internal storage path leakage in shipped model description
# Reads the internal s3:// path from the hand_gesture model's metadata.
# Works on a stock device (file is world-readable) or from an IPSW extract.

# Try iOS path, then macOS versioned path
CANDIDATES=(
  "/System/Library/PrivateFrameworks/MediaAnalysis.framework/hand_gesture_dynamic_two_hand_5fps.mlmodelc"
  "/System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/Resources/hand_gesture_dynamic_two_hand_5fps.mlmodelc"
)
if [ -n "$1" ]; then
    MODEL="$1"
else
    MODEL=""
    for c in "${CANDIDATES[@]}"; do
        if [ -f "$c/metadata.json" ]; then
            MODEL="$c"
            break
        fi
    done
fi

if [ -z "$MODEL" ]; then
    echo "Model not found. Usage: $0 /path/to/hand_gesture_dynamic_two_hand_5fps.mlmodelc"
    exit 1
fi

echo "Reading metadata from: $MODEL"
echo "----------------------------------------"

if [ -f "$MODEL/metadata.json" ]; then
    grep -o 's3:[^"]*' "$MODEL/metadata.json"
    grep -o '"shortDescription"[^,]*' "$MODEL/metadata.json" | head -1
    echo "----------------------------------------"
    echo "The s3:// path is an internal Apple training-artifact reference."
    echo "It ships in plaintext in the model's shortDescription field."
    echo "Public S3 endpoints for this bucket return 404 (not accessible)."
else
    echo "No metadata.json found at $MODEL"
    echo "Try: $0 /path/to/the/model.mlmodelc"
fi
