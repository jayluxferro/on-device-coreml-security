#!/bin/bash
# PoC: mldb_token leakage in MIL model bundle metadata
# Reads the internal build token from any shipped mlProgram bundle.
# Works on a stock device (files are world-readable) or from an IPSW.

CANDIDATES=(
  "/System/Library/DuetExpertCenter/Assets/Assets.bundle/AssetData/ATXAppPredictionMLModel.mlmodelc"
  "/System/Library/PrivateFrameworks/MCCKitCategorization.framework/BlackPearlAssets/assets_BLACKPEARL_SPARROW/BlackPearlModelPackage/BlackPearl.mlpackage"
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
    echo "No MIL bundle found. Usage: $0 /path/to/any/mlProgram.mlmodelc"
    exit 1
fi

echo "Reading metadata from: $MODEL"
echo "----------------------------------------"

if [ -f "$MODEL/metadata.json" ]; then
    grep -o '"mldb_token"[^,}]*' "$MODEL/metadata.json"
    grep -o '"coremlc-version"[^,}]*' "$MODEL/metadata.json" 2>/dev/null
    grep -o '"coremlc-component-MILHost"[^,}]*' "$MODEL/metadata.json" 2>/dev/null
    grep -o '"coremltools-version"[^,}]*' "$MODEL/metadata.json" 2>/dev/null
    echo "----------------------------------------"
    echo "The mldb_token is an internal Apple build identifier."
    echo "It ships in plaintext in every MIL (mlProgram) bundle."
else
    echo "No metadata.json found at $MODEL"
    echo "Try: $0 /path/to/any/other.mlmodelc"
fi
