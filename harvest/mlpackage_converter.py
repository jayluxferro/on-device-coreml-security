"""
mlmodelc → mlpackage converter for Apple's native compiled bundles.

Apple's native .mlmodelc bundles (as found on iOS/macOS) lack the
Manifest.json that coremltools requires.  This tool converts them to
.mlpackage format, which coremltools can load directly.

Supports three bundle layouts:
  1. MIL bundles:      model.mil + weights/weight.bin       (→ mlpackage with mil)
  2. Tree ensembles:   coremldata.bin + _B0000.DAT         (→ mlpackage with protobuf)
  3. Legacy protobuf:  coremldata.bin only                 (→ mlpackage with protobuf)

Usage:
  uv run python harvest/mlpackage_converter.py <input.mlmodelc> [output.mlpackage]
  uv run python harvest/mlpackage_converter.py --bulk <directory> <output_dir>

Author note: kept as a permanent research tool — the conversion logic is
needed for future firmware/device harvests.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid

PACKAGE_AUTHOR = "com.apple.CoreML"


def _build_manifest(filename: str, path: str) -> dict:
    """Build a valid mlpackage Manifest.json."""
    root_id = str(uuid.uuid4()).upper()
    return {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            root_id: {
                "author": PACKAGE_AUTHOR,
                "description": "CoreML Model Specification",
                "name": filename,
                "path": path,
            }
        },
        "rootModelIdentifier": root_id,
    }


def convert_mlmodelc_to_mlpackage(input_path: str, output_path: str | None = None) -> str:
    """Convert a native .mlmodelc bundle to a loadable .mlpackage.

    Returns the output package path.
    """
    if not os.path.isdir(input_path):
        raise ValueError(f"Not a directory: {input_path}")
    if output_path is None:
        output_path = input_path.rstrip("/") + ".mlpackage"

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    data_dir = os.path.join(output_path, "Data", PACKAGE_AUTHOR)
    os.makedirs(os.path.join(data_dir, "weights"), exist_ok=True)

    has_mil = os.path.exists(os.path.join(input_path, "model.mil"))
    has_weights = os.path.exists(os.path.join(input_path, "weights", "weight.bin"))

    if has_mil:
        # MIL bundle: model.mil + weights
        shutil.copy(os.path.join(input_path, "model.mil"), os.path.join(data_dir, "model.mil"))
        if has_weights:
            shutil.copy(
                os.path.join(input_path, "weights", "weight.bin"),
                os.path.join(data_dir, "weights", "weight.bin"),
            )
        # Check for additional weight files
        weights_dir = os.path.join(input_path, "weights")
        for f in os.listdir(weights_dir):
            src = os.path.join(weights_dir, f)
            dst = os.path.join(data_dir, "weights", f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy(src, dst)
        manifest = _build_manifest("model.mil", f"{PACKAGE_AUTHOR}/model.mil")
    else:
        # Protobuf bundle: coremldata.bin holds the spec
        spec_src = os.path.join(input_path, "coremldata.bin")
        if not os.path.exists(spec_src):
            raise ValueError(f"No model.mil or coremldata.bin in {input_path}")
        shutil.copy(spec_src, os.path.join(data_dir, "model.mlmodel"))
        manifest = _build_manifest("model.mlmodel", f"{PACKAGE_AUTHOR}/model.mlmodel")

    with open(os.path.join(output_path, "Manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return output_path


def bulk_convert(input_dir: str, output_dir: str) -> list[dict]:
    """Convert every .mlmodelc under input_dir into output_dir.

    Returns [{name, output, status, error}] records.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for root, _dirs, files in os.walk(input_dir):
        if not root.endswith(".mlmodelc"):
            continue
        name = os.path.basename(root)
        out_path = os.path.join(output_dir, name + ".mlpackage")
        try:
            convert_mlmodelc_to_mlpackage(root, out_path)
            results.append({"name": name, "output": out_path, "status": "ok", "error": ""})
        except Exception as exc:
            results.append({"name": name, "output": "", "status": "error", "error": str(exc)})
    return results


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage:")
        print("  uv run python harvest/mlpackage_converter.py <input.mlmodelc> [output.mlpackage]")
        print("  uv run python harvest/mlpackage_converter.py --bulk <directory> <output_dir>")
        sys.exit(1)

    if sys.argv[1] == "--bulk":
        results = bulk_convert(sys.argv[2], sys.argv[3])
        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"Converted {ok}/{len(results)} bundles")
        for r in results:
            if r["status"] != "ok":
                print(f"  ❌ {r['name']}: {r['error'][:80]}")
    else:
        out = convert_mlmodelc_to_mlpackage(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        print(f"Converted: {out}")


if __name__ == "__main__":
    main()
