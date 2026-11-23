#!/usr/bin/env python3
"""Crash-or-not harness for CoreML bundle mutations (MIL + weight extents).

Motivation: the one reproducible Apple-side fault we found (ORCHARD-2026-001)
was a missing file-extent bound check in BNNS AOT lowering of a MIL weight
walk: a model whose weight blob is shorter than the declared tensor needs
walks past the mapping and faults. That was found by hand. This harness
searches the same class at scale over Apple-shipped bundles.

Each mutation is applied to a throwaway copy of a bundle and loaded through
the *native* macOS CoreML runtime in a subprocess, so a SIGSEGV is isolated:

    uv run python harvest/mil_fuzz.py --bundle <path.mlmodelc> --rounds 20
    uv run python harvest/mil_fuzz.py --bundle <path> --control   # known-crash model

Output: datasets/mil_fuzz_<os>.jsonl, one record per attempt:
  {bundle, mutation, params, rc, signal, stderr_tail}

Exit codes: 0 = loaded, 139/-11 = SIGSEGV, 134/-6 = SIGABRT, 124 = timeout.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADER_SRC = os.path.join(ROOT, "poc", "load_model.swift")
DATASETS = os.path.join(ROOT, "datasets")


def build_loader() -> str:
    """Compile the native CoreML loader once; return the binary path."""
    out = os.path.join(tempfile.gettempdir(), "coreml_loader")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(LOADER_SRC):
        return out
    subprocess.run(["swiftc", "-O", LOADER_SRC, "-o", out], check=True,
                   capture_output=True)
    return out


def run_loader(loader: str, bundle: str, timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run([loader, bundle], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-400:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"


# ---------------------------------------------------------------- mutations

def _weight_files(bundle: str) -> list[str]:
    hits = []
    for root, _dirs, files in os.walk(bundle):
        for f in files:
            if f in ("weight.bin", "weights.bin") or f.endswith(".weights"):
                hits.append(os.path.join(root, f))
    return hits


def _mil_files(bundle: str) -> list[str]:
    return [os.path.join(r, f) for r, _d, fs in os.walk(bundle)
            for f in fs if f.endswith(".mil")]


def mut_truncate_weights(bundle: str, frac: float) -> dict:
    """Shrink a weight blob to `frac` of its size — the ORCHARD-2026-001 class."""
    files = _weight_files(bundle)
    if not files:
        return {"applied": False, "why": "no weight file"}
    target = max(files, key=os.path.getsize)
    size = os.path.getsize(target)
    with open(target, "r+b") as f:
        f.truncate(max(1024, int(size * frac)))
    return {"applied": True, "mutation": "truncate_weights", "frac": frac,
            "file": os.path.basename(target), "orig_bytes": size}


def mut_shape_scale(bundle: str, factor: int) -> dict:
    """Multiply one tensor dimension in model.mil (declared > mapped extent)."""
    files = _mil_files(bundle)
    if not files:
        return {"applied": False, "why": "no .mil"}
    path = files[0]
    text = open(path, encoding="utf-8", errors="replace").read()
    dims = list(re.finditer(r"\[\s*(\d{2,4})(?:\s*,\s*(\d{1,4}))?\s*\]", text))
    if not dims:
        return {"applied": False, "why": "no shape literal"}
    m = random.Random(factor).choice(dims)
    new = f"[{int(m.group(1)) * factor}]"
    open(path, "w", encoding="utf-8").write(text[:m.start()] + new + text[m.end():])
    return {"applied": True, "mutation": "shape_scale", "factor": factor,
            "from": m.group(0), "to": new}


def mut_zero_weights(bundle: str, n: int) -> dict:
    files = _weight_files(bundle)
    if not files:
        return {"applied": False, "why": "no weight file"}
    target = max(files, key=os.path.getsize)
    with open(target, "r+b") as f:
        f.seek(0)
        f.write(b"\x00" * n)
    return {"applied": True, "mutation": "zero_weights", "bytes": n,
            "file": os.path.basename(target)}


MUTATIONS = [
    ("truncate_0.5", lambda b: mut_truncate_weights(b, 0.5)),
    ("truncate_0.1", lambda b: mut_truncate_weights(b, 0.1)),
    ("truncate_0.01", lambda b: mut_truncate_weights(b, 0.01)),
    ("shape_x2", lambda b: mut_shape_scale(b, 2)),
    ("shape_x10", lambda b: mut_shape_scale(b, 10)),
    ("zero_4k", lambda b: mut_zero_weights(b, 4096)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--rounds", type=int, default=1, help="repeats per mutation")
    ap.add_argument("--only", help="comma-separated mutation names")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bundle = os.path.abspath(args.bundle)
    if not os.path.isdir(bundle):
        sys.exit(f"not a bundle: {bundle}")

    loader = build_loader()
    rc, tail = run_loader(loader, bundle)
    baseline = {"bundle": bundle, "mutation": "baseline", "params": {},
                "rc": rc, "signal": "SEGV" if rc in (-11, 139) else
                ("ABRT" if rc in (-6, 134) else None), "stderr_tail": tail}
    print(f"baseline rc={rc} {'(crashes unmodified!)' if rc not in (0,) else 'loads'}")
    records = [baseline]

    wanted = set(args.only.split(",")) if args.only else None
    for name, fn in MUTATIONS:
        if wanted and name not in wanted:
            continue
        for r in range(args.rounds):
            tmp = tempfile.mkdtemp(prefix="milfuzz_")
            dst = os.path.join(tmp, "m.mlmodelc")
            shutil.copytree(bundle, dst)
            try:
                info = fn(dst)
                if not info.get("applied"):
                    print(f"  {name}: skipped ({info.get('why')})")
                    continue
                rc, tail = run_loader(loader, dst)
                rec = {"bundle": bundle, "mutation": name, "params": info,
                       "rc": rc,
                       "signal": "SEGV" if rc in (-11, 139) else
                                 ("ABRT" if rc in (-6, 134) else None),
                       "stderr_tail": tail}
                records.append(rec)
                flag = "  <-- CRASH" if rec["signal"] else ""
                print(f"  {name}[{r}] rc={rc} {rec['signal'] or ''}{flag}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    out = args.out or os.path.join(
        DATASETS, f"mil_fuzz_{platform.mac_ver()[0]}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    crashes = [r for r in records if r["signal"]]
    print(f"\n{len(records)} attempts, {len(crashes)} crash(es); appended to {out}")


if __name__ == "__main__":
    main()
