"""
verify_bundles.py — full-tree verification driver for mlmodelc_loader.

Runs convert → load → predict for every *.mlmodelc bundle under a tree,
isolating each bundle in its own subprocess so a single compiler crash
(SIGTRAP/STL exception inside CoreML's E5RT — seen on a few bundles) cannot
kill the run.  Writes a JSON report and prints a per-bundle status table.

Usage::

    uv run python harvest/verify_bundles.py <tree> <outdir> [--report report.json] [--timeout 180]

For each bundle the per-bundle status is one of:

    converted        loads through coremltools and predicts (or predict skipped
                     for image inputs)
    predict_failed   loads but prediction raised (compile-time IR errors such
                     as pixel_buffer inputs or shape-inference failures)
    parse_failed     model.mil could not be parsed
    convert_failed   spec assembly / save_spec failed
    load_failed      the produced .mlpackage does not load
    not_mil          no model.mil (compiled-only bundle; native runtime path)
    crashed          the verification subprocess died (signal) — compile crash

Note: the CoreML compiler prints ``E5RT encountered an STL exception...``
warnings to stdout *on the same line before* the JSON result; the driver
scans for the JSON object anywhere in the line (an earlier version required
the line to start with "{" and misreported every such model as crashed).

Author note: written 2026-09 as the standard "did we harvest this firmware
correctly" check.  Point it at any future extracted tree.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

_CHECK_TEMPLATE = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from harvest.mlmodelc_loader import convert, make_test_inputs
bundle, out = sys.argv[1], sys.argv[2]
rec = None
try:
    rec = convert(bundle, out)
    if rec.status == "not_mil":
        # compiled-only bundle: check the native runtime can at least load it
        from harvest.mlmodelc_loader import load_native
        try:
            load_native(bundle)
            print(json.dumps({"status": "native_ok", "error": ""})); sys.exit(0)
        except Exception as e:
            print(json.dumps({"status": "native_fail", "error": str(e)[:300]})); sys.exit(0)
    if rec.status != "converted":
        print(json.dumps({"status": rec.status, "error": rec.error[:300]})); sys.exit(0)
    import coremltools as ct
    model = ct.models.MLModel(out)
    inputs = make_test_inputs(model)
    if any(v is None for v in inputs.values()):
        print(json.dumps({"status": "converted", "error": "predict skipped (image input)"})); sys.exit(0)
    outs = model.predict(inputs)
    shapes = {k: tuple(np.asarray(v).shape) for k, v in outs.items()}
    print(json.dumps({"status": "converted", "outputs": shapes})); sys.exit(0)
except Exception as e:
    print(json.dumps({"status": "predict_failed" if rec.status == "converted" else "convert_failed",
                      "error": str(e)[:300]}))
    sys.exit(0)
"""


def find_bundles(tree: str) -> list[str]:
    bundles = []
    for root, dirs, files in os.walk(tree):
        if root.endswith(".mlmodelc"):
            bundles.append(root)
    return sorted(bundles)


def run_one(bundle: str, out: str, timeout: int) -> dict:
    cmd = [
        sys.executable, "-c", _CHECK_TEMPLATE, bundle, out,
    ]
    env = dict(os.environ)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                # the CoreML compiler prints "E5RT encountered an STL
                # exception..." warnings to stdout *on the same line* before
                # the JSON, so scan for the JSON object anywhere in the line
                idx = line.find("{")
                if idx >= 0:
                    try:
                        return json.loads(line[idx:])
                    except json.JSONDecodeError:
                        continue
            return {"status": "crashed", "error": "no JSON on stdout"}
        return {
            "status": "crashed",
            "error": f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}",
        }
    except subprocess.TimeoutExpired:
        return {"status": "crashed", "error": f"timeout after {timeout}s"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tree", help="tree containing .mlmodelc bundles")
    ap.add_argument("outdir", help="where to write converted .mlpackages")
    ap.add_argument("--report", default="", help="JSON report path (default: <outdir>/report.json)")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    bundles = find_bundles(args.tree)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for i, bundle in enumerate(bundles):
        rel = os.path.relpath(bundle, args.tree)
        out_path = out_root / (rel + ".mlpackage")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = str(out_path)
        res = run_one(bundle, out, args.timeout)
        res["bundle"] = rel
        results.append(res)
        mark = {"converted": "OK ", "not_mil": "-- "}.get(res["status"], "!! ")
        print(f"{mark} {rel:<70} {res['status']:16} {res.get('error', '')[:60]}", flush=True)

    report = {
        "tree": args.tree,
        "counts": {s: sum(1 for r in results if r["status"] == s) for s in
                   {r["status"] for r in results}},
        "results": results,
    }
    report_path = args.report or str(out_root / "report.json")
    Path(report_path).write_text(json.dumps(report, indent=2))
    print(f"\n{json.dumps(report['counts'])}  → {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
