#!/usr/bin/env python3
"""Inventory the on-disk Apple Intelligence surface of the running macOS.

Walks /System/Library/AssetsV2 for com_apple_MobileAsset_UAF_* families and
records, per family: bundle count, payload bytes, du-style total, and every
plaintext artifact a local reader can reach without defeating anything —
system prompts (chat.role.system), promptTemplates, tokenizer maps, model_type,
and the per-variant Restore cryptex DMGs.

Output is JSON so two OS versions can be diffed mechanically.

    uv run python harvest/fm27_inventory.py            # writes datasets/uaf_surface_<os>.json
    uv run python harvest/fm27_inventory.py --diff a.json b.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any

ASSETS = "/System/Library/AssetsV2"
PREFIX = "com_apple_MobileAsset_UAF_"


def _walk(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def _du_mb(path: str) -> int:
    try:
        out = subprocess.run(["du", "-sm", path], capture_output=True, text=True, timeout=120)
        return int(out.stdout.split()[0])
    except Exception:
        return -1


def family(fam_dir: str) -> dict:
    name = os.path.basename(fam_dir)[len(PREFIX):]
    bundles = sorted(glob.glob(os.path.join(fam_dir, "**", "*.asset"), recursive=True))
    prompts: dict[str, int] = {}
    templates: dict[str, int] = {}
    model_types: dict[str, int] = {}
    tokens: dict[str, int] = {}
    dmg: list[int] = []
    for mp in glob.glob(os.path.join(fam_dir, "**", ".AssetData", "metadata.json"), recursive=True):
        try:
            d = json.load(open(mp))
        except Exception:
            continue
        for p, v in _walk(d):
            if not isinstance(v, str) or not v:
                continue
            if "chat.role.system" in p:
                prompts.setdefault(hashlib.sha256(v.encode()).hexdigest()[:8], len(v))
            elif "promptTemplates" in p:
                templates.setdefault(hashlib.sha256(v.encode()).hexdigest()[:8], len(v))
            elif p.endswith("/model_type"):
                model_types[v] = model_types.get(v, 0) + 1
            elif "special_tokens_map" in p:
                tokens[p.rsplit("/", 1)[-1]] = tokens.get(p.rsplit("/", 1)[-1], 0) + 1
    for dm in glob.glob(os.path.join(fam_dir, "**", "*.dmg"), recursive=True):
        try:
            dmg.append(os.path.getsize(dm))
        except OSError:
            pass
    return {
        "family": name,
        "bundles": len(bundles),
        "du_mb": _du_mb(fam_dir),
        "prompts": {"count": len(prompts), "max_len": max(prompts.values()) if prompts else 0,
                    "hashes": dict(sorted(prompts.items(), key=lambda kv: -kv[1])[:40])},
        "templates": {"count": len(templates)},
        "model_types": dict(sorted(model_types.items(), key=lambda kv: -kv[1])),
        "token_map_keys": tokens,
        "dmg_count": len(dmg),
        "dmg_bytes": sum(dmg),
        "dmg_max": max(dmg) if dmg else 0,
    }


def inventory() -> dict:
    v = platform.mac_ver()[0] or "unknown"
    build = ""
    try:
        out = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True, timeout=10)
        build = out.stdout.strip()
    except Exception:
        pass
    fams = [family(d) for d in sorted(glob.glob(os.path.join(ASSETS, PREFIX + "*"))) if os.path.isdir(d)]
    return {
        "os_version": v,
        "os_build": build,
        "host": platform.machine(),
        "families": fams,
        "totals": {
            "families": len(fams),
            "bundles": sum(f["bundles"] for f in fams),
            "du_mb": sum(max(f["du_mb"], 0) for f in fams),
            "distinct_prompts": sum(f["prompts"]["count"] for f in fams),
            "dmgs": sum(f["dmg_count"] for f in fams),
            "dmg_bytes": sum(f["dmg_bytes"] for f in fams),
        },
    }


def diff(a: dict, b: dict) -> None:
    print(f"{a['os_version']} ({a.get('os_build','')})  ->  {b['os_version']} ({b.get('os_build','')})")
    fa = {f["family"]: f for f in a["families"]}
    fb = {f["family"]: f for f in b["families"]}
    added = sorted(set(fb) - set(fa))
    removed = sorted(set(fa) - set(fb))
    print(f"\nnew families ({len(added)}): {', '.join(added) if added else '-'}")
    print(f"gone families ({len(removed)}): {', '.join(removed) if removed else '-'}")
    print(f"\n{'family':34s} {'du MB':>10s} {'bundles':>8s} {'prompts':>8s}  delta")
    for name in sorted(set(fa) & set(fb), key=lambda n: -fb[n]["du_mb"]):
        x, y = fa[name], fb[name]
        d = y["du_mb"] - x["du_mb"]
        if abs(d) >= 50 or x["prompts"]["count"] != y["prompts"]["count"]:
            print(f"{name:34s} {x['du_mb']:5d}->{y['du_mb']:5d} {x['bundles']:4d}->{y['bundles']:4d} "
                  f"{x['prompts']['count']:4d}->{y['prompts']['count']:4d}  {d:+d} MB")
    pa = {h for f in a["families"] for h in f["prompts"]["hashes"]}
    pb = {h for f in b["families"] for h in f["prompts"]["hashes"]}
    print(f"\nprompt hashes: {len(pa)} -> {len(pb)}  |  new {len(pb - pa)}, retained {len(pa & pb)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"))
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.diff:
        diff(json.load(open(args.diff[0])), json.load(open(args.diff[1])))
        return
    inv = inventory()
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets", f"uaf_surface_{inv['os_version']}_{inv['os_build']}.json")
    json.dump(inv, open(out, "w"), indent=1)
    t = inv["totals"]
    print(f"{inv['os_version']} ({inv['os_build']}) {inv['host']}: "
          f"{t['families']} families, {t['bundles']} bundles, {t['du_mb']} MB, "
          f"{t['distinct_prompts']} distinct prompts, {t['dmgs']} DMGs "
          f"({t['dmg_bytes'] / 2**30:.1f} GiB)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
