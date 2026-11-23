#!/usr/bin/env python3
"""Evidence sweep: are Apple's on-device model cryptex images encrypted?

For every *.dmg under the UAF asset tree, record `hdiutil isencrypted`, the
image class if recoverable, size, and (optionally, --mount N) whether an
unprivileged user can mount it and read model files out of it.

    uv run python harvest/dmg_encryption_sweep.py
    uv run python harvest/dmg_encryption_sweep.py --mount 3

Writes datasets/dmg_encryption_<os>_<build>.json — evidence for disclosure.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import plistlib
import subprocess
import tempfile

ASSETS = "/System/Library/AssetsV2"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return -1, str(exc)


def is_encrypted(path: str) -> bool | None:
    rc, out = _run(["hdiutil", "isencrypted", path])
    if rc != 0:
        return None
    return "encrypted: YES" in out or out.strip().endswith("YES")


def image_info(path: str) -> dict:
    rc, out = _run(["hdiutil", "imageinfo", path])
    info = {}
    if rc == 0:
        for key in ("Class Name", "Format", "Encrypted"):
            for line in out.splitlines():
                if line.strip().startswith(key):
                    info[key] = line.split(":", 1)[1].strip()
    return info


def mount_probe(path: str) -> dict:
    mnt = tempfile.mkdtemp(prefix="dmgprobe_")
    rc, out = _run(["hdiutil", "attach", "-readonly", "-noverify", "-noautoopen",
                    "-mountpoint", mnt, path])
    res: dict = {"mounted": rc == 0}
    if rc == 0:
        files = [f for f in glob.glob(os.path.join(mnt, "**", "*"), recursive=True)
                 if os.path.isfile(f)]
        res["files"] = len(files)
        res["sample"] = [os.path.relpath(f, mnt) for f in files[:6]]
        res["has_mil"] = any(f.endswith(".mil") for f in files)
        res["has_weights"] = any(os.path.basename(f) in ("weights.bin",) for f in files)
        _run(["hdiutil", "detach", mnt, "-force"])
    else:
        res["error"] = out.splitlines()[-1] if out else ""
    try:
        os.rmdir(mnt)
    except OSError:
        pass
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount", type=int, default=0,
                    help="mount this many images and record what is readable")
    ap.add_argument("--pattern", default="com_apple_MobileAsset_UAF_FM_*")
    args = ap.parse_args()

    dmgs = sorted(glob.glob(os.path.join(ASSETS, args.pattern, "**", "*.dmg"), recursive=True))
    # SecureMobileAsset.dmg is a 0-byte symlink target in most bundles
    dmgs = [d for d in dmgs if os.path.getsize(d) > 0]

    rows = []
    for path in dmgs:
        enc = is_encrypted(path)
        row = {
            "path": os.path.relpath(path, ASSETS),
            "family": path[len(ASSETS) + 1:].split("/")[0].replace("com_apple_MobileAsset_UAF_", ""),
            "bytes": os.path.getsize(path),
            "encrypted": enc,
            "imageinfo": image_info(path),
        }
        rows.append(row)

    mounted = []
    for row in [r for r in rows if r["bytes"] > 10 * 1024 * 1024][: args.mount] if args.mount else []:
        abs_path = os.path.join(ASSETS, row["path"])
        mounted.append({"path": row["path"], **mount_probe(abs_path)})
        print(f"  mount probe: {os.path.basename(row['path'])[:60]} -> {mounted[-1]['mounted']}")

    build = _run(["sw_vers", "-buildVersion"])[1] or "unknown"
    out = {
        "os_version": platform.mac_ver()[0],
        "os_build": build,
        "host": platform.machine(),
        "pattern": args.pattern,
        "images": len(rows),
        "encrypted_yes": sum(1 for r in rows if r["encrypted"] is True),
        "encrypted_no": sum(1 for r in rows if r["encrypted"] is False),
        "unknown": sum(1 for r in rows if r["encrypted"] is None),
        "total_bytes": sum(r["bytes"] for r in rows),
        "mount_probes": mounted,
        "detail": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = os.path.join(OUT_DIR, f"dmg_encryption_{out['os_version']}_{build}.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"{out['os_version']} ({build}): {out['images']} images, "
          f"encrypted YES={out['encrypted_yes']} NO={out['encrypted_no']} "
          f"unknown={out['unknown']}, {out['total_bytes'] / 2**30:.1f} GiB")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
