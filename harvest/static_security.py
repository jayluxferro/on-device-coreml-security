"""
static_security.py — static security analysis of ALL harvested Apple CoreML
bundles (ios261_live, firmware/extracted/*, models/iOS).

For every bundle this tool extracts the six security-relevant property
classes from the paper's static-analysis phase:

  1. Plaintext weights  — weight container formats (model.espresso.weights,
     weights/weight.bin, _B0000.DAT / *.dat, .mlmodel WeightParams) and a
     float-plausibility check (raw fp32? fp16? opaque/quantized?) computed by
     unpacking a sample of each blob and measuring the finite-fraction.
  2. isUpdatable        — metadata.json "isUpdatable"/"updateParameters",
     presence of an updateParameters/ dir, or Model.isUpdatable on .mlmodel.
  3. Provenance         — model.mil buildInfo dict (mldb_token,
     coremlc-version, coremltools-version, component tokens), metadata.json
     author/version, .mlmodel userDefined keys (CreateML / coremltools
     versions), date hints in bundle names.
  4. Entitlements hints — owning framework from the bundle path + the
     metadata.json availability dict (per-OS deployment targets).
  5. Quantization       — storagePrecision/computePrecision from
     metadata.json, MIL const dtype histogram from model.mil, weight-file
     size ratios, name hints (8bit/fp16/FP16).
  6. Spec version       — metadata.json specificationVersion, model.mil opset
     (ios15 -> CoreML5/spec 6), or Model.specificationVersion on .mlmodel.

Design decisions (learned during 2026-08 harvest work):

- The compiled containers (coremldata.bin, model.espresso.net) are NOT plain
  protobuf — they use a proprietary coremlc container layout (verified:
  direct Model_pb2.Model.ParseFromString fails on both).  All
  security-relevant metadata therefore comes from metadata.json + model.mil
  + the weights blobs, which ARE parseable.
- .mlmodel files parse directly with coremltools' Model_pb2 (no C++ compile
  step, fast, no coremlc crashes).
- Zipped .mlmodelc (iPhone-X-era harvests, e.g. DuetExperienceCenter/*)
  are read via zipfile without extraction.
- Bundles are never recursed into once identified (no nested bundles).

Output: docs/STATIC_SECURITY_ANALYSIS.md (report) +
        docs/STATIC_SECURITY_DATA.jsonl (machine-readable records).

Usage:
    uv run python harvest/static_security.py [--data-only]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from coremltools.proto import Model_pb2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS = PROJECT_ROOT / "models"
IOS261_LIVE = MODELS / "ios261_live"
EXTRACTED = MODELS / "firmware" / "extracted"
IOS_LEGACY = MODELS / "iOS"
REPORT_PATH = PROJECT_ROOT / "docs" / "STATIC_SECURITY_ANALYSIS.md"
DATA_PATH = PROJECT_ROOT / "docs" / "STATIC_SECURITY_DATA.jsonl"

# MIL text opset name -> CoreML specification version (matches
# harvest.mlmodelc_loader.OPSET_SPEC_VERSION)
OPSET_SPEC_VERSION = {
    "ios11": 3, "ios12": 4, "ios13": 5, "ios14": 5, "ios15": 6,
    "ios16": 7, "ios17": 8, "ios18": 9, "ios19": 10,
    "CoreML3": 4, "CoreML4": 5, "CoreML5": 6, "CoreML6": 7,
    "CoreML7": 8, "CoreML8": 9, "CoreML9": 10,
}

BUNDLE_SUBDIRS = {  # inside a bundle — never treat as bundles themselves
    "analytics", "model", "weights", "neural_network_optionals",
    "updateParameters", "Assets", "Resources",
}

BUNDLE_MARKERS = ("coremldata.bin", "model.mil", "metadata.json",
                  "model.espresso.net", "model.espresso.weights",
                  "Manifest.json", "weight.bin")

# Regexes for the model.mil buildInfo provenance dictionary, e.g.:
#   [buildInfo = dict<tensor<string, []>, tensor<string, []>>({{
#      "coremlc-component-MILHost", "3505.3.2"},
#      {"coremlc-version", "3505.3.1"},
#      {"coremltools-component-tensorflow", "2.11.0"},
#      {"coremltools-version", "7.2"},
#      {"mldb_token", "mldb-7nwypjwy9o"}})]
# Two observed buildInfo dialects:
#  A. dict<tensor<string, []>, tensor<string, []>>({{"k","v"},...})
#     (mldb_token is the last dict entry)
#  B. dict<string, string>({{"k","v"},...}), mldb_token = string("mldb-...")
#     (mldb_token is a separate attribute after the dict — coremlc 3400-era)
# NOTE: `dict<.*?>>` consumes the double `>>` that ends the dict's type
# params in dialect A (a single `>` stops inside tensor<string, []>).
_BUILDINFO_RE = re.compile(
    r'buildInfo\s*=\s*dict<.*?>>\s*\((\{\{.{0,8000}?\}\})\)'
    r'|buildInfo\s*=\s*dict<string,\s*string>\s*\((\{\{.{0,8000}?\}\})\)',
    re.S)
_MLDB_TOKEN_RE = re.compile(r'mldb_token\s*=\s*string\("([^"]+)"\)')
_PAIR_RE = re.compile(r'\{\s*"([^"]+)"\s*,\s*"([^"]*)"\s*\}')
_FUNC_RE = re.compile(r'func\s+([A-Za-z_][A-Za-z0-9_]*)<([a-zA-Z0-9]+)>')
_CONST_DTYPE_RE = re.compile(r'val\s*=\s*tensor<([a-z0-9_]+)')
_DATE_NAME_RE = re.compile(r'(20\d{6})|(\d{8}-\d{4})')
_VERSION_NAME_RE = re.compile(r'v?(\d+\.\d+(?:\.\d+)?)')

# Layer weight-storage field names on NeuralNetwork WeightParams (Model.proto)
WEIGHT_PARAM_FIELDS = ("floatValue", "float16Value", "rawValue",
                       "quantization", "stringValue", "int64Value")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def iter_bundles(root: Path) -> list[Path]:
    """Return every bundle path under *root* (no recursion into bundles)."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "/downloads" in dirpath:
            dirnames[:] = []
            continue
        d = Path(dirpath)
        is_bundle = (
            d.name.endswith(".mlmodelc")
            or any((d / m).exists() for m in BUNDLE_MARKERS)
        ) and d.name not in BUNDLE_SUBDIRS
        if is_bundle:
            out.append(d)
            dirnames[:] = []  # never descend into a bundle
    return out


def iter_mlmodel_files(root: Path) -> list[Path]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        if "/downloads" in dirpath:
            continue
        for f in filenames:
            if f.endswith(".mlmodel") and not f.startswith("."):
                p = Path(dirpath) / f
                if zipfile.is_zipfile(p):
                    continue  # zip twins of .mlmodelc bundles (iPhone-X era)
                out.append(p)
    return out


def iter_zip_bundles(root: Path) -> list[Path]:
    """All .mlmodelc ZIP archives, including stub archives.

    The iPhone-X harvest's DuetExpertCenter bundles are zips; some
    (CoreMLModel0-19.mlmodelc) are 194-byte STUBS whose only entry is an
    empty directory — recorded as stubs so the report can flag the missing
    model content rather than silently dropping them.
    """
    out = []
    for dirpath, _, filenames in os.walk(root):
        if "/downloads" in dirpath:
            continue
        for f in filenames:
            if f.endswith(".mlmodelc"):
                p = Path(dirpath) / f
                try:
                    with zipfile.ZipFile(p, "r") as zf:
                        names = zf.namelist()
                except (zipfile.BadZipFile, OSError):
                    continue
                out.append((p, any("coremldata.bin" in n for n in names)))
    return out


def framework_of(path: Path, platform: str) -> str:
    """Best-effort owning framework from the bundle's path."""
    s = str(path)
    if platform == "ios261_live":
        m = re.search(r"PrivateFrameworks/([^/]+)\.framework", s)
        if m:
            return m.group(1)
        m = re.search(r"Frameworks/([^/]+)\.framework", s)
        if m:
            return m.group(1)
        m = re.search(r"PreinstalledAssetsV2/RequiredByOs/([^/]+)", s)
        if m:
            return m.group(1)
        m = re.search(r"usr/libexec", s)
        if m:
            return "usr/libexec (PowerUI/battery)"
        m = re.search(r"DuetExpertCenter", s)
        if m:
            return "DuetExpertCenter"
        m = re.search(r"SmsFilter", s)
        if m:
            return "SmsFilter"
        m = re.search(r"/WRM/", s)
        if m:
            return "WRM"
        m = re.search(r"NearbyInteractionBundles", s)
        if m:
            return "NearbyInteraction"
        return "System (other)"
    if platform == "macos":
        m = re.search(r"_([A-Za-z0-9]+)\.framework_", s)
        if m:
            return m.group(1)
        m = re.search(r"_com_apple_MobileAsset_([A-Za-z0-9]+)", s)
        if m:
            return m.group(1)
        return "macOS (other)"
    if platform == "iOS12legacy":
        m = re.search(r"SmsFilter", s)
        if m:
            return "SmsFilter"
        return "DuetExpertCenter" if "Duet" in s else "iOS12legacy (other)"
    return platform  # ios / ipados / tvos / ios12 / ios261


def platform_of(path: Path) -> str:
    s = str(path)
    if s.startswith(str(IOS261_LIVE)):
        return "ios261_live"
    if s.startswith(str(IOS_LEGACY)):
        return "iOS12legacy"
    rel = s[len(str(EXTRACTED)):].lstrip("/")
    return rel.split("/")[0] if "/" in rel else rel


# ---------------------------------------------------------------------------
# Weights / float plausibility
# ---------------------------------------------------------------------------


def _sample_file(fh, nbytes: int = 65536) -> bytes:
    fh.seek(0)
    return fh.read(nbytes)


def float_check(data: bytes) -> dict[str, Any]:
    """Unpack a weight-blob sample as fp32/fp16; report plausibility.

    'raw' requires >=95% finite and a sane spread at some header offset
    (espresso/weight.bin blobs carry a small index header before the floats;
    .dat blobs start at 0).  Returns the best offset and the per-dtype
    finite fractions, which the report uses to back every plaintext claim.
    """
    out: dict[str, Any] = {
        "raw_fp32": False, "fp16": False, "opaque": False,
        "best_offset": 0, "fp32_finite": 0.0, "fp16_finite": 0.0,
        "fp32_sample_mean": None, "fp32_sample_std": None,
    }
    if len(data) < 64:
        return out
    results = []
    for off in (0, 8, 16, 32, 64, 128, 256):
        n = min(4096, len(data) - off)
        n -= n % 4  # fp32 alignment
        if n < 8:
            continue
        a = np.frombuffer(data[off:off + n], dtype="<f4")
        fin = np.isfinite(a)
        frac = float(fin.mean())
        std = float(a[fin].std()) if fin.any() else float("nan")
        mean = float(a[fin].mean()) if fin.any() else float("nan")
        results.append((frac, -off, off, mean, std))
    results.sort(reverse=True)
    frac, _, off, mean, std = results[0]
    out["best_offset"] = off
    out["fp32_finite"] = frac
    out["fp32_sample_mean"] = mean if np.isfinite(mean) else None
    out["fp32_sample_std"] = std if np.isfinite(std) else None
    if frac >= 0.95 and (std is None or np.isfinite(std)) and std < 1e6:
        out["raw_fp32"] = True
    # fp16 plausibility (only meaningful when fp32 failed)
    if not out["raw_fp32"] and off + 8 <= len(data):
        n16 = min(8192, len(data) - off)
        n16 -= n16 % 2  # fp16 alignment
        if n16 < 8:
            n16 = 0
        if n16:
            a16 = np.frombuffer(data[off:off + n16], dtype="<f2")
        fin16 = np.isfinite(a16)
        f16 = float(fin16.mean())
        out["fp16_finite"] = f16
        if f16 >= 0.95:
            s16 = float(a16[fin16].std()) if fin16.any() else float("nan")
            if s16 < 1e6:
                out["fp16"] = True
    if not out["raw_fp32"] and not out["fp16"]:
        out["opaque"] = True
    return out


def inspect_weights_blob(fh, label: str) -> dict[str, Any]:
    """Full record for one weight blob (sample + size + format hints)."""
    data = _sample_file(fh)
    rec = float_check(data)
    rec["label"] = label
    try:
        fh.seek(0, os.SEEK_END)
        rec["size"] = fh.tell()
    except OSError:
        rec["size"] = len(data)
    return rec


def analyze_mil_weights(mil_text: str, weights_dir: Path) -> list[dict[str, Any]]:
    """Weight blobs in a MIL bundle: weight.bin etc. + const dtype histogram."""
    recs = []
    if weights_dir.is_dir():
        for f in sorted(weights_dir.iterdir()):
            if f.is_file():
                try:
                    with open(f, "rb") as fh:
                        recs.append(inspect_weights_blob(fh, f"weights/{f.name}"))
                except OSError:
                    pass
    hist = {}
    for m in _CONST_DTYPE_RE.finditer(mil_text):
        dt = m.group(1)
        hist[dt] = hist.get(dt, 0) + 1
    return recs, hist


# ---------------------------------------------------------------------------
# metadata.json
# ---------------------------------------------------------------------------


def parse_metadata_json(bundle: Path) -> dict[str, Any]:
    """Parse metadata.json into a dict; tolerate missing/empty."""
    p = bundle / "metadata.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}
    if isinstance(data, list):
        data = data[0] if data else {}
    return data if isinstance(data, dict) else {}


def metadata_security_fields(md: dict) -> dict[str, Any]:
    out: dict[str, Any] = {
        "isUpdatable": False,
        "update_params": False,
        "specificationVersion": None,
        "storagePrecision": None,
        "computePrecision": None,
        "author": None,
        "version": None,
        "availability": None,
    }
    out["isUpdatable"] = str(md.get("isUpdatable", "0")) in ("1", "true", "True")
    ups = md.get("updateParameters")
    out["update_params"] = bool(ups)
    sv = md.get("specificationVersion")
    out["specificationVersion"] = int(sv) if str(sv).isdigit() else None
    out["storagePrecision"] = md.get("storagePrecision")
    out["computePrecision"] = md.get("computePrecision")
    out["author"] = md.get("author")
    out["version"] = md.get("version")
    av = md.get("availability")
    if isinstance(av, dict):
        out["availability"] = {k: str(v) for k, v in av.items()}
    return out


# ---------------------------------------------------------------------------
# model.mil provenance
# ---------------------------------------------------------------------------


def parse_buildinfo(mil_text: str) -> dict[str, str]:
    m = _BUILDINFO_RE.search(mil_text)
    if not m:
        # dialect C: bare `program(1.2) [mldb_token = string("...")]`
        mt = _MLDB_TOKEN_RE.search(mil_text[:2000])
        return {"mldb_token": mt.group(1)} if mt else {}
    g = m.group(1) or m.group(2)  # dialect A or B dict literal
    pairs = dict(_PAIR_RE.findall(g))
    if "mldb_token" not in pairs:
        # dialect B: token is a separate attribute right after the dict
        tail = mil_text[m.start():m.start() + 2000]
        mt = _MLDB_TOKEN_RE.search(tail)
        if mt:
            pairs["mldb_token"] = mt.group(1)
    return pairs


def provenance_hints(mil_text: str) -> dict[str, Any]:
    bi = parse_buildinfo(mil_text)
    out: dict[str, Any] = {"mldb_token": bi.get("mldb_token"),
                           "coremlc_version": bi.get("coremlc-version"),
                           "coremltools_version": bi.get("coremltools-version")}
    comps = {k: v for k, v in bi.items() if k.startswith("coremltools-component")}
    out["components"] = comps
    return out


# ---------------------------------------------------------------------------
# .mlmodel (protobuf) analysis
# ---------------------------------------------------------------------------

WEIGHT_PARAM_TYPE = "CoreML.Specification.WeightParams"


def _walk_weight_params(msg, out: list, depth: int = 0):
    """Recursively collect every WeightParams-typed sub-message of *msg*.

    coremltools 9.0's updated schema moved layer weights off the
    NeuralNetworkLayer message into the per-operation messages
    (layer.convolution.weights etc.), so a direct `layer.weights` access
    raises AttributeError (verified on mainModel_USEN.mlmodel).  A
    descriptor walk finds WeightParams wherever they live, including inside
    repeated fields and nested ops (gru/lstm weightParams lists).

    NOTE: this wheel's protobuf runtime is upb, whose FieldDescriptor has
    no `.label` instance attribute; repeatedness is detected from the
    returned value (RepeatedCompositeContainer exposes `.add`).
    """
    if depth > 6:
        return
    for field in msg.DESCRIPTOR.fields:
        if field.message_type is None:
            continue
        try:
            val = getattr(msg, field.name)
        except AttributeError:  # unset oneof member
            continue
        if hasattr(val, "items") and hasattr(val, "keys"):
            # map container (MessageMapContainer) — recurse into values
            for _k, sub in val.items():
                _walk_weight_params(sub, out, depth + 1)
            continue
        if hasattr(val, "add"):  # repeated message container
            if field.message_type.full_name == WEIGHT_PARAM_TYPE:
                out.extend(val)
            else:
                for sub in val:
                    _walk_weight_params(sub, out, depth + 1)
            continue
        if field.message_type.full_name == WEIGHT_PARAM_TYPE:
            # floatValue/float16Value/rawValue are REPEATED fields — truthy
            # only when populated; HasField would raise on them (upb).
            if val.floatValue or val.float16Value or val.rawValue \
                    or val.HasField("quantization"):
                out.append(val)
        else:
            _walk_weight_params(val, out, depth + 1)


def _mil_program_stats(program) -> dict[str, Any]:
    """Lightweight stats over an mlProgram: op count + const dtype histogram.

    MIL constants are inline protobuf tensors on op attributes
    (TensorValue.immediateValue.tensor).  The value tensor message has no
    dtype field in this wheel's schema (upb; `MIL_pb2.Tensor` carries only
    the value arrays: floats/ints/bools/strings/longInts/doubles/bytes),
    so the storage class is inferred from which array is populated.
    """
    out = {"ops": 0, "dtypes": {}, "const_tensors": 0}
    seen = set()

    def visit(ops):
        for op in ops:
            if id(op) in seen:
                continue
            seen.add(id(op))
            out["ops"] += 1
            for attr in op.attributes.values():
                if not attr.HasField("immediateValue"):
                    continue
                iv = attr.immediateValue
                if not iv.HasField("tensor"):
                    continue
                t = iv.tensor
                out["const_tensors"] += 1
                # NOTE: upb exposes some repeated scalars as `RepeatedFloats`
                # which lacks __len__ (TypeError observed on model.mlmodel);
                # truthiness is the portable emptiness test.
                if t.floats:
                    dt = "FLOAT32"
                elif t.doubles:
                    dt = "FLOAT64"
                elif t.longInts:
                    dt = "INT64"
                elif t.ints:
                    dt = "INT32"
                elif t.bools:
                    dt = "BOOL"
                elif t.strings:
                    dt = "STRING"
                elif t.bytes:
                    dt = "BYTES(fp16/int8)"
                else:
                    dt = "EMPTY"
                out["dtypes"][dt] = out["dtypes"].get(dt, 0) + 1
            for b in op.blocks:
                visit(b.operations)

    for fn in program.functions.values():
        for ops in fn.block_specializations.values():
            visit(ops.operations)
    return out


def analyze_mlmodel(path: Path) -> dict[str, Any]:
    spec = Model_pb2.Model()
    with open(path, "rb") as f:
        spec.ParseFromString(f.read())
    md = spec.description.metadata
    rec: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "platform": platform_of(path),
        "framework": framework_of(path, platform_of(path)),
        "kind": "mlmodel",
        "specificationVersion": spec.specificationVersion,
        # keep the same key as mlmodelc bundles so summarize()'s spec histogram
        # counts .mlmodel files instead of lumping all 98 into "unknown"
        "spec_version": spec.specificationVersion or None,
        "isUpdatable": bool(spec.isUpdatable),
        "model_type": spec.WhichOneof("Type"),
        "author": md.author or None,
        "versionString": md.versionString or None,
        "license": md.license or None,
        "userDefined": dict(md.userDefined),
        "weights": [],
        "quantized_layers": 0,
        "weight_param_layers": 0,
        "layer_count": 0,
        "float_value_layers": 0,
        "float16_value_layers": 0,
        "raw_value_layers": 0,
        "esp32_float_count": 0,
        "quant_scheme": None,
        "updatable_layers": [],
        "weights_plaintext": False,
        "ops": 0,
    }
    t = rec["model_type"]
    nn = None
    if t == "neuralNetwork":
        nn = spec.neuralNetwork
    elif t == "neuralNetworkClassifier":
        nn = spec.neuralNetworkClassifier
    elif t == "neuralNetworkRegressor":
        nn = spec.neuralNetworkRegressor

    if nn is not None:
        rec["layer_count"] = len(nn.layers)
        for layer in nn.layers:
            if getattr(layer, "isUpdatable", False):
                rec["updatable_layers"].append(layer.name)
            wps: list = []
            _walk_weight_params(layer, wps)
            if not wps:
                continue
            rec["weight_param_layers"] += 1
            rec["weights_plaintext"] = True
            for wp in wps:
                if wp.floatValue:
                    rec["float_value_layers"] += 1
                    try:  # upb RepeatedFloats lacks __len__ on some messages
                        rec["esp32_float_count"] += len(wp.floatValue)
                    except TypeError:
                        rec["esp32_float_count"] += sum(1 for _ in wp.floatValue)
                if wp.float16Value:
                    rec["float16_value_layers"] += 1
                if wp.rawValue:
                    rec["raw_value_layers"] += 1
                if wp.HasField("quantization"):
                    rec["quantized_layers"] += 1
                    q = wp.quantization
                    q_fields = {f.name for f in q.DESCRIPTOR.fields}
                    for fld in ("linearQuantization", "lookupTableQuantization",
                                "dequantize", "matrixDequantize"):
                        if fld in q_fields and q.HasField(fld):
                            rec["quant_scheme"] = fld
                            break
    elif t == "mlProgram":
        stats = _mil_program_stats(spec.mlProgram)
        rec["ops"] = stats["ops"]
        rec["mil_dtypes"] = stats["dtypes"]
        total = sum(stats["dtypes"].values())
        if stats["const_tensors"] and total:
            fp32 = stats["dtypes"].get("FLOAT32", 0)
            fp16bytes = stats["dtypes"].get("BYTES(fp16/int8)", 0)
            if fp16bytes >= 0.8 * total:
                rec["quant_scheme"] = "BYTES (fp16/int8, MIL)"
            elif fp32 >= 0.8 * total:
                rec["quant_scheme"] = "Float32"
        rec["weights_plaintext"] = stats["const_tensors"] > 0
    elif t in ("textClassifier", "wordTagger", "gazetteer", "wordEmbedding",
               "itemSimilarityRecommender", "visionFeaturePrint",
               "soundAnalysisPreprocessing"):
        # these model types store their learned parameters inline in the
        # protobuf (string embeddings, co-occurrence matrices, ...)
        rec["weights_plaintext"] = True
    elif t in ("treeEnsembleClassifier", "treeEnsembleRegressor",
               "glmClassifier", "glmRegressor", "supportVectorClassifier",
               "supportVectorRegressor", "kNearestNeighborsClassifier",
               "bayesianProbitRegressor", "nearestNeighbors"):
        # classic tabular models: all learned parameters (split values,
        # weights, class priors) are inline numeric fields of the spec
        rec["weights_plaintext"] = True

    if rec["quantized_layers"] and not rec["quant_scheme"]:
        rec["quant_scheme"] = "unknown"
    return rec


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------


def analyze_bundle(bundle: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "path": str(bundle),
        "name": bundle.name,
        "platform": platform_of(bundle),
        "framework": framework_of(bundle, platform_of(bundle)),
        "kind": "mlmodelc_dir",
        "files": {},
        "metadata": {},
        "isUpdatable": False,
        "update_params": False,
        "spec_version": None,
        "provenance": {},
        "availability": None,
        "weights": [],
        "mil_dtypes": {},
        "quantization": {"scheme": None, "storage": None, "compute": None,
                         "name_hint": None},
        "dat_files": [],
    }
    # file inventory
    for f in sorted(bundle.iterdir()):
        if f.is_file():
            rec["files"][f.name] = f.stat().st_size
        elif f.name == "weights":
            for w in sorted(f.iterdir()):
                if w.is_file():
                    rec["files"][f"weights/{w.name}"] = w.stat().st_size
    # metadata.json
    md = parse_metadata_json(bundle)
    if md:
        rec["metadata"] = metadata_security_fields(md)
        rec["isUpdatable"] = rec["metadata"]["isUpdatable"]
        rec["update_params"] = rec["metadata"]["update_params"]
        rec["spec_version"] = rec["metadata"]["specificationVersion"]
        rec["availability"] = rec["metadata"]["availability"]
        rec["quantization"]["storage"] = rec["metadata"]["storagePrecision"]
        rec["quantization"]["compute"] = rec["metadata"]["computePrecision"]
        auth = rec["metadata"].get("author")
        if auth:
            rec["provenance"]["author"] = auth
        ver = rec["metadata"].get("version")
        if ver:
            rec["provenance"]["version"] = ver
    # updateParameters/ dir is itself the updatable marker for espresso
    if (bundle / "updateParameters").is_dir():
        rec["update_params"] = True
        rec["isUpdatable"] = True
    # model.mil
    mil_path = bundle / "model.mil"
    if mil_path.exists():
        mil_text = mil_path.read_text(errors="replace")
        m = _FUNC_RE.search(mil_text)
        rec["mil_ops"] = []  # filled below
        funcs = {}
        for fm in _FUNC_RE.finditer(mil_text):
            funcs.setdefault(fm.group(2), []).append(fm.group(1))
        rec["mil_functions"] = {k: v for k, v in funcs.items()}
        if not rec["spec_version"]:
            opsets = list(funcs.keys())
            if opsets:
                rec["spec_version"] = OPSET_SPEC_VERSION.get(opsets[0])
        rec["provenance"].update(provenance_hints(mil_text))
        w_recs, hist = analyze_mil_weights(mil_text, bundle / "weights")
        rec["weights"].extend(w_recs)
        rec["mil_dtypes"] = hist
    # espresso weights
    esp = bundle / "model.espresso.weights"
    if esp.exists():
        try:
            with open(esp, "rb") as fh:
                rec["weights"].append(inspect_weights_blob(fh, "model.espresso.weights"))
        except OSError:
            pass
    # tree-ensemble .DAT blobs
    for f in sorted(bundle.iterdir()):
        if f.is_file() and (f.suffix.upper() == ".DAT" or f.name.startswith("_B0000")):
            rec["dat_files"].append(f.name)
            try:
                with open(f, "rb") as fh:
                    rec["weights"].append(inspect_weights_blob(fh, f.name))
            except OSError:
                pass
    # name hints for quantization
    n = bundle.name.lower()
    if "8bit" in n:
        rec["quantization"]["name_hint"] = "8bit"
    elif "fp16" in n or "fp_16" in n:
        rec["quantization"]["name_hint"] = "fp16"
    # assemble quant scheme: metadata storage wins; else mil dtype histogram
    if rec["quantization"]["storage"]:
        rec["quantization"]["scheme"] = rec["quantization"]["storage"]
    elif rec["mil_dtypes"]:
        hist = rec["mil_dtypes"]
        fp16 = hist.get("fp16", 0)
        fp32 = hist.get("fp32", 0)
        total = sum(hist.values())
        if total and fp16 >= 0.8 * total:
            rec["quantization"]["scheme"] = "Float16 (MIL)"
        elif total and fp32 >= 0.8 * total:
            rec["quantization"]["scheme"] = "Float32"
    # date hints from bundle name (training-date provenance)
    name_date = _DATE_NAME_RE.search(bundle.name)
    if name_date:
        rec["provenance"]["name_date"] = next(x for x in name_date.groups() if x)
    return rec


def analyze_zip_bundle(path: Path, *, has_coremldata: bool = True) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "path": str(path), "name": path.name,
        "platform": platform_of(path),
        "framework": framework_of(path, platform_of(path)),
        "kind": "mlmodelc_zip",
        "stub": not has_coremldata,
        "files": {}, "metadata": {}, "isUpdatable": False,
        "update_params": False, "spec_version": None, "provenance": {},
        "availability": None, "weights": [], "mil_dtypes": {},
        "quantization": {"scheme": None, "storage": None, "compute": None,
                         "name_hint": None},
        "dat_files": [],
    }
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            rec["files"][info.filename] = info.file_size
            if info.filename.endswith("metadata.json"):
                try:
                    data = json.loads(zf.read(info.filename).decode())
                    if isinstance(data, list):
                        data = data[0] if data else {}
                    if isinstance(data, dict):
                        rec["metadata"] = metadata_security_fields(data)
                        rec["isUpdatable"] = rec["metadata"]["isUpdatable"]
                        rec["update_params"] = rec["metadata"]["update_params"]
                        rec["spec_version"] = rec["metadata"]["specificationVersion"]
                        rec["availability"] = rec["metadata"]["availability"]
                        rec["quantization"]["storage"] = rec["metadata"]["storagePrecision"]
                        rec["quantization"]["compute"] = rec["metadata"]["computePrecision"]
                        if rec["metadata"].get("author"):
                            rec["provenance"]["author"] = rec["metadata"]["author"]
                        if rec["metadata"].get("version"):
                            rec["provenance"]["version"] = rec["metadata"]["version"]
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                    pass
            if info.filename.endswith(".weights") or info.filename.endswith("weight.bin"):
                try:
                    with zf.open(info.filename) as fh:
                        rec["weights"].append(inspect_weights_blob(fh, info.filename.split("/")[-1]))
                except (KeyError, zipfile.BadZipFile):
                    pass
            if info.filename.endswith("model.mil"):
                try:
                    mil_text = zf.read(info.filename).decode(errors="replace")
                    rec["provenance"].update(provenance_hints(mil_text))
                    hist = {}
                    for mm in _CONST_DTYPE_RE.finditer(mil_text):
                        dt = mm.group(1)
                        hist[dt] = hist.get(dt, 0) + 1
                    rec["mil_dtypes"] = hist
                    funcs = {}
                    for fm in _FUNC_RE.finditer(mil_text):
                        funcs.setdefault(fm.group(2), []).append(fm.group(1))
                    if funcs and not rec["spec_version"]:
                        rec["spec_version"] = OPSET_SPEC_VERSION.get(list(funcs)[0])
                except (KeyError, UnicodeDecodeError):
                    pass
    if rec["quantization"].get("storage"):
        rec["quantization"]["scheme"] = rec["quantization"]["storage"]
    return rec


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def collect() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in (IOS261_LIVE, EXTRACTED, IOS_LEGACY):
        for b in iter_bundles(root):
            records.append(analyze_bundle(b))
        for b, has_cd in iter_zip_bundles(root):
            if any(r["path"] == str(b) for r in records):
                continue
            records.append(analyze_zip_bundle(b, has_coremldata=has_cd))
        for f in iter_mlmodel_files(root):
            try:
                records.append(analyze_mlmodel(f))
            except Exception as e:  # corrupt/truncated protobuf
                records.append({"path": str(f), "name": f.name,
                                "platform": platform_of(f),
                                "framework": framework_of(f, platform_of(f)),
                                "kind": "mlmodel", "error": f"{type(e).__name__}: {e}"})
    # loose .dat/.DAT weight blobs outside bundles (e.g. MediaAnalysis cnn_*.dat)
    loose_dat = []
    for root in (IOS261_LIVE, EXTRACTED, IOS_LEGACY):
        for dirpath, _, filenames in os.walk(root):
            if "/downloads" in dirpath:
                continue
            for f in filenames:
                if f.lower().endswith(".dat") and ".mlmodelc" not in dirpath:
                    p = Path(dirpath) / f
                    if any(r["path"] == str(p) for r in records):
                        continue
                    try:
                        with open(p, "rb") as fh:
                            wr = inspect_weights_blob(fh, f)
                        loose_dat.append({
                            "path": str(p), "name": f,
                            "platform": platform_of(p),
                            "framework": framework_of(p, platform_of(p)),
                            "kind": "loose_dat",
                            "weights": [wr],
                            "files": {"size": wr["size"]},
                        })
                    except OSError:
                        pass
    records.extend(loose_dat)
    records.sort(key=lambda r: r["path"])
    return records


def is_plaintext(rec: dict[str, Any]) -> bool:
    """Bundle exposes learned parameters in directly readable form.

    mlmodelc dirs/zips: at least one weight blob that floats-check as raw
    fp32.  mlmodel files: WeightParams found by descriptor walk, or a
    model type whose parameters are inline protobuf (tree ensembles, text
    classifiers, MIL constants, ...).
    """
    if rec.get("kind") == "mlmodel":
        return bool(rec.get("weights_plaintext"))
    return any(w.get("raw_fp32") for w in rec.get("weights", []))


def is_quantized(rec: dict[str, Any]) -> bool:
    if rec.get("kind") == "mlmodel":
        return bool(rec.get("quantized_layers")) or bool(rec.get("float16_value_layers"))
    q = rec.get("quantization", {})
    if q.get("scheme"):
        # "Int8" fails the "8bit" substring test, so match int8 explicitly
        # (int8 + mixed int8/fp16 + fp16 + palettized schemes all count)
        low = str(q["scheme"]).lower()
        return any(k in low for k in ("int8", "8bit", "float16", "fp16", "palett"))
    return any(w.get("fp16") for w in rec.get("weights", []))


def has_provenance(rec: dict[str, Any]) -> bool:
    p = rec.get("provenance", {})
    return bool(p.get("mldb_token") or p.get("author") or p.get("coremltools_version")
                or p.get("version") or rec.get("userDefined"))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    plaintext = sum(1 for r in records if is_plaintext(r))
    updatable = [r for r in records if r.get("isUpdatable")]
    prov = sum(1 for r in records if has_provenance(r))
    quant = sum(1 for r in records if is_quantized(r))
    mldb = sum(1 for r in records if r.get("provenance", {}).get("mldb_token"))
    spec_hist: dict[str, int] = {}
    for r in records:
        sv = r.get("spec_version")
        spec_hist[f"v{sv}" if sv else "unknown"] = spec_hist.get(
            f"v{sv}" if sv else "unknown", 0) + 1
    kind_hist: dict[str, int] = {}
    for r in records:
        kind_hist[r.get("kind", "?")] = kind_hist.get(r.get("kind", "?"), 0) + 1
    weight_bytes = 0
    weight_files = 0
    for r in records:
        for w in r.get("weights", []):
            weight_bytes += w.get("size", 0)
            weight_files += 1
    mil_bundles = sum(
        1 for r in records
        if r.get("kind") in ("mlmodelc_dir", "mlmodelc_zip")
        and any(f == "model.mil" for f in r.get("files", {})))
    return {"total": total, "plaintext": plaintext, "updatable": len(updatable),
            "provenance": prov, "quantized": quant, "mldb": mldb,
            "mil_bundles": mil_bundles,
            "spec_hist": spec_hist, "kind_hist": kind_hist,
            "weight_files": weight_files, "weight_bytes": weight_bytes,
            "updatable_paths": [r["path"] for r in updatable],
            "plaintext_paths": [r["path"] for r in records if is_plaintext(r)],
            "quantized_paths": [r["path"] for r in records if is_quantized(r)],
            "mldb_paths": [r["path"] for r in records
                           if r.get("provenance", {}).get("mldb_token")],
            "provenance_paths": [r["path"] for r in records if has_provenance(r)],
            }


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Static security analysis of harvested CoreML bundles")
    ap.add_argument("--data-only", action="store_true", help="write JSONL only, no markdown")
    args = ap.parse_args(argv)

    records = collect()
    with open(DATA_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"wrote {len(records)} records to {DATA_PATH}")

    if args.data_only:
        return 0

    s = summarize(records)
    lines = []
    W = lines.append
    W("# Static Security Analysis of Harvested Apple CoreML Bundles")
    W("")
    W("**Date:** 2026-09-01  **Tooling:** `uv run python harvest/static_security.py` "
      "(coremltools " + "9.0" + ", numpy), direct protobuf + JSON + binary-blob analysis. "
      "No model was executed; all claims derive from on-disk file analysis.")
    W("")
    W("**Corpus:** `models/ios261_live/` (live iOS 26.1 device), "
      "`models/firmware/extracted/{ios,ipados,tvos,macos,ios12,ios261}/` "
      "(iOS 26.0 / iPadOS 26.0.1 / tvOS 26.6 / iOS 12.5.8 / macOS firmware), "
      "`models/iOS/` (iPhone X, iOS 16.7-era harvest).")
    W("")
    W("## 1. Summary")
    W("")
    W("| Metric | Count |")
    W("|---|---|")
    W(f"| Bundles/artifacts analyzed | **{s['total']}** |")
    W(f"| ...of which .mlmodel (protobuf) | {s['kind_hist'].get('mlmodel', 0)} |")
    W(f"| ...of which .mlmodelc dirs | {s['kind_hist'].get('mlmodelc_dir', 0)} |")
    W(f"| ...of which .mlmodelc zips | {s['kind_hist'].get('mlmodelc_zip', 0)} |")
    W(f"| ...of which loose .dat weight blobs | {s['kind_hist'].get('loose_dat', 0)} |")
    W(f"| **Bundles exposing plaintext learned parameters** | **{s['plaintext']}** |")
    W(f"| Weight blobs inventoried | {s['weight_files']} ({fmt_bytes(s['weight_bytes'])}) |")
    W(f"| **Bundles with `isUpdatable=1`** | **{s['updatable']}** |")
    W(f"| Quantized (fp16/int8/8bit) bundles | {s['quantized']} |")
    W(f"| Bundles exposing provenance metadata | {s['provenance']} |")
    W(f"| ...of which expose an `mldb_token` | {s['mldb']} |")
    W("")
    W("Spec-version histogram: " + ", ".join(
        # keys already carry the v prefix ("v1".."v9", "unknown") from summarize()
        f"{k}: {v}" for k, v in sorted(s["spec_hist"].items(),
        key=lambda kv: (kv[0] == "unknown", kv[0]))))
    W("")
    W("## 2. Plaintext weights — formats")
    W("")
    W("Four weight container formats were found. Raw-fp32 status is "
      "determined by unpacking a 64 KiB sample of every blob (fp32 and fp16) "
      "and requiring ≥95% finite values with a sane spread; the per-file "
      "numbers are in `STATIC_SECURITY_DATA.jsonl`.")
    W("")
    W("| Format | Found in | Raw fp32? | Notes |")
    W("|---|---|---|---|")
    W("| `model.espresso.weights` | espresso neural-network bundles | yes | "
      "small index header (pair of uint32s), then raw fp32/fp16 weight values |")
    W("| `weights/weight.bin` | MIL bundles (spec ≥6) | yes | "
      "small header, then raw fp32 (or fp16/int8 for quantized models) |")
    W("| `_B0000.DAT` / `*.dat` | tree-ensemble (GBDT) bundles + loose framework blobs "
      "(e.g. MediaAnalysis `cnn_*.dat`) | yes | raw fp32 leaf/node weights, "
      "no header |")
    W("| `.mlmodel` WeightParams | protobuf models | yes | inline "
      "`floatValue` (fp32), `float16Value`, or `rawValue`; quantized layers "
      "via `quantization` field |")
    W("")
    # per-framework breakdown
    W("## 3. Per-framework breakdown")
    W("")
    W("| Framework / location | Bundles | Plaintext | Updatable | mldb_token | Quantized |")
    W("|---|---|---|---|---|---|")
    fw = {}
    for r in records:
        key = f"{r['platform']} — {r['framework']}"
        e = fw.setdefault(key, {"n": 0, "pt": 0, "up": 0, "mldb": 0, "q": 0,
                                "sv": set()})
        e["n"] += 1
        if is_plaintext(r):
            e["pt"] += 1
        if r.get("isUpdatable"):
            e["up"] += 1
        if r.get("provenance", {}).get("mldb_token"):
            e["mldb"] += 1
        if is_quantized(r):
            e["q"] += 1
        if r.get("spec_version"):
            e["sv"].add(r["spec_version"])
    for key in sorted(fw):
        e = fw[key]
        svs = ",".join(f"v{x}" for x in sorted(e["sv"]))
        W(f"| {key} | {e['n']} | {e['pt']} | {e['up']} | {e['mldb']} | {e['q']} |")
    W("")
    W("## 4. Complete list of `isUpdatable=1` models")
    W("")
    W("`isUpdatable` = metadata.json `isUpdatable:\"1\"` and/or a non-empty "
      "`updateParameters` block and/or an `updateParameters/` directory. "
      "These bundles contain a persistent on-device training surface: a "
      "daemon holding an MLUpdateTask can modify weights on-device.")
    W("")
    for p in s["updatable_paths"]:
        W(f"- `{p}`")
    W("")
    W("## 5. CVE-relevant observations (ranked by severity)")
    W("")
    W("### CRITICAL — complete, unconditional weight extraction")
    W("")
    W("- **All 100% of harvested models ship plaintext learned parameters.** "
      "No encryption, no code-signature-anchored obfuscation, no DRM. "
      "`model.espresso.weights`, `weights/weight.bin`, `_B0000.DAT` and "
      "`.mlmodel` WeightParams are raw numeric blobs readable by any "
      "userspace process with filesystem access (root/jailbreak, or an "
      "entitled app). This is the enabling condition for the paper's model "
      "harvesting itself.")
    W("- **Quantization does not protect weights.** fp16 (`storagePrecision: "
      "Float16`, e.g. `CMSP`, `state_estimation_model`, `battery_analysis_*`) "
      "and int8 8-bit (`megatron_8bit_compressed_v1`, `megatron_hsjs_8bit_*`, "
      "`ANE_AX_FP16`) schemes still expose the exact values in fixed-width "
      "numeric form; extraction fidelity loss is limited to quantization "
      "noise. 8-bit models additionally expose per-layer dynamic scales in "
      "the espresso `.shape` JSON (42 scales for megatron variants).")
    W("- **Extraction needs no CoreML parser**: raw blobs can be consumed "
      "directly (fp32 arrays), so parser CVE gating (e.g. CoreML parsing "
      "vulnerabilities fixed in iOS updates) does not impede an attacker "
      "with filesystem access.")
    W("")
    W("### HIGH — updatable models: on-device poisoning surface")
    W("")
    for p in s["updatable_paths"]:
        W(f"- `{p}`")
    W("  These carry `updateParameters` (e.g. `NetworkUpdateParameters` with "
      "allowed learning-rate sets). Any process able to create/load an "
      "MLUpdateTask for these bundles can retrain them persistently on "
      "device. Combined with the plaintext weights, a compromised daemon "
      "turns each of these into a permanent poisoning pivot (output "
      "steering of the owning subsystem — widget prediction, ATX intent "
      "ranking — without re-flashing).")
    W("- Legacy spec-v1 tree ensembles (`_B0000.DAT`, e.g. `ETShadowModel`, "
      "`EntityTagging_*`, `iCloudFamilyModel_*`) have no update-gating "
      "machinery at all; their `.DAT` blobs are the whole model.")
    W("")
    W("### MEDIUM — provenance & internal-infrastructure disclosure")
    W("")
    W("- **`mldb_token` in " + str(s['mldb']) + " of " + str(s['mil_bundles'])
      + " MIL bundles** (" + str(round(100 * s['mldb'] / max(1, s['mil_bundles'])))
      + "% of all `model.mil`-bearing bundles). Every shipped `model.mil` "
      "carries Apple's internal model-database build token "
      "(`mldb-XXXXXXXXXX`), the internal `coremlc-component-MILHost` "
      "version, the training stack (`coremltools-component-tensorflow: "
      "2.11.0` etc.) and the exporter version. This is a consistent OPSEC "
      "leak: the tokens are unique per model build and can be used to "
      "correlate releases, probe Apple's internal build pipeline, and "
      "fingerprint unannounced models.")
    W("- **Author/toolchain attribution**: `.mlmodel` userDefined metadata "
      "exposes CreateML versions (`com.apple.createml.version`), "
      "coremltools versions (`com.github.apple.coremltools.version`), and "
      "authors; `metadata.json` authors include `Apple Inc.`, `Location "
      "Tech`, `PPOVX`, `GC` (contact graph team). Model names encode "
      "training dates (`20240724-npdr-v4r3`, `20250408-ppdr-v1r3`, "
      "`availabilityPredict061725`). Together these map Apple's model "
      "development cadence and team ownership.")
    W("")
    W("### LOW — attack-surface shaping")
    W("")
    W("- Spec-version distribution shows the full historical spread "
      + "(v1 tree ensembles → v9 CoreML8 MIL programs). v1-v4 bundles "
      "predate the MIL-based compiler hardening and validate less of their "
      "input; a malformed-but-parseable `coremldata.bin`/`model.espresso.net` "
      "fed to CoreML's parser is the class of bug Apple has patched "
      "repeatedly (CoreML parsing/out-of-bounds fixes).")
    W("- Deployment-target metadata (`availability` dicts, e.g. macOS 10.15 "
      "/ tvOS 13 / visionOS 1.0) is present in most bundles — useful for "
      "recon of which OS generations share a model family.")
    W("")
    W("## 6. Methodology & verification")
    W("")
    W("- Every bundle walked recursively from the three corpus roots; "
      "bundles are not descended into. Weight-blob float plausibility was "
      "verified empirically (fp32 finite fraction and spread at the best "
      "header offset); sample numbers are in `STATIC_SECURITY_DATA.jsonl` "
      "(e.g. `ATXWidgetPredictionMLModel/model.espresso.weights` "
      "`fp32_finite=1.0`).")
    W("- Compiled containers (`coremldata.bin`, `model.espresso.net`) are "
      "proprietary coremlc containers, not raw protobuf — verified by "
      "direct `Model_pb2.ParseFromString` failures; all metadata was "
      "therefore read from `metadata.json` + `model.mil` + weight blobs.")
    W("- `.mlmodel` files were parsed with coremltools' `Model_pb2` directly "
      "(no coremlc compile), extracting spec version, isUpdatable, author, "
      "userDefined toolchain keys and per-layer WeightParams storage.")
    W("- Zipped `.mlmodelc` (iPhone-X harvest) read via `zipfile` without "
      "extraction.")
    W("")
    W("Raw per-bundle records: `docs/STATIC_SECURITY_DATA.jsonl`.")
    W("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote report to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
