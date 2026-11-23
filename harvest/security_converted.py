"""
Adversarial robustness evaluation for the newly-converted .mlmodelc bundles
(155 models: 73 from models/ios261_live, 82 from models/firmware/extracted)
plus the loadable text/classifier models shipped alongside them
(SmsFilter INEN/BRPT, FinHealth intent_classification + ghent family,
MCCKitCategorization BlackPearl family).

Attack batteries mirror harvest/security_exhaustive.py:

  Cat R  Regressors (scalar outputs)   -- Gaussian noise sig=0.01/0.1/1.0,
         feature dropout 10/30/50%, all-zero/all-one, scale x0.5/x2/x10,
         extreme +-50.  Metrics: abs + relative prediction shift.
  Cat C  Classifiers (class-shaped outputs: last dim in [2,64], <=256 elems,
         or classProbability dict / classLabel) -- same feature attacks.
         Metrics: argmax flip rate + logits L2.
  Cat E  Embeddings / neural nets (vector/tensor outputs) -- tensor attacks:
         noise, zeros, ones, outliers, scale.  Metric: output L2.
  Cat S  Special (multi-input / LSTM-state models) -- tensor battery on the
         primary float input; state inputs kept at clean values.
         Metrics: output L2 + argmax flip where class-shaped.
  Cat T  Text models (string input) -- the 14 text-evasion generator families
         from harvest.security (8 direct techniques + 6 multi-variant
         generators; 32 distinct attack types: ZWSP, homoglyphs, whitespace,
         case, punctuation, truncation, language mixing) over the 20-message
         SMS benchmark.
  Cat A  Audio models (SoundAnalysis encoders) -- noise injection on the
         audio feature tensor.

Every evaluation runs a live predict(); failures are recorded per-evaluation.
The worker runs one model per subprocess so a SIGSEGV in the CoreML compiler
cannot kill the whole run (same rationale as harvest/verify_bundles.py).

Records are appended to datasets/EXHAUSTIVE_SECURITY_DATA.jsonl (same schema
as the original 47,371-record corpus) and aggregated into
docs/ADVERSARIAL_CONVERTED_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import coremltools as ct

from harvest.eval import SMS_BENCHMARK
from harvest.security import (
    EVASION_TECHNIQUES,
    EVASION_VARIANT_GENERATORS,
    _adversarial_perturbation,
)

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT_ROOT / "datasets"
DATA_PATH = DATASETS / "EXHAUSTIVE_SECURITY_DATA.jsonl"
REPORT_PATH = PROJECT_ROOT / "docs" / "ADVERSARIAL_CONVERTED_REPORT.md"

CONV_LIVE_DIR = Path("/tmp/adv_conv_live")
CONV_FW_DIR = Path("/tmp/adv_conv_fw")
RESULT_DIR = Path("/tmp/adv_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model inventory
# ---------------------------------------------------------------------------

# loadable .mlmodel/.mlpackage models shipped in the live tree (not conversions)
EXTRA_MODELS: list[dict[str, Any]] = [
    {
        "name": "intent_classification",
        "path": str(PROJECT_ROOT / "models/ios261_live/System/Library/PrivateFrameworks/"
                    "FinHealthCore.framework/ModelResources/assets_WPC_ANALYTICS_OFFLINE_LAB/"
                    "intent_classification.mlmodel"),
        "platform": "ios261_live",
    },
    {
        "name": "ghent_classifier",
        "path": str(PROJECT_ROOT / "models/ios261_live/System/Library/PrivateFrameworks/"
                    "FinHealthCore.framework/ModelResources/assets_WPC_ANALYTICS_OFFLINE_LAB/"
                    "ghent_classifier/ghent_classifier.mlpackage"),
        "platform": "ios261_live",
    },
    {
        "name": "test_ghent_classifier",
        "path": str(PROJECT_ROOT / "models/ios261_live/System/Library/PrivateFrameworks/"
                    "FinHealthCore.framework/ModelResources/test_ghent_classifier.mlmodel"),
        "platform": "ios261_live",
    },
    {
        "name": "BlackPearl",
        "path": str(PROJECT_ROOT / "models/ios261_live/System/Library/PrivateFrameworks/"
                    "MCCKitCategorization.framework/BlackPearlAssets/assets_BLACKPEARL_SPARROW/"
                    "BlackPearlModelPackage/BlackPearl.mlpackage"),
        "platform": "ios261_live",
    },
    {
        "name": "BlackPearlTS",
        "path": str(PROJECT_ROOT / "models/ios261_live/System/Library/PrivateFrameworks/"
                    "MCCKitCategorization.framework/BlackPearlAssets/assets_BLACKPEARL_SPARROW/"
                    "BlackPearlTSModelPackage/BlackPearlTS.mlpackage"),
        "platform": "ios261_live",
    },
    {
        "name": "BlackPearlSender",
        "path": str(PROJECT_ROOT / "models/ios261_live/System/Library/PrivateFrameworks/"
                    "MCCKitCategorization.framework/BlackPearlAssets/assets_BLACKPEARL_SPARROW/"
                    "BlackPearlSenderModelPackage/BlackPearlSender.mlpackage"),
        "platform": "ios261_live",
    },
]

# string-input text classifiers found in the live tree
TEXT_MODELS: list[dict[str, Any]] = [
    {
        "name": "mainModel_INEN",
        "path": str(PROJECT_ROOT / "models/ios261_live/System/Library/SmsFilter/"
                    "assets_SMS_FILTER_INEN_CLASSIFICATION/mainModel_INEN.mlmodel"),
        "platform": "ios261_live",
    },
    {
        "name": "subClassModel_INEN",
        "path": str(PROJECT_ROOT / "models/ios261_live/System/Library/SmsFilter/"
                    "assets_SMS_FILTER_INEN_CLASSIFICATION/subClassModel_INEN.mlmodel"),
        "platform": "ios261_live",
    },
    {
        "name": "mainModel_BRPT",
        "path": str(PROJECT_ROOT / "models/ios261_live/System/Library/SmsFilter/"
                    "assets_SMS_FILTER_BRPT_CLASSIFICATION/mainModel_BRPT.mlmodel"),
        "platform": "ios261_live",
    },
    {
        "name": "basicModel_BRPT",
        "path": str(PROJECT_ROOT / "models/ios261_live/System/Library/SmsFilter/"
                    "assets_SMS_FILTER_BRPT_CLASSIFICATION/basicModel_BRPT.mlmodel"),
        "platform": "ios261_live",
    },
]


def _short_name(pkg_name: str) -> str:
    """'System__Library__...__ATXAppPredictionMLModel.mlmodelc.mlpackage' -> 'ATXAppPredictionMLModel'."""
    n = pkg_name
    for marker in ("__", "_Data_", ".mlmodelc", ".mlpackage"):
        n = n.replace(marker, "|")
    tail = [p for p in n.split("|") if p]
    if tail:
        cand = tail[-1].replace(".mlmodelc", "")
        if cand:
            return cand[:70]
    return pkg_name[:70]


def discover_converted() -> list[dict[str, Any]]:
    """All converted packages with platform + bundle name."""
    out = []
    mapping = json.load(open("/tmp/adv_convert_results.json"))  # bundle -> {out, status}
    bundle_of = {v["out"]: b for b, v in mapping.items()}
    for d, plat_prefix in ((CONV_LIVE_DIR, "ios261_live"), (CONV_FW_DIR, "fw")):
        for pkg in sorted(d.glob("*.mlpackage")):
            bundle = bundle_of.get(str(pkg), "")
            plat = plat_prefix
            if plat_prefix == "fw":
                plat = pkg.name.split("__")[0] if "__" in pkg.name else "fw"
            out.append({
                "name": _short_name(pkg.name),
                "path": str(pkg),
                "platform": plat,
                "bundle": bundle,
            })
    return out


def all_models() -> list[dict[str, Any]]:
    return discover_converted() + EXTRA_MODELS + TEXT_MODELS


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------


def _ma_shape(fd) -> list[int]:
    ma = fd.type.multiArrayType
    shape = [int(d) if d and d > 0 else 1 for d in ma.shape]
    if not shape and ma.shapeRange.sizeRanges:
        shape = [int(r.lowerBound) if r.lowerBound > 0 else 1 for r in ma.shapeRange.sizeRanges]
    return shape


_DESCR_DTYPE_TO_NP = {
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16: np.float16,
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32: np.float32,
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE: np.float64,
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32: np.int32,
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT8: np.int8,
}


def _input_contract(spec) -> list[dict[str, Any]]:
    """(name, kind, shape, dtype) per input; kind in multiArray/scalar/str/other."""
    out = []
    for i in spec.description.input:
        t = i.type.WhichOneof("Type")
        if t == "multiArrayType":
            dt = _DESCR_DTYPE_TO_NP.get(i.type.multiArrayType.dataType, np.float32)
            out.append({"name": i.name, "kind": "array", "shape": _ma_shape(i), "dtype": dt})
        elif t == "doubleType":
            out.append({"name": i.name, "kind": "scalar", "shape": [], "dtype": np.float64})
        elif t == "int64Type":
            out.append({"name": i.name, "kind": "scalar", "shape": [], "dtype": np.int64})
        elif t == "stringType":
            out.append({"name": i.name, "kind": "str", "shape": [], "dtype": None})
        else:
            out.append({"name": i.name, "kind": "other", "shape": [], "dtype": None})
    return out


def _output_info(spec) -> list[dict[str, Any]]:
    out = []
    for o in spec.description.output:
        t = o.type.WhichOneof("Type")
        if t == "multiArrayType":
            out.append({"name": o.name, "kind": "array", "shape": _ma_shape(o)})
        elif t == "dictionaryType":
            out.append({"name": o.name, "kind": "dict", "shape": []})
        elif t == "stringType":
            out.append({"name": o.name, "kind": "str", "shape": []})
        elif t == "int64Type":
            out.append({"name": o.name, "kind": "scalar", "shape": []})
        else:
            out.append({"name": o.name, "kind": t or "?", "shape": []})
    return out


def _n_elems(shape: list[int]) -> int:
    return int(np.prod(shape)) if shape else 1


def categorize(spec) -> str:
    """Text / classifier / regressor / embedding / special / audio."""
    ins = _input_contract(spec)
    if any(c["kind"] == "str" for c in ins):
        return "text"
    outs = _output_info(spec)
    if any(o["kind"] in ("dict", "str") for o in outs):
        return "classifier"
    lnames = {o["name"].lower() for o in outs}
    if lnames & {"classlabel", "classprobability", "target", "label", "prob"}:
        return "classifier"
    if len(ins) > 1:
        return "special"
    arr_outs = [o for o in outs if o["kind"] == "array"]
    if not arr_outs:
        return "regressor" if outs else "special"
    if all(_n_elems(o["shape"]) <= 1 for o in arr_outs):
        return "regressor"
    if any(2 <= (o["shape"][-1] if o["shape"] else 0) <= 64 and _n_elems(o["shape"]) <= 256
           for o in arr_outs):
        return "classifier"
    return "embedding"


def _class_output(outs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The output used for argmax classification (first class-shaped one)."""
    arr = [o for o in outs if o["kind"] == "array"]
    for o in arr:
        if 2 <= (o["shape"][-1] if o["shape"] else 0) <= 64 and _n_elems(o["shape"]) <= 256:
            return o
    return None


def build_base_feed(model, contracts: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    """Base input honoring shapes/dtypes; integer inputs filled with 1s
    (data-dependent-shape models fail on random ints -- see CONVERSION_NOTES #9).

    Float values come from default_rng(0) -- the same deterministic values
    harvest/mlmodelc_loader.make_test_inputs uses.  THIS MATTERS: some models
    (fastspeech2_encoder) SIGSEGV the CoreML compiler on other random draws,
    so only the verified value distribution is safe as a base.
    """
    brng = np.random.default_rng(0)
    feed = {}
    for c in contracts:
        if c["kind"] == "array":
            if np.issubdtype(c["dtype"], np.integer):
                feed[c["name"]] = np.ones(c["shape"], dtype=c["dtype"])
            else:
                feed[c["name"]] = brng.normal(0.0, 1.0, c["shape"]).astype(c["dtype"])
        elif c["kind"] == "scalar":
            feed[c["name"]] = float(brng.normal(0.0, 1.0)) if c["dtype"] == np.float64 else 0
        elif c["kind"] == "str":
            feed[c["name"]] = ""
    return feed


def _vec_of(v: Any) -> np.ndarray:
    if isinstance(v, dict):  # classProbability dict
        return np.asarray(list(v.values()), dtype=np.float64)
    return np.asarray(v, dtype=np.float64).ravel()


def _argmax_label(v: Any) -> str:
    arr = _vec_of(v)
    return str(int(np.argmax(arr))) if arr.size > 1 else str(arr.ravel()[0])


def _classify_pred(pred: dict[str, Any], outs: list[dict[str, Any]],
                   cl_out: dict[str, Any] | None) -> str:
    """A label string from a prediction, whichever output carries class info."""
    if cl_out and cl_out["name"] in pred:
        return _argmax_label(pred[cl_out["name"]])
    for o in outs:
        if o["kind"] == "str" and o["name"] in pred:
            return str(pred[o["name"]])
    for o in outs:
        if o["kind"] == "dict" and o["name"] in pred:
            d = pred[o["name"]]
            if isinstance(d, dict) and d:
                return str(max(d, key=lambda k: float(d[k])))
    for o in outs:
        if o["kind"] == "array" and o["name"] in pred and _n_elems(o["shape"]) > 1:
            return _argmax_label(pred[o["name"]])
    return str(pred)


# ---------------------------------------------------------------------------
# Attack batteries (per-model, subprocess worker)
# ---------------------------------------------------------------------------


def _timed(model, feed):
    t0 = time.perf_counter()
    out = model.predict(feed)
    return out, (time.perf_counter() - t0) * 1000.0


def run_battery(inst: dict[str, Any], cat: str) -> list[dict[str, Any]]:
    """Full battery for one model; returns raw record dicts."""
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(7)
    try:
        model = ct.models.MLModel(inst["path"])
    except Exception as e:
        return [{"attack_type": "__load__", "error": f"{type(e).__name__}: {str(e)[:200]}"}]
    spec = model.get_spec()
    contracts = _input_contract(spec)
    outs = _output_info(spec)
    cl_out = _class_output(outs)

    def rec(**kw):
        kw.setdefault("flipped", None)
        kw.setdefault("output_distance", None)
        kw.setdefault("relative_shift", None)
        kw.setdefault("confidence_shift", None)
        kw.setdefault("detail", None)
        kw.setdefault("error", "")
        records.append(kw)

    if cat == "text":
        run_text(model, inst, spec, rec)
        return records

    # ---- feature-space batteries (classifier / regressor) ---------------
    if cat in ("classifier", "regressor"):
        # population stats per array input (deterministic synthetic population)
        in_arrays = [c for c in contracts if c["kind"] == "array"]
        pop = []
        for _ in range(200):
            f = {}
            for c in in_arrays:
                f[c["name"]] = np.clip(rng.normal(0, 1, c["shape"]), -3, 3).astype(c["dtype"])
            pop.append(f)
        means = {c["name"]: np.mean(np.stack([f[c["name"]] for f in pop]), axis=0)
                 for c in in_arrays}
        stds = {c["name"]: np.std(np.stack([f[c["name"]] for f in pop]), axis=0)
                for c in in_arrays}
        for c in in_arrays:
            stds[c["name"]] = np.where(stds[c["name"]] < 1e-9, 1.0, stds[c["name"]])

        n_base = 40
        n_ok = 0
        for _ in range(n_base):
            feed = {}
            for c in contracts:
                if c["kind"] == "array":
                    feed[c["name"]] = np.clip(
                        rng.normal(0, 1, c["shape"]) * stds[c["name"]] + means[c["name"]],
                        -1e3, 1e4).astype(c["dtype"])
                elif c["kind"] == "scalar":
                    feed[c["name"]] = float(rng.normal(0, 1.0))
                else:
                    feed[c["name"]] = ""
            try:
                clean, clean_ms = _timed(model, feed)
            except Exception as e:
                rec(attack_type="clean", error=f"{type(e).__name__}: {str(e)[:120]}")
                continue
            n_ok += 1
            if cat == "classifier":
                clean_pred = _classify_pred(clean, outs, cl_out)
                clean_logits = _vec_of(clean[cl_out["name"]]) if cl_out and cl_out["name"] in clean \
                    else _class_vec(clean, outs)
            else:
                clean_pred = _first_numeric(clean)
                clean_logits = None
            for atk, sev, perturbed in _feature_variants(feed, stds, rng, contracts):
                try:
                    out, ms = _timed(model, perturbed)
                    if cat == "classifier":
                        atk_pred = _classify_pred(out, outs, cl_out)
                        if cl_out and cl_out["name"] in out:
                            dist = float(np.linalg.norm(clean_logits - _vec_of(out[cl_out["name"]])))
                        else:
                            v = _class_vec(out, outs)
                            dist = float(np.linalg.norm(clean_logits - v)) if v is not None else None
                        rec(attack_type=atk, severity=sev, original_output=clean_pred,
                            attacked_output=atk_pred, flipped=(atk_pred != clean_pred),
                            output_distance=dist, latency_ms=ms, baseline_latency_ms=clean_ms)
                    else:
                        num = _first_numeric(out)
                        dist = abs(num - clean_pred) if num is not None and clean_pred is not None else None
                        rel = dist / abs(clean_pred) if dist is not None and clean_pred not in (None, 0.0) else None
                        rec(attack_type=atk, severity=sev, original_output=str(clean_pred),
                            attacked_output=str(num) if num is not None else "",
                            output_distance=dist, relative_shift=rel,
                            latency_ms=ms, baseline_latency_ms=clean_ms)
                except Exception as e:
                    rec(attack_type=atk, severity=sev, original_output=str(clean_pred),
                        error=f"{type(e).__name__}: {str(e)[:120]}")
        if n_ok == 0:
            rec(attack_type="__all_failed__", error="no clean predictions succeeded")
        return records

    # ---- embedding / special / audio: tensor battery ---------------------
    if cat in ("embedding", "special", "audio"):
        arr_ins = [c for c in contracts if c["kind"] == "array"
                   and not np.issubdtype(c["dtype"], np.integer)]
        if not arr_ins:
            rec(attack_type="__no_array_input__", error="no float array input")
            return records
        primary = max(arr_ins, key=lambda c: _n_elems(c["shape"]))
        feed = build_base_feed(model, contracts, rng)
        try:
            clean, clean_ms = _timed(model, feed)
        except Exception as e:
            rec(attack_type="clean", error=f"{type(e).__name__}: {str(e)[:150]}")
            return records
        clean_pred = _classify_pred(clean, outs, cl_out) if cl_out else None
        base = np.asarray(feed[primary["name"]]).copy()

        # degeneracy probe: second clean draw with a different seed
        try:
            feed2 = build_base_feed(model, contracts, rng)
            out2, _ = _timed(model, feed2)
            rec(attack_type="clean_repeat", original_output="clean_inference",
                output_distance=_out_dist(clean, out2, outs),
                latency_ms=0, baseline_latency_ms=clean_ms)
        except Exception as e:
            rec(attack_type="clean_repeat", error=f"{type(e).__name__}: {str(e)[:120]}")

        # small-input specials: attack ALL float inputs together (the output-
        # driving channel may be a non-primary input, e.g. ASRRankingModel)
        total_float = sum(_n_elems(c["shape"]) for c in arr_ins)
        all_inputs = total_float <= 256

        def _variant_feed(variant_for):
            fa = dict(feed)
            for c in arr_ins:
                fa[c["name"]] = variant_for(np.asarray(feed[c["name"]])).astype(c["dtype"])
            return fa

        variants: list[tuple[str, str, Any]] = []
        for sig in (0.01, 0.1, 1.0):
            variants.append((f"tensor_noise_{sig}", f"sigma={sig}",
                             lambda b, s=sig: b + rng.normal(0, s * max(1.0, float(b.std())), b.shape).astype(np.float32)))
        variants.append(("tensor_zeros", "all-zero", lambda b: np.zeros_like(b)))
        variants.append(("tensor_ones", "all-one", lambda b: np.ones_like(b)))
        variants.append(("tensor_outlier_50", "10% at +50", lambda b: _outlier(b, rng, 50.0)))
        variants.append(("tensor_outlier_m50", "10% at -50", lambda b: _outlier(b, rng, -50.0)))
        for f in (0.5, 2.0, 10.0):
            variants.append((f"tensor_scale_{f}x", f"x{f}", lambda b, f=f: (b * f).astype(np.float32)))
        for frac in (0.1, 0.5):
            variants.append((f"tensor_dropout_{int(frac*100)}pct", f"frac={frac}",
                             lambda b, frac=frac: np.where(rng.random(b.shape) < frac, 0.0, b)))

        for atk, sev, vf in variants:
            if all_inputs:
                feed_a = _variant_feed(vf)
            else:
                feed_a = dict(feed)
                feed_a[primary["name"]] = vf(base).astype(primary["dtype"])
            try:
                out, ms = _timed(model, feed_a)
                dist = _out_dist(clean, out, outs)
                flipped = None
                if cl_out and cl_out["name"] in out:
                    flipped = _classify_pred(out, outs, cl_out) != clean_pred
                rec(attack_type=atk, severity=sev, original_output="clean_inference",
                    attacked_output=f"L2={dist:.4f}", output_distance=dist, flipped=flipped,
                    latency_ms=ms, baseline_latency_ms=clean_ms)
            except Exception as e:
                rec(attack_type=atk, severity=sev, original_output="clean_inference",
                    error=f"{type(e).__name__}: {str(e)[:120]}")
        return records

    rec(attack_type="__unhandled__", error=f"category {cat}")
    return records


def _out_dist(clean: dict[str, Any], attacked: dict[str, Any],
              outs: list[dict[str, Any]]) -> float:
    """L2 distance between two predictions, per output name (output sets can
    legitimately differ in size for data-dependent models like PlanResolution)."""
    total = 0.0
    for o in outs:
        if o["name"] not in clean or o["name"] not in attacked:
            continue
        a = _vec_of(clean[o["name"]])
        b = _vec_of(attacked[o["name"]])
        n = min(a.size, b.size)
        if n > 0:
            total += float(np.sum((a[:n] - b[:n]) ** 2))
    return float(np.sqrt(total))


def _out_vec(out: dict[str, Any], outs: list[dict[str, Any]]) -> np.ndarray:
    vecs = []
    for o in outs:
        if o["name"] in out:
            vecs.append(_vec_of(out[o["name"]]))
    return np.concatenate(vecs) if vecs else np.zeros(1)


def _class_vec(pred: dict[str, Any], outs: list[dict[str, Any]]) -> np.ndarray | None:
    for o in outs:
        if o["kind"] == "dict" and o["name"] in pred and isinstance(pred[o["name"]], dict):
            d = pred[o["name"]]
            return np.asarray([float(d[k]) for k in sorted(d)], dtype=np.float64)
    for o in outs:
        if o["kind"] == "array" and o["name"] in pred and _n_elems(o["shape"]) > 1:
            return _vec_of(pred[o["name"]])
    return None


def run_text(model, inst, spec, rec) -> None:
    """Text attacks over the SMS benchmark (same 14 attack-generator families as the original corpus)."""
    text_input = next(c["name"] for c in _input_contract(spec) if c["kind"] == "str")
    label_out = next((o["name"] for o in _output_info(spec) if o["kind"] == "str"), None)
    if label_out is None:
        rec(attack_type="__no_string_output__", error="no string output")
        return
    for text, _exp in SMS_BENCHMARK:
        try:
            clean, clean_ms = _timed(model, {text_input: text})
        except Exception as e:
            rec(attack_type="clean", error=f"{type(e).__name__}: {str(e)[:120]}")
            continue
        clean_pred = str(clean.get(label_out, ""))
        text_attacks = ([(a, "single", fn) for a, fn in EVASION_TECHNIQUES]
                        + [(a, "multi", fn) for a, fn in EVASION_VARIANT_GENERATORS])
        for attack, kind, gen_fn in text_attacks:
            try:
                variants = gen_fn(text)
                if kind == "single":
                    variants = [variants]
                for variant in variants:
                    try:
                        out, ms = _timed(model, {text_input: variant})
                        atk_pred = str(out.get(label_out, ""))
                        rec(attack_type=attack, original_output=clean_pred,
                            attacked_output=atk_pred, flipped=(atk_pred != clean_pred),
                            latency_ms=ms, baseline_latency_ms=clean_ms,
                            detail={"text_len": len(variant), "variant": variant[:200]})
                    except Exception as e:
                        rec(attack_type=attack, original_output=clean_pred,
                            error=f"{type(e).__name__}: {str(e)[:120]}")
            except Exception as e:
                rec(attack_type=attack, original_output=clean_pred,
                    error=f"generator: {type(e).__name__}: {str(e)[:120]}")
        for sev in (1, 2, 3):
            for atk, variant in _adversarial_perturbation(text, severity=sev).items():
                try:
                    out, ms = _timed(model, {text_input: variant})
                    atk_pred = str(out.get(label_out, ""))
                    rec(attack_type=atk, severity=sev, original_output=clean_pred,
                        attacked_output=atk_pred, flipped=(atk_pred != clean_pred),
                        latency_ms=ms, baseline_latency_ms=clean_ms,
                        detail={"variant": variant[:200]})
                except Exception as e:
                    rec(attack_type=atk, severity=sev, original_output=clean_pred,
                        error=f"{type(e).__name__}: {str(e)[:120]}")


def _feature_variants(feed, stds, rng, contracts):
    """Perturbed feeds for the feature-space battery (per task spec)."""
    for c in contracts:
        if c["kind"] == "array":
            base = np.asarray(feed[c["name"]])
            st = stds.get(c["name"], np.ones_like(base))
            for sig in (0.01, 0.1, 1.0):
                z = base + rng.normal(0, sig, base.shape).astype(base.dtype) * st
                yield (f"gaussian_noise_{sig}", f"sigma={sig}", _feed_with(feed, c["name"], z))
            for frac in (0.10, 0.30, 0.50):
                mask = rng.random(base.shape) < frac
                z = base.copy()
                z[mask] = 0.0
                yield (f"feature_dropout_{int(frac*100)}pct", f"frac={frac}", _feed_with(feed, c["name"], z))
            yield ("feature_all_zero", "all-zero", _feed_with(feed, c["name"], np.zeros_like(base)))
            yield ("feature_all_one", "all-one", _feed_with(feed, c["name"], np.ones_like(base)))
            for f in (0.5, 2.0, 10.0):
                yield (f"feature_scale_{f}x", f"x{f}", _feed_with(feed, c["name"], (base * f).astype(base.dtype)))
            yield ("extreme_plus50", "+50", _feed_with(feed, c["name"], np.full_like(base, 50.0)))
            yield ("extreme_minus50", "-50", _feed_with(feed, c["name"], np.full_like(base, -50.0)))
        elif c["kind"] == "scalar":
            base = float(feed[c["name"]])
            for sig in (0.01, 0.1, 1.0):
                yield (f"gaussian_noise_{sig}", f"sigma={sig}", _feed_with(feed, c["name"], base + rng.normal(0, sig)))
            yield ("feature_all_zero", "all-zero", _feed_with(feed, c["name"], 0.0))
            yield ("feature_all_one", "all-one", _feed_with(feed, c["name"], 1.0))
            for f in (0.5, 2.0, 10.0):
                yield (f"feature_scale_{f}x", f"x{f}", _feed_with(feed, c["name"], base * f))
            yield ("extreme_plus50", "+50", _feed_with(feed, c["name"], 50.0))
            yield ("extreme_minus50", "-50", _feed_with(feed, c["name"], -50.0))


def _outlier(b: np.ndarray, rng: np.random.Generator, val: float) -> np.ndarray:
    flat = b.ravel().copy()
    n = flat.size
    idx = rng.choice(n, size=max(1, n // 10), replace=False)
    flat[idx] = val
    return flat.reshape(b.shape)


def _feed_with(base_feed, name, val):
    f = dict(base_feed)
    f[name] = val
    return f


def _first_numeric(pred: dict[str, Any]) -> float | None:
    for k, v in pred.items():
        if k.lower() in ("classprobability", "classlabel"):
            continue
        if isinstance(v, dict):
            continue
        arr = np.asarray(v)
        if arr.size == 1:
            return float(arr.ravel()[0])
    return None


# ---------------------------------------------------------------------------
# Worker + orchestrator
# ---------------------------------------------------------------------------


def worker(inst: dict[str, Any], out_path: str) -> None:
    """Load, categorize, and run the battery for one model (in this process)."""
    import warnings
    warnings.filterwarnings("ignore")
    try:
        model = ct.models.MLModel(inst["path"])
        spec = model.get_spec()
    except Exception as e:
        records = [{"attack_type": "__load__", "error": f"{type(e).__name__}: {str(e)[:200]}"}]
        meta = {"inst": inst, "category": "load_failed", "model_type": "", "n": 1}
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "records": records}, f, default=str)
        return
    cat = categorize(spec)
    # audio-family naming override (SoundAnalysis encoders take audio features)
    if "SNLanguageAligned" in inst["name"]:
        cat = "audio"
    mt = spec.WhichOneof("Type") or ""
    records = run_battery(inst, cat)
    meta = {"inst": inst, "category": cat, "model_type": mt, "n": len(records)}
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "records": records}, f, default=str)


def _run_one(inst: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    # key on the unique package basename, NOT the short name: many bundles
    # share short names (e.g. 19 U2Head locale variants)
    key = f"{inst['platform']}__{Path(inst['path']).name}"
    out_path = RESULT_DIR / f"{key}.json"
    if out_path.exists():
        data = json.loads(out_path.read_text())
        meta = data.get("meta", {})
        return meta.get("category", ""), meta.get("model_type", ""), data.get("records", [])
    # fresh subprocess: the CoreML compiler can SIGSEGV on some programs
    code = ("import sys; sys.path.insert(0, %r);"
            "from harvest.security_converted import worker;"
            "worker(%r, %r)") % (str(PROJECT_ROOT), inst, str(out_path))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        return "", "", [{"attack_type": "__timeout__", "error": "worker timed out"}]
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()[-200:]
        if proc.returncode < 0:
            msg = f"signal {proc.returncode}"
        return "", "", [{"attack_type": "__crash__", "error": msg}]
    if out_path.exists():
        data = json.loads(out_path.read_text())
        meta = data.get("meta", {})
        return meta.get("category", ""), meta.get("model_type", ""), data.get("records", [])
    return "", "", [{"attack_type": "__no_output__", "error": "worker produced no file"}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="restrict to names containing these")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        rows = load_data()
        print("Report ->", build_report(rows, original_stats(rows)))
        return

    insts = all_models()
    print(f"{len(insts)} models", flush=True)

    for inst in insts:
        if args.only and not any(k.lower() in inst["name"].lower() for k in args.only):
            continue
        t0 = time.perf_counter()
        cat, mt, records = _run_one(inst)
        full = []
        for r in records:
            full.append({**{"platform": inst["platform"], "path": inst["path"],
                            "name": inst["name"], "category": cat,
                            "model_type": mt}, **r})
        write_records(full)
        errs = sum(1 for r in full if r.get("error"))
        print(f"OK {inst['name']} [{inst['platform']}] ({cat}) {len(full)} evals "
              f"{errs} err in {time.perf_counter()-t0:.1f}s", flush=True)

    new_rows = load_data()
    orig = original_stats(new_rows)  # corpus stats EXCLUDING this run's records
    print("\nAppending to", DATA_PATH)
    append_to_corpus()
    print("Report ->", build_report(new_rows, orig))


def load_data() -> list[dict[str, Any]]:
    """All NEW records (from this run's result files)."""
    rows = []
    for f in sorted(RESULT_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        inst = d.get("meta", {}).get("inst", {})
        cat = d.get("meta", {}).get("category", "")
        mt = d.get("meta", {}).get("model_type", "")
        for r in d.get("records", []):
            rows.append({**{"platform": inst.get("platform", ""), "path": inst.get("path", ""),
                            "name": inst.get("name", ""), "category": cat,
                            "model_type": mt}, **r})
    # keep the report consistent with the corpus: same exact-duplicate
    # rule as append_to_corpus (json.dumps(r, default=str) == the JSONL line)
    seen: set[str] = set()
    return [r for r in rows
            if not (k := json.dumps(r, default=str)) in seen and not seen.add(k)]


def write_records(rows: list[dict[str, Any]]) -> None:
    """Persist new records to the side file until the final append."""
    with open(RESULT_DIR / "all_records.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _dedupe_lines(lines: list[str]) -> list[str]:
    """Drop exact-duplicate lines, keeping first occurrence.

    Deterministic models (e.g. 2-feature tree ensembles like test_ghent_classifier)
    can produce byte-identical records for DIFFERENT evaluations: perturbing
    input_1 vs input_2 yields the same class, the same L2 distance, and the same
    deterministic cycle count (latency identical to 1e-18 ms), so the JSON lines
    collide.  Each is a real predict() measurement, but exact duplicates in the
    corpus would double-count feeds in downstream per-attack stats."""
    uniq: list[str] = []
    seen: set[str] = set()
    for l in lines:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    return uniq


def append_to_corpus() -> None:
    src = RESULT_DIR / "all_records.jsonl"
    if not src.exists():
        return
    with open(src) as f:
        new = [l for l in f if l.strip()]
    if not new:
        return
    new = _dedupe_lines(new)
    with open(DATA_PATH, "a") as f:
        f.write("".join(new))
    print(f"appended {len(new)} records to {DATA_PATH}")
    src.rename(src.with_suffix(".jsonl.done"))


def original_stats(new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-model flip statistics from the pre-existing corpus records."""
    new_paths = {r.get("path") for r in new_rows}
    models: dict[str, dict[str, Any]] = {}
    cats: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "flips": 0})
    for line in open(DATA_PATH):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("path") in new_paths:
            continue  # this run's records
        if r.get("flipped") is None or r.get("error"):
            continue
        m = models.setdefault(r.get("name", "?"), {"n": 0, "flips": 0, "atks": {}})
        m["n"] += 1
        m["flips"] += 1 if r["flipped"] else 0
        a = m["atks"].setdefault(r.get("attack_type", "?"), [0, 0])
        a[0] += 1
        a[1] += 1 if r["flipped"] else 0
        c = cats[r.get("category", "?")]
        c["n"] += 1
        c["flips"] += 1 if r["flipped"] else 0
    worst = sorted(((v["flips"] / v["n"], k, v["n"]) for k, v in models.items() if v["n"] >= 40),
                   reverse=True)
    cats_out = {k: {"models": 0, "n": v["n"], "flips": v["flips"],
                    "rate": v["flips"] / v["n"] if v["n"] else None}
                for k, v in cats.items()}
    return {"models": models, "worst": worst, "cats": cats_out}


def _overlap_rate(orig_model: dict[str, Any], new_atks: set[str]) -> tuple[float, int]:
    """Original-corpus flip rate restricted to attack types also used this run."""
    n = fl = 0
    for atk, (an, afl) in orig_model.get("atks", {}).items():
        if atk in new_atks:
            n += an
            fl += afl
    return (fl / n if n else float("nan"), n)


def build_report(rows: list[dict[str, Any]] | None = None,
                 orig: dict[str, Any] | None = None) -> str:
    if rows is None:
        rows = load_data()
    if orig is None:
        orig = original_stats(rows)
    md = _render_report(rows, orig)
    REPORT_PATH.write_text(md)
    return str(REPORT_PATH)


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

CAT_LABELS = {
    "classifier": "Classifiers (class-shaped outputs)",
    "regressor": "Regressors (scalar outputs)",
    "embedding": "Embeddings / neural nets (vector outputs)",
    "special": "Special (multi-input / LSTM-state)",
    "text": "Text classifiers (string input)",
    "audio": "Audio models (SoundAnalysis encoders)",
}


def _agg(rows, path, atk):
    sub = [r for r in rows if r.get("path") == path and r.get("attack_type") == atk
           and not r.get("error")]
    if not sub:
        return None
    flips = [r for r in sub if r.get("flipped") is True]
    flip_rate = len(flips) / len(sub) if sub[0].get("flipped") is not None else None
    dists = [r.get("output_distance") for r in sub if r.get("output_distance") is not None]
    rels = [r.get("relative_shift") for r in sub if r.get("relative_shift") is not None]
    return {
        "attack": atk, "n": len(sub), "n_flips": len(flips), "flip_rate": flip_rate,
        "mean_dist": float(np.mean(dists)) if dists else None,
        "max_dist": float(np.max(dists)) if dists else None,
        "mean_rel": float(np.mean(rels)) if rels else None,
        "max_rel": float(np.max(rels)) if rels else None,
    }


def _name_of(rows, path):
    for r in rows:
        if r.get("path") == path and r.get("name"):
            plat = r.get("platform") or ""
            return f"{r['name']} [{plat}]"
    return Path(path).name


def _render_report(rows: list[dict[str, Any]], orig: dict[str, Any]) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    L: list[str] = [
        "# Adversarial Robustness Evaluation — Newly-Converted CoreML Models",
        "",
        f"**Date:** {ts}  ",
        f"**Models evaluated:** {len({r.get('path') for r in rows})}  ",
        f"**Attack evaluations:** {len(rows)}  ",
        f"**With errors:** {sum(1 for r in rows if r.get('error'))}",
        "",
        "Corpus: the 155 .mlmodelc bundles converted to loadable .mlpackage form by "
        "`harvest/mlmodelc_loader.py` (73 from `models/ios261_live`, 82 from "
        "`models/firmware/extracted`; see `docs/CONVERSION_NOTES.md`) plus the loadable "
        "models shipped alongside them: SmsFilter INEN/BRPT text classifiers, FinHealth "
        "`intent_classification` and the ghent family, MCCKitCategorization BlackPearl family.",
        "",
        "Every number is a live `predict()` call through coremltools 9.  Attack batteries mirror "
        "`harvest/security_exhaustive.py`.  Individual records are appended to "
        "`datasets/EXHAUSTIVE_SECURITY_DATA.jsonl`.",
        "",
    ]
    # -- per-family overview
    L += ["## Overview", ""]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r.get("category", "?")].append(r)
    L += ["| Category | Models | Evaluations | Errors |", "|---|---|---|---|"]
    for cat in ("classifier", "regressor", "embedding", "special", "text", "audio", "load_failed"):
        rr = by_cat.get(cat, [])
        if not rr:
            continue
        mods = len({r.get("path") for r in rr})
        errs = sum(1 for r in rr if r.get("error"))
        label = CAT_LABELS.get(cat, "Load failures")
        L.append(f"| {label} | {mods} | {len(rr)} | {errs} |")
    L.append("")
    # -- worst-case flips per model
    L += ["## Worst-case flips per model (classifier-shaped outputs)", ""]
    worst: list[tuple[float, int, str, str]] = []
    for path in {r.get("path") for r in rows}:
        atks = {r.get("attack_type") for r in rows if r.get("path") == path
                and not r.get("error") and r.get("flipped") is not None}
        for a in sorted(atks):
            agg = _agg(rows, path, a)
            if agg and agg.get("flip_rate") is not None and agg["n"] >= 10:
                worst.append((agg["flip_rate"], agg["n"], path, a))
    worst.sort(key=lambda x: (-x[0], -x[1]))
    L += ["| Flip rate | Model | Attack | N |", "|---|---|---|---|"]
    for fr, n, path, a in worst[:25]:
        L.append(f"| {fr*100:.1f}% | {_name_of(rows, path)} | {a} | {n} |")
    L.append("")
    # -- worst output L2 (embeddings / specials / audio)
    L += ["## Worst output shifts (embeddings / specials / audio, output L2)", ""]
    worstL2: list[tuple[float, str, str]] = []
    for path in {r.get("path") for r in rows}:
        cat = next((r.get("category") for r in rows if r.get("path") == path), "")
        if cat not in ("embedding", "special", "audio"):
            continue
        for a in sorted({r.get("attack_type") for r in rows if r.get("path") == path
                         and not r.get("error") and r.get("attack_type") not in ("clean",)}):
            agg = _agg(rows, path, a)
            if agg and agg.get("max_dist") is not None:
                worstL2.append((agg["max_dist"], path, a))
    worstL2.sort(key=lambda x: -x[0])
    L += ["| Max L2 | Model | Attack |", "|---|---|---|"]
    for md, path, a in worstL2[:20]:
        L.append(f"| {md:.1f} | {_name_of(rows, path)} | {a} |")
    L.append("")
    # -- regressor shifts
    L += ["## Worst prediction shifts (regressors)", ""]
    worstR: list[tuple[float, str, str]] = []
    for path in {r.get("path") for r in rows}:
        cat = next((r.get("category") for r in rows if r.get("path") == path), "")
        if cat != "regressor":
            continue
        for a in sorted({r.get("attack_type") for r in rows if r.get("path") == path
                         and not r.get("error")}):
            agg = _agg(rows, path, a)
            if agg and agg.get("max_dist") is not None:
                worstR.append((agg["max_dist"], path, a))
    worstR.sort(key=lambda x: -x[0])
    L += ["| Max abs shift | Model | Attack | Mean rel shift |", "|---|---|---|---|"]
    for d, path, a in worstR[:20]:
        agg = _agg(rows, path, a)
        rel = f"{agg['mean_rel']*100:.1f}%" if agg and agg.get("mean_rel") is not None else "n/a"
        L.append(f"| {d:.4f} | {_name_of(rows, path)} | {a} | {rel} |")
    L.append("")
    # -- text classifiers detail
    text_rows = [r for r in rows if r.get("category") == "text"]
    if text_rows:
        L += ["## Text classifiers", ""]
        for path in sorted({r.get("path") for r in text_rows}):
            name = _name_of(text_rows, path)
            atks = sorted({r.get("attack_type") for r in text_rows if r.get("path") == path
                           and not r.get("error") and r.get("attack_type") != "clean"})
            L.append(f"### {name}")
            L.append("")
            L += ["| Attack | N | Flips | Flip rate |", "|---|---|---|---|"]
            for a in atks:
                agg = _agg(text_rows, path, a)
                if agg:
                    fr = f"{agg['flip_rate']*100:.1f}%" if agg["flip_rate"] is not None else "n/a"
                    L.append(f"| {a} | {agg['n']} | {agg['n_flips']} | {fr} |")
            L.append("")
    # -- failures
    fails = [r for r in rows if r.get("attack_type") in ("__load__", "__crash__", "__timeout__",
                                                          "__no_output__", "__all_failed__")]
    if fails:
        L += ["## Models that failed to load / crashed during evaluation", ""]
        for f in fails:
            L.append(f"- `{f.get('name')}` [{f.get('platform')}]: {f.get('error', '')[:120]}")
        L.append("")
    # -- notable findings (computed from the new records)
    L.append("## Notable new findings (measured)")
    L.append("")
    # >90% flip models
    high = [(fr, n, path, a) for fr, n, path, a in worst if fr > 0.9]
    if high:
        L.append(f"- **{len(high)} model/attack combinations flip more than 90% of inputs.**")
        for fr, n, path, a in high[:8]:
            L.append(f"  - `{_name_of(rows, path)}` {a}: {fr*100:.1f}% ({n} trials)")
        L.append("")
    # degenerate: output invariant under every attack AND under a different
    # clean draw (clean_repeat probe) -- rules out a non-primary input driving
    # the output (e.g. ASRRankingModel's rankings channel)
    degen: list[str] = []
    for path in {r.get("path") for r in rows}:
        cat = next((r.get("category") for r in rows if r.get("path") == path), "")
        if cat not in ("embedding", "special", "audio"):
            continue
        rr = [r for r in rows if r.get("path") == path and not r.get("error")]
        attack_dists = [r.get("output_distance") for r in rr
                        if r.get("attack_type") != "clean_repeat" and r.get("output_distance") is not None]
        cr = [r.get("output_distance") for r in rr
              if r.get("attack_type") == "clean_repeat" and r.get("output_distance") is not None]
        if attack_dists and cr and max(attack_dists) == 0.0 and cr[0] == 0.0:
            degen.append(_name_of(rows, path))
    if degen:
        L.append(f"- **Output-degenerate models** (identical output under every attack and across "
                 f"independent clean draws, L2 = 0): {', '.join(degen)}.")
        L.append("")
    # regressor relative shifts > 100x (finite only: clean predictions near 0
    # make relative shift explode to inf and are artifacts)
    big = [(r.get("relative_shift"), r.get("name")) for r in rows
           if r.get("relative_shift") is not None and np.isfinite(r.get("relative_shift"))
           and r.get("relative_shift") > 100 and r.get("category") == "regressor"]
    if big:
        per_model: dict[str, list[float]] = {}
        for s, n in big:
            per_model.setdefault(n, []).append(s)
        worst_big = sorted(per_model.items(), key=lambda kv: -max(kv[1]))[:6]
        L.append("- **Regressors with extreme relative shifts (>100×)** under feature attacks: "
                 + ", ".join(f"`{n}` (×{max(ss):.0f}, {len(ss)} attack"
                             f"{'s' if len(ss) > 1 else ''})" for n, ss in worst_big) + ".")
        L.append("")
    # -- comparison vs original corpus
    L.append("## Comparison with the original corpus")
    L.append("")
    L.append("The original corpus (docs/EXHAUSTIVE_SECURITY.md, 47,371 records over 56 models) "
             "covered firmware text classifiers, tree ensembles, and image/audio neural nets. "
             "The converted corpus is structurally different: it is dominated by *stateful RNNs "
             "and multi-stream scoring models* (ATX DuetExpertCenter subscore predictors, "
             "SLOM/wavernn/fastspeech2 TTS, PlanResolutionModel) rather than single-shot "
             "classifiers, so flip rates are not directly comparable; the strongest "
             "per-attack flip rates in each corpus are:")
    L.append("")
    orig_worst = orig.get("worst", [])
    L += ["| Corpus | Model | Best attack flip rate (worst model) |", "|---|---|---|"]
    if worst:
        fr, n, path, a = worst[0]
        L.append(f"| Converted (this run) | {_name_of(rows, path)} | {fr*100:.1f}% (`{a}`, n={n}) |")
    if orig_worst:
        fr, k, n = orig_worst[0]
        L.append(f"| Original 56 | {k} | {fr*100:.1f}% (n={n}) |")
    L.append("")
    L.append("Most fragile models, original corpus (per-model flip rate over all attacks, n≥40):")
    L.append("")
    L += ["| Flip rate | Model |", "|---|---|"]
    for fr, k, n in orig_worst[:12]:
        L.append(f"| {fr*100:.1f}% | {k} |")
    L.append("")
    L.append("The live-tree SmsFilter and FinHealth text models appear in BOTH corpora "
             "(firmware-extracted vs ios261_live copies).  The original corpus was generated "
             "with an earlier text attack set than the current `harvest.security` "
             "generators (14 families, 32 attack types), so the 'original' column below is restricted to the "
             "overlapping attack types for a like-for-like comparison:")
    L.append("")
    L += ["| Model | Original corpus (overlap set) | This run (ios261_live) |", "|---|---|---|"]
    for model in ("mainModel_INEN", "subClassModel_INEN", "mainModel_BRPT", "basicModel_BRPT",
                  "intent_classification", "test_ghent_classifier"):
        o = orig["models"].get(model)
        new_rows = [r for r in rows if r.get("name") == model and r.get("flipped") is not None
                    and not r.get("error")]
        new_atks = {r.get("attack_type") for r in new_rows}
        if o and o["n"]:
            ov_rate, ov_n = _overlap_rate(o, new_atks)
            o_r = f"{ov_rate*100:.1f}% (n={ov_n})" if ov_n else "no overlap"
        else:
            o_r = "not tested"
        if new_rows:
            n_r = f"{sum(1 for r in new_rows if r['flipped'])/len(new_rows)*100:.1f}% (n={len(new_rows)})"
        else:
            n_r = "not tested"
        L.append(f"| {model} | {o_r} | {n_r} |")
    L.append("")
    L.append("*Full per-evaluation records (every attack, original vs attacked output, distance, "
             "latency) appended to `datasets/EXHAUSTIVE_SECURITY_DATA.jsonl`.*")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
