"""
mlmodelc_loader.py — load Apple's native .mlmodelc bundles with coremltools.

Apple's device/firmware .mlmodelc bundles ship their MIL programs as *text*
(``model.mil``) plus a weights blob, with no protobuf spec and no manifest —
formats coremltools cannot load directly.  This tool converts them:

    model.mil ──(mil_text_parser)──► MILSpec Program proto
    metadata.json ────────────────► CoreML Model description
    weights/ ─────────────────────► package weights
                 │
                 ▼
        Model proto (mlProgram) ──save_spec──► .mlpackage  ──► MLModel()

The conversion pipeline is the one proven in docs/CONVERSION_NOTES.md:
(a) the mil text is parsed into the MILSpec ``Program`` proto by
    ``harvest.mil_text_parser``; (b) the ``Model`` proto is assembled with the
    program in the ``mlProgram`` field and a description built from the
    bundle's ``metadata.json`` (or inferred from the program's function
    signature); (c) the package is written with ``coremltools.utils.save_spec``
    so the Manifest.json and weights layout match what coremltools 9.0's
    loader requires.

Both mil text dialects are handled by the parser: the classic typed-tensor
dialect (coremlc 3400–3520) and the compact scalar-constructor dialect (newer
device-side compilation).

Bundles WITHOUT a ``model.mil`` (espresso neural-network ``.coredata`` or
tree-ensemble ``.dat`` files) cannot be converted — their weights live in
compiled formats that cannot be transcribed back to protobuf.  For those,
``load_native`` hands the bundle straight to the bundled native runtime
(``CompiledMLModel``), which reads the compiled layout natively.

Usage::

    uv run python harvest/mlmodelc_loader.py convert  <bundle.mlmodelc> [out.mlpackage]
    uv run python harvest/mlmodelc_loader.py bulk     <tree> <outdir> [--verify]
    uv run python harvest/mlmodelc_loader.py load     <bundle-or-package> [--native]

Tree-scale verification (convert → load → predict per bundle, each in its own
subprocess) lives in harvest/verify_bundles.py.

Author note: written 2026-09 as a permanent research tool for firmware/device
harvests.  The parser grammar was reconstructed from observed coremlc output;
if a new firmware drops a construct it rejects, add a corpus sample and extend
harvest/mil_text_parser.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

import coremltools as ct
from coremltools import ComputeUnit
from coremltools.models import CompiledMLModel, utils as ct_utils
from coremltools.proto import MIL_pb2, Model_pb2
from coremltools.proto import FeatureTypes_pb2 as ft

save_spec = ct_utils.save_spec

from harvest.mil_text_parser import (
    ParseError,
    extract_weights_refs,
    parse_mil_file,
    strip_model_path,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIL_FILE = "model.mil"
METADATA_FILE = "metadata.json"
WEIGHTS_DIR_NAME = "weights"

# opset → CoreML specification version (mlProgram requires >= 6).
# MIL text written by coremlc uses legacy names ("ios15" ... "ios18");
# the current CoreML compiler accepts only the CoreML<sv-1> naming for
# freshly assembled specs (verified empirically: an ios15-labelled spec is
# rejected with "default function is not written in an opset that is
# supported" while the identical program labelled CoreML5 compiles).
OPSET_SPEC_VERSION = {
    "CoreML3": 4,
    "CoreML4": 5,
    "CoreML5": 6,
    "CoreML6": 7,
    "CoreML7": 8,
    "CoreML8": 9,
    "CoreML9": 10,
    # legacy text names (kept so lookups never fail)
    "ios12": 4,
    "ios13": 5,
    "ios14": 5,
    "ios15": 6,
    "ios16": 7,
    "ios17": 8,
    "ios18": 9,
}

_COREML_OPSET = {  # legacy text opset -> new-schema opset name
    "ios13": "CoreML3",
    "ios14": "CoreML4",
    "ios15": "CoreML5",
    "ios16": "CoreML6",
    "ios17": "CoreML7",
    "ios18": "CoreML8",
    "ios19": "CoreML9",
}


def _rename_opset(fn: MIL_pb2.Function) -> None:
    """Rewrite a function's opset to the new-schema CoreML<sv-1> naming.

    The parser records the mil text's legacy name verbatim; the runtime
    compiler only understands the CoreML naming in a freshly built spec.
    block_specializations keys must be renamed in lockstep.
    """
    new = _COREML_OPSET.get(fn.opset, fn.opset)
    if new != fn.opset:
        fn.opset = new
        old = list(fn.block_specializations.keys())
        if old:
            block = fn.block_specializations[old[0]]
            del fn.block_specializations[old[0]]
            fn.block_specializations[new].CopyFrom(block)

# NOTE: coremltools 9.0 ships the updated CoreML specification schema in
# which ArrayFeatureType.dataType is restricted to six values:
# FLOAT32 / DOUBLE / INT32 / INT8 / FLOAT16 (INVALID=0).  Other dtypes in the
# metadata (Int64, UInt8, ...) are coerced to the closest supported one; the
# MIL program itself carries the true dtypes.
_DATATYPE_MAP = {
    "Float32": ft.ArrayFeatureType.FLOAT32,
    "Float16": ft.ArrayFeatureType.FLOAT16,
    "Float64": ft.ArrayFeatureType.DOUBLE,
    "Double": ft.ArrayFeatureType.DOUBLE,
    "Int32": ft.ArrayFeatureType.INT32,
    "Int64": ft.ArrayFeatureType.INT32,
    "UInt8": ft.ArrayFeatureType.INT8,
    "UInt16": ft.ArrayFeatureType.INT32,
    "UInt32": ft.ArrayFeatureType.INT32,
    "UInt64": ft.ArrayFeatureType.INT32,
    "Int8": ft.ArrayFeatureType.INT8,
    "Int16": ft.ArrayFeatureType.INT32,
    "String": ft.ArrayFeatureType.INT32,  # not representable; coerced
    "Bool": ft.ArrayFeatureType.INT32,  # not representable; coerced
}
# mil dtype enum → description datatype enum (for inference when no metadata)
_MIL_TO_DESCR_DTYPE = {
    MIL_pb2.FLOAT16: ft.ArrayFeatureType.FLOAT16,
    MIL_pb2.FLOAT32: ft.ArrayFeatureType.FLOAT32,
    MIL_pb2.FLOAT64: ft.ArrayFeatureType.DOUBLE,
    MIL_pb2.INT32: ft.ArrayFeatureType.INT32,
    MIL_pb2.INT64: ft.ArrayFeatureType.INT32,
    MIL_pb2.UINT8: ft.ArrayFeatureType.INT8,
    MIL_pb2.UINT16: ft.ArrayFeatureType.INT32,
    MIL_pb2.UINT32: ft.ArrayFeatureType.INT32,
    MIL_pb2.UINT64: ft.ArrayFeatureType.INT32,
    MIL_pb2.INT8: ft.ArrayFeatureType.INT8,
    MIL_pb2.INT16: ft.ArrayFeatureType.INT32,
    MIL_pb2.BOOL: ft.ArrayFeatureType.INT32,
    MIL_pb2.STRING: ft.ArrayFeatureType.INT32,
}

# mil dtype enum → numpy dtype (for generating test inputs)
_MIL_TO_NP = {
    MIL_pb2.FLOAT16: np.float16,
    MIL_pb2.FLOAT32: np.float32,
    MIL_pb2.FLOAT64: np.float64,
    MIL_pb2.INT32: np.int32,
    MIL_pb2.INT64: np.int64,
    MIL_pb2.UINT8: np.uint8,
    MIL_pb2.UINT16: np.uint16,
    MIL_pb2.UINT32: np.uint32,
    MIL_pb2.UINT64: np.uint64,
    MIL_pb2.INT8: np.int8,
    MIL_pb2.INT16: np.int16,
    MIL_pb2.BOOL: np.bool_,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StatusRecord:
    """Outcome of one bundle's conversion/verification."""

    name: str
    bundle: str
    status: str  # "converted" | "load_failed" | "crashed" | "not_mil" | "parse_failed" | "native_only"
    output: str = ""
    error: str = ""
    functions: List[str] = field(default_factory=list)
    ops: int = 0
    spec_version: int = 0


# ---------------------------------------------------------------------------
# Description construction
# ---------------------------------------------------------------------------


def _parse_shape(shape: str) -> List[int]:
    """Parse the metadata shape string ``"[1, 1, 3]"``."""
    if not shape:
        return []
    return [int(x) for x in re.findall(r"-?\d+", shape)]


def _parse_shape_range(rng: str) -> List[tuple]:
    """Parse ``"[[1, 1], [1, 5000], [5, 5]]"`` → [(lo, hi), ...]."""
    if not rng:
        return []
    out = []
    for m in re.finditer(r"\[(-?\d+)\s*,\s*(-?\d+)\]", rng):
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def _metadata_datatype_to_descr(data_type: str) -> int:
    dt = _DATATYPE_MAP.get(data_type)
    if dt is None:
        raise ValueError(f"unknown metadata dataType {data_type!r}")
    return dt


def _set_array_feature_type(
    ft_desc: ft.ArrayFeatureType,
    data_type: int,
    shape: List[int],
    shape_range: List[tuple],
) -> None:
    ft_desc.dataType = data_type
    for d in shape:
        ft_desc.shape.append(d)
    if shape_range:
        for lo, hi in shape_range:
            r = ft_desc.shapeRange.sizeRanges.add()
            r.lowerBound = lo
            r.upperBound = hi


def _metadata_entry_to_description(
    entry: dict, dest: Model_pb2.ModelDescription, *, is_input: bool
) -> None:
    """Append one metadata schema entry to *dest* as input or output."""
    name = entry.get("name", "")
    ftype = entry.get("type", "MultiArray")
    data_type = entry.get("dataType", "Float32")
    shape = _parse_shape(entry.get("shape", ""))
    shape_range = _parse_shape_range(entry.get("shapeRange", ""))
    is_optional = str(entry.get("isOptional", "0")) in ("1", "true", "True")

    fd = dest.input.add() if is_input else dest.output.add()
    fd.name = name
    fd.shortDescription = entry.get("shortDescription", "")

    if ftype == "Image":
        img = fd.type.imageType
        img.colorSpace = ft.ImageFeatureType.BGR
        if len(shape) >= 3:
            img.width = shape[-2]
            img.height = shape[-3]
        elif len(shape) == 2:
            img.width = shape[1]
            img.height = shape[0]
    elif ftype in ("Int64", "Int32"):
        fd.type.int64Type.SetInParent()
    elif ftype == "String":
        fd.type.stringType.SetInParent()
    elif ftype in ("Float64", "Double"):
        fd.type.doubleType.SetInParent()
    elif ftype == "Dictionary":
        # class-probability style dict<string, fp32> output
        fd.type.dictionaryType.stringKeyType.SetInParent()
    elif ftype == "Sequence":
        fd.type.sequenceType.SetInParent()
    else:
        # MultiArray / Tensor / anything else → multi-array feature type
        _set_array_feature_type(
            fd.type.multiArrayType, _metadata_datatype_to_descr(data_type), shape, shape_range
        )
    fd.type.isOptional = is_optional


def build_description_from_metadata(md: dict) -> Model_pb2.ModelDescription:
    """Build the Model description from a metadata.json entry."""
    desc = Model_pb2.ModelDescription()
    for inp in md.get("inputSchema", []):
        _metadata_entry_to_description(inp, desc, is_input=True)
    for out in md.get("outputSchema", []):
        _metadata_entry_to_description(out, desc, is_input=False)
    return desc


def _dim_to_int(dim: MIL_pb2.Dimension) -> Optional[int]:
    if dim.HasField("constant"):
        return dim.constant.size
    return None


def _flexible_shape_info(program: MIL_pb2.Program, func_name: str):
    """Return (default_shapes, range_dims) dicts from the function's
    FlexibleShapeInformation attribute, if present."""
    fn = program.functions.get(func_name)
    if fn is None:
        return {}, {}
    info = fn.attributes.get("FlexibleShapeInformation")
    if info is None or not info.HasField("immediateValue"):
        return {}, {}
    tup = info.immediateValue.tuple
    if len(tup.values) < 2:
        return {}, {}

    def shapes_from(entry) -> dict:
        out = {}
        d = entry.immediateValue.tuple.values[1].immediateValue.dictionary
        for kv in d.values:
            key = kv.key.immediateValue.tensor.strings.values[0]
            val = kv.value
            if val.HasField("immediateValue") and val.immediateValue.HasField("tensor"):
                out[key] = [
                    int(x) for x in val.immediateValue.tensor.ints.values
                ]
            elif val.HasField("immediateValue") and val.immediateValue.HasField("list"):
                out[key] = [
                    int(v.immediateValue.tensor.ints.values[0])
                    for v in val.immediateValue.list.values
                    if v.HasField("immediateValue")
                    and v.immediateValue.HasField("tensor")
                    and v.immediateValue.tensor.ints.values
                ]
        return out

    return shapes_from(tup.values[0]), shapes_from(tup.values[1])


def build_description_from_program(program: MIL_pb2.Program) -> Model_pb2.ModelDescription:
    """Infer a description from the main function signature (no metadata.json).

    Uses the function's FlexibleShapeInformation attribute for default shapes
    and range dims when present (device bundles always carry it).
    """
    func_name = _default_function_name(program)
    fn = program.functions[func_name]
    block = fn.block_specializations[fn.opset]
    defaults, ranges = _flexible_shape_info(program, func_name)

    desc = Model_pb2.ModelDescription()
    for nvt in fn.inputs:
        fd = desc.input.add()
        fd.name = nvt.name
        tt = nvt.type.tensorType
        declared = [_dim_to_int(d) for d in tt.dimensions]
        dflt = defaults.get(nvt.name)
        rng = ranges.get(nvt.name)
        if dflt is None or rng is None:
            # per-dim: known dims keep their size; unknown dims get a default
            # of 1 with a wide range (the compiler rejects multiarray inputs
            # without shape constraints — predictions with data-dependent
            # shapes may still fail, but the spec compiles).
            built_d, built_r = [], []
            for size in declared:
                if size is not None:
                    built_d.append(size)
                    built_r.append((size, size))
                else:
                    built_d.append(1)
                    built_r.append((1, 65535))
            if dflt is None:
                dflt = built_d
            if rng is None:
                rng = built_r
        dt = _MIL_TO_DESCR_DTYPE.get(tt.dataType, ft.ArrayFeatureType.FLOAT32)
        _set_array_feature_type(fd.type.multiArrayType, dt, dflt, rng)
    # output types come from the producing ops — the validator requires the
    # description type to match the program's return values (an all-fp16 model
    # with a FLOAT32-declared output is rejected).
    ret_types = {}
    for op in _iter_ops(block.operations):
        for ov in op.outputs:
            if ov.type.WhichOneof("type") == "tensorType":
                tt = ov.type.tensorType
                ret_types[ov.name] = (
                    _MIL_TO_DESCR_DTYPE.get(tt.dataType, ft.ArrayFeatureType.FLOAT32),
                    _tensor_shape_with_fallbacks(tt),
                )
    for name in block.outputs:
        fd = desc.output.add()
        fd.name = name
        dt, (shape, ranges) = ret_types.get(
            name, (ft.ArrayFeatureType.FLOAT32, ([], []))
        )
        _set_array_feature_type(fd.type.multiArrayType, dt, shape, ranges)
    return desc


def _tensor_shape_with_fallbacks(tt) -> Tuple[List[int], List[tuple]]:
    """(shape, size_ranges) for a MIL tensorType, preserving its rank.

    Constant dims keep their size with an exact (size, size) range; unknown
    dims get 1 with a wide (1, 65535) range.  Dropping unknown dims entirely
    would collapse the rank, which fails validation ("Model output 'X' has a
    different rank than its corresponding return value to main").
    """
    shape, ranges = [], []
    for d in tt.dimensions:
        if d.HasField("constant"):
            shape.append(d.constant.size)
            ranges.append((d.constant.size, d.constant.size))
        else:
            shape.append(1)
            ranges.append((1, 65535))
    return shape, ranges


def _reconcile_description(
    desc: Model_pb2.ModelDescription,
    program: MIL_pb2.Program,
    func_name: str,
) -> None:
    """Make the description's I/O match the program function's actual I/O.

    metadata.json describes the ORIGINAL wrapper model (e.g. a classifier with
    ``classLabel``/``classProbability`` outputs or renamed inputs), while the
    mil text is the raw tensor function.  The compiler validates the
    description against the program's return values, so:

    - metadata inputs whose name is not a program input are dropped,
    - program inputs missing from the description are appended,
    - metadata outputs whose name is not a program return value are replaced
      by the program's actual return values (name + tensor type).
    """
    fn = program.functions[func_name]
    block = fn.block_specializations[fn.opset]
    prog_inputs = {nvt.name for nvt in fn.inputs}
    prog_outputs = set(block.outputs)

    kept = [fd for fd in desc.input if fd.name in prog_inputs]
    del desc.input[:]
    desc.input.extend(kept)
    have = {fd.name for fd in desc.input}
    for nvt in fn.inputs:
        if nvt.name not in have:
            fd = desc.input.add()
            fd.name = nvt.name
            tt = nvt.type.tensorType
            shape, ranges = _tensor_shape_with_fallbacks(tt)
            dt = _MIL_TO_DESCR_DTYPE.get(tt.dataType, ft.ArrayFeatureType.FLOAT32)
            _set_array_feature_type(fd.type.multiArrayType, dt, shape, ranges)

    kept_out = [fd for fd in desc.output if fd.name in prog_outputs]
    del desc.output[:]
    desc.output.extend(kept_out)
    if kept_out:
        return
    # none of the metadata outputs matched — emit the program's returns
    ret_types = {}
    for op in _iter_ops(block.operations):
        for ov in op.outputs:
            if ov.type.WhichOneof("type") == "tensorType":
                tt = ov.type.tensorType
                ret_types[ov.name] = (
                    _MIL_TO_DESCR_DTYPE.get(tt.dataType, ft.ArrayFeatureType.FLOAT32),
                    _tensor_shape_with_fallbacks(tt),
                )
    for name in block.outputs:
        fd = desc.output.add()
        fd.name = name
        dt, (shape, ranges) = ret_types.get(
            name, (ft.ArrayFeatureType.FLOAT32, ([], []))
        )
        _set_array_feature_type(fd.type.multiArrayType, dt, shape, ranges)


def _default_function_name(program: MIL_pb2.Program) -> str:
    """The runtime loads the function named 'main' by default; some device
    bundles (e.g. vmtracker) use initialFrame/regularFrame/updateFrame."""
    if "main" in program.functions:
        return "main"
    return next(iter(program.functions))


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _opset_to_spec_version(opset: str) -> int:
    """Spec version required for a program whose opset is *opset*."""
    return OPSET_SPEC_VERSION.get(opset, 6)


_LOAD_CHECK_TEMPLATE = (
    "import sys, warnings; warnings.filterwarnings('ignore');"
    "import coremltools as ct;"
    "ct.models.MLModel(sys.argv[1]); print('OK')"
)


def _load_check(package: str, timeout: int = 300) -> tuple:
    """Load *package* in a fresh subprocess; return (ok, error_message).

    ``ct.models.MLModel`` invokes the CoreML compiler (coremlc), which is a
    C++ program that can die on a signal for some device programs — that is
    not catchable from Python, so the check must live in its own process.
    Returns (True, "") on success; on failure the error string starts with
    "signal" when the compiler crashed hard (status "crashed") so callers can
    distinguish it from a regular load failure.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _LOAD_CHECK_TEMPLATE, package],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"load check timed out after {timeout}s"
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        if proc.returncode < 0:
            return False, f"signal {proc.returncode} ({msg[-200:]})"
        return False, (msg or f"exit {proc.returncode}")[-300:]
    return True, ""


def convert(bundle_path: str, output_path: Optional[str] = None) -> StatusRecord:
    """Convert one native .mlmodelc bundle into a loadable .mlpackage.

    Returns a StatusRecord describing the outcome.
    """
    bundle = Path(bundle_path)
    if not bundle.is_dir():
        raise ValueError(f"Not a directory: {bundle_path}")
    if output_path is None:
        output_path = str(bundle).rstrip("/") + ".mlpackage"
    elif not output_path.endswith(".mlpackage"):
        # save_spec infers the container from the extension
        output_path += ".mlpackage"

    record = StatusRecord(name=bundle.name, bundle=str(bundle), status="")

    mil_path = bundle / MIL_FILE
    if not mil_path.exists():
        record.status = "not_mil"
        record.error = "no model.mil (compiled espresso/tree-ensemble bundle)"
        return record

    # 1. parse mil text → Program proto
    try:
        program = parse_mil_file(str(mil_path))
    except ParseError as e:
        record.status = "parse_failed"
        record.error = str(e)
        return record
    # 1b. the mil text's legacy opset names ("ios15") are not accepted by the
    # current compiler when building a fresh spec — rename to CoreML<sv-1>.
    for fn in program.functions.values():
        _rename_opset(fn)
    record.functions = list(program.functions.keys())
    func_name = _default_function_name(program)
    block = program.functions[func_name].block_specializations[
        program.functions[func_name].opset
    ]
    record.ops = sum(1 for _ in _iter_ops(block.operations))

    # 2. description
    metadata = None
    md_path = bundle / METADATA_FILE
    if md_path.exists():
        try:
            md_list = json.loads(md_path.read_text())
            if isinstance(md_list, list):
                metadata = md_list[0] if md_list else {}
            else:
                metadata = md_list
        except json.JSONDecodeError as e:
            record.error = f"bad metadata.json: {e}"
            return record
    if metadata:
        desc = build_description_from_metadata(metadata)
        spec_version = int(metadata.get("specificationVersion", 0) or 0)
    else:
        desc = build_description_from_program(program)
        spec_version = 0
    _reconcile_description(desc, program, func_name)

    # 3. assemble the Model proto.
    # coremltools 9.0 ships the updated CoreML spec schema, where the
    # Program's contents live directly under Model.mlProgram (functions map,
    # attributes map, version) instead of a nested `program` message.
    spec = Model_pb2.Model()
    spec.description.CopyFrom(desc)
    for fname, fn in program.functions.items():
        spec.mlProgram.functions[fname].CopyFrom(fn)
    for aname, aval in program.attributes.items():
        spec.mlProgram.attributes[aname].CopyFrom(aval)
    spec.mlProgram.version = program.version
    spec.specificationVersion = max(
        spec_version,
        _opset_to_spec_version(program.functions[func_name].opset),
    )
    record.spec_version = spec.specificationVersion

    # 4. weights → temp dir (save_spec copies it into the package)
    weights_src = bundle / WEIGHTS_DIR_NAME
    with tempfile.TemporaryDirectory(prefix="mlmodelc_w_") as wdir:
        if weights_src.is_dir():
            for f in sorted(weights_src.iterdir()):
                if f.is_file():
                    shutil.copy2(f, Path(wdir) / f.name)
        out = Path(output_path)
        if out.exists():
            shutil.rmtree(out)
        try:
            save_spec(spec, str(out), weights_dir=wdir)
        except Exception as e:
            record.status = "convert_failed"
            record.error = f"save_spec: {e}"
            return record

    # 5. verify the package loads through coremltools — in a subprocess.
    # The CoreML compiler can hard-crash (SIGSEGV, not catchable in-process)
    # on some device programs when the description honestly matches their
    # tensor I/O (e.g. hand_gesture_dynamic_two_hand_5fps), so the load check
    # must not run in the caller's process.
    ok, err = _load_check(str(out))
    if not ok:
        record.status = "load_failed" if "signal" not in err else "crashed"
        record.error = err
        return record

    record.status = "converted"
    record.output = str(out)
    return record


def _iter_ops(operations):
    for op in operations:
        yield op
        for b in op.blocks:
            yield from _iter_ops(b.operations)


_PREDICT_TEMPLATE = (
    "import sys, warnings; warnings.filterwarnings('ignore')\n"
    "import numpy as np\n"
    "import coremltools as ct\n"
    "from harvest.mlmodelc_loader import make_test_inputs\n"
    "m = ct.models.MLModel(sys.argv[1])\n"
    "inputs = make_test_inputs(m)\n"
    "if any(v is None for v in inputs.values()):\n"
    "    print('SKIP image input'); sys.exit(0)\n"
    "outs = m.predict(inputs)\n"
    "print('OK ' + ', '.join(f'{k}{tuple(np.asarray(v).shape)}' for k, v in outs.items()))\n"
)


def _predict_check(package: str, timeout: int = 300) -> tuple:
    """Run one predict() in a subprocess; return (ok, message).

    Same rationale as ``_load_check``: predict-time IR compilation can kill
    the process on a signal for some device programs.  The template prints
    "OK <output shapes>" or "SKIP image input"; anything else is an error.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PREDICT_TEMPLATE, package],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"predict check timed out after {timeout}s"
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        if proc.returncode < 0:
            return False, f"signal {proc.returncode} ({err[-150:]})"
        return False, (err or out or f"exit {proc.returncode}")[-300:]
    if out.startswith("OK"):
        return True, out[3:]
    return True, out  # "SKIP image input"


def bulk_convert(
    tree: str,
    output_dir: str,
    *,
    verify: bool = True,
    verbose: bool = True,
) -> List[StatusRecord]:
    """Convert every *.mlmodelc bundle under *tree* into *output_dir*.

    Returns per-bundle StatusRecords.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    records: List[StatusRecord] = []
    for root, dirs, files in os.walk(tree):
        if not root.endswith(".mlmodelc"):
            continue
        bundle = root
        # output path mirrors the source tree (name + .mlpackage);
        # when *tree* IS the bundle itself, relpath is "." — fall back to
        # the bundle's name (otherwise the path degrades to a stray file)
        rel = os.path.relpath(bundle, tree)
        if rel == ".":
            rel = os.path.basename(bundle)
        out = out_root / (rel + ".mlpackage")
        out.parent.mkdir(parents=True, exist_ok=True)
        rec = convert(bundle, str(out))
        # prediction check: the task requires converted models to produce
        # predictions where the spec allows.  Runs in a subprocess — predict
        # can SIGSEGV on some device programs and must not kill the run.
        if verify and rec.status == "converted":
            ok, msg = _predict_check(str(out))
            if not ok:
                rec.status = "predict_failed"
                rec.error = msg
            else:
                rec.error = f"predict ok: {msg}"
        records.append(rec)
        if verbose:
            mark = {"converted": "OK ", "not_mil": "-- "}.get(rec.status, "!! ")
            print(f"{mark} {rel:<70} {rec.status} {rec.error[:60]}")
    return records


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(path: str, *, fallback_to_native: bool = True) -> Any:
    """Load a model, transparently converting native bundles.

    - .mlpackage / .mlmodel      → ``ct.models.MLModel`` directly
    - .mlmodelc (MIL text)       → converted on the fly to a temp package
    - .mlmodelc (compiled only)  → ``CompiledMLModel`` (native runtime) when
      *fallback_to_native* is set, else raises
    """
    p = Path(path)
    if p.suffix in (".mlpackage", ".mlmodel"):
        return ct.models.MLModel(str(p))
    if not p.is_dir():
        raise ValueError(f"not a loadable model path: {path}")

    if (p / MIL_FILE).exists():
        with tempfile.TemporaryDirectory(prefix="mlmodelc_load_") as td:
            out = os.path.join(td, "model.mlpackage")
            rec = convert(str(p), out)
            if rec.status != "converted":
                raise RuntimeError(f"cannot convert {path}: {rec.error}")
            return ct.models.MLModel(out)
    if fallback_to_native:
        return load_native(str(p))
    raise ValueError(
        f"{path} has no model.mil; pass fallback_to_native=True to use the "
        "bundled native runtime (CompiledMLModel)"
    )


def load_native(path: str, function_name: Optional[str] = None) -> CompiledMLModel:
    """Load a compiled bundle directly with the bundled native runtime.

    Works for both MIL-text bundles and compiled-only bundles, but does not
    expose the Python spec (no ``get_spec``).  Use this when a bundle cannot
    be converted (no model.mil) or as a reference for prediction checks.
    """
    kwargs = {"compute_units": ComputeUnit.CPU_ONLY}
    if function_name:
        kwargs["function_name"] = function_name
    return CompiledMLModel(str(path), **kwargs)


def make_test_inputs(model: Any) -> dict:
    """Generate plausible random inputs for *model* from its spec.

    Works for converted (spec-bearing) models; fills every input with random
    data of the right dtype and default shape (or the shape range lower bound
    for flexible dims).
    """
    spec = model.get_spec()
    inputs = {}
    for fd in spec.description.input:
        t = fd.type.WhichOneof("Type")
        if t == "multiArrayType":
            mt = fd.type.multiArrayType
            shape = list(mt.shape)
            rng = list(mt.shapeRange.sizeRanges)
            if not shape and rng:
                shape = [int(r.lowerBound) for r in rng]
            shape = [d if d > 0 else 1 for d in shape]
            dt = _descr_dtype_to_np(mt.dataType)
            if np.issubdtype(dt, np.integer):
                # integer inputs are usually indices/lengths — random values
                # (e.g. negative lengths) make data-dependent-shape models
                # fail at runtime, so fill with 1s
                inputs[fd.name] = np.ones(shape, dtype=dt)
            else:
                inputs[fd.name] = np.random.default_rng(0).normal(size=shape).astype(dt)
        elif t == "imageType":
            inputs[fd.name] = None  # caller must supply an image
        elif t in ("int64Type", "doubleType", "stringType"):
            inputs[fd.name] = {"int64Type": 0, "doubleType": 0.0, "stringType": ""}[t]
    return inputs


def _descr_dtype_to_np(dt: int) -> np.dtype:
    return {
        ft.ArrayFeatureType.FLOAT16: np.float16,
        ft.ArrayFeatureType.FLOAT32: np.float32,
        ft.ArrayFeatureType.DOUBLE: np.float64,
        ft.ArrayFeatureType.INT32: np.int32,
        ft.ArrayFeatureType.INT8: np.int8,
    }.get(dt, np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlmodelc_loader",
        description="Convert/load Apple native .mlmodelc bundles with coremltools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="convert one bundle to .mlpackage")
    p_convert.add_argument("bundle", help="path to a .mlmodelc directory")
    p_convert.add_argument("output", nargs="?", help="output .mlpackage path")

    p_bulk = sub.add_parser("bulk", help="convert every bundle under a tree")
    p_bulk.add_argument("tree", help="directory tree containing .mlmodelc bundles")
    p_bulk.add_argument("output_dir", help="output directory")
    p_bulk.add_argument("--no-verify", action="store_true", help="skip load check")

    p_load = sub.add_parser("load", help="load a bundle/package and print summary")
    p_load.add_argument("path", help=".mlpackage / .mlmodel / .mlmodelc path")
    p_load.add_argument("--native", action="store_true", help="use CompiledMLModel directly")
    p_load.add_argument("--predict", action="store_true", help="run a random-input prediction")

    args = parser.parse_args(argv)

    if args.command == "convert":
        rec = convert(args.bundle, args.output)
        print(f"{rec.status}: {rec.output or args.bundle}  {rec.error}")
        return 0 if rec.status == "converted" else 1

    if args.command == "bulk":
        records = bulk_convert(args.tree, args.output_dir, verify=not args.no_verify)
        stats = {}
        for r in records:
            stats[r.status] = stats.get(r.status, 0) + 1
        print("\nsummary:", stats)
        return 0 if stats.get("load_failed") is None and stats.get("parse_failed") is None else 1

    if args.command == "load":
        try:
            model = load_native(args.path) if args.native else load(args.path)
        except Exception as e:
            print(f"load failed: {e}")
            return 1
        print(f"loaded {type(model).__name__}: {args.path}")
        try:
            spec = model.get_spec()
            ins = [f"{i.name}: {i.type.WhichOneof('Type')}" for i in spec.description.input]
            outs = [f"{o.name}: {o.type.WhichOneof('Type')}" for o in spec.description.output]
            print("  inputs :", ", ".join(ins))
            print("  outputs:", ", ".join(outs))
        except Exception:
            pass  # CompiledMLModel has no spec
        if args.predict and getattr(model, "predict", None):
            try:
                inputs = make_test_inputs(model)
                if any(v is None for v in inputs.values()):
                    print("  (skipping predict: image inputs need real images)")
                else:
                    out = model.predict(inputs)
                    print("  predict ok:", {k: str(np.asarray(v).shape) for k, v in out.items()})
            except Exception as e:
                print(f"  predict failed: {e}")
                return 1
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
