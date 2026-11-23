"""
mil_text_parser.py — MIL text (``model.mil``) → MILSpec protobuf parser.

Apple's native ``.mlmodelc`` bundles carry their MIL programs as *text* files
(``model.mil``) plus a blob of weights (``weights/weight.bin``).  coremltools
can load MIL programs only in their *protobuf* form (the ``mlProgram`` field of
the CoreML ``Model`` proto).  This module is a from-scratch recursive-descent
parser for the MIL text grammar that emits the protobuf form, so that native
bundles can be repackaged as loadable ``.mlpackage``/``.mlmodel`` files.

It covers the two dialects seen in the wild:

* the "classic" dialect (coremlc ≈ 3400–3520, iOS 15-era compile targets),
  where every value is written as a typed tensor, e.g.
  ``val = tensor<fp16, [128, 512]>(BLOBFILE(path = tensor<string, []>(
  "@model_path/weights/weight.bin"), offset = tensor<uint64, []>(640)))`` and
  attributes look like ``[name = tensor<string, []>("op_3")]``;

* the "compact" dialect (newer device-side compilation), where scalar values
  are written as scalar constructors, e.g. ``name = string("op_3")``,
  ``offset = uint64(64)``, and op inputs may be bound directly to values:
  ``indices = tensor<uint4, [...]>(BLOBFILE(...))``.

The parser is deliberately dependency-free (only the generated MIL_pb2 proto
classes).  It does NOT resolve weights or evaluate the program — it is a pure
syntax → proto transcription.

Grammar sketch (both dialects):

    program  := 'program' '(' ver ')' buildinfo? '{' func* '}'
    buildinfo:= '[' 'buildInfo' '=' value ']'
    func     := 'func' name '<' opset '>' '(' params ')' attrs? block '->' '(' names ')' ';'
    block    := '{' op* '}'
    op       := typed_outputs '=' name '(' op_args ')' attrs? blockdef* ';'
    op_args  := (name '=' argval) (',' name '=' argval)*
    argval   := name                    -- binding to an SSA var
              | value                   -- binding to an inline value
              | '(' argval (',' argval)* ')'   -- multiple bindings (variadic/tuple inputs)
    blockdef := '(' params ')' attrs? block '->' '(' names ')'
    params   := (type name) (',' type name)*
    attrs    := '[' (name '=' value) (',' name '=' value)* ']'
    value    := 'tensor' '<' dtype ',' '[' dims ']' '>' '(' tcontents ')'
              | 'list'  '<' type ',' len '>' '(' lcontents ')'
              | 'tuple' '<' type (',' type)* '>' '(' tcontents ')'
              | 'dict'  '<' ktype ',' vtype '>' '(' dcontents ')'
              | dtype '(' literal ')'                  -- compact scalar
              | 'BLOBFILE' '(' 'path' '=' value ',' 'offset' '=' value ')'
              | literal | '[' literals ']' | '(' values ')' | '{' entries '}'  -- shorthand
    tcontents:= 'BLOBFILE' '(' ... ')' | '[' literal (',' literal)* ']' | literal
    dims     := ('?' | '...' | integer | '*') (',' ...)*

Author note: written 2026-09 for the mlmodelc_loader project.  The grammar is
reconstructed from observed coremlc output (iOS 26 device bundles and
coremltools-compiled reference models); if you hit a construct this parser
rejects, add a corpus sample to the tests before extending the grammar.
"""

from __future__ import annotations

import re
import struct
from typing import List, Optional, Tuple

from coremltools.proto import MIL_pb2

# ---------------------------------------------------------------------------
# Constants: dtype name → MILSpec.DataType enum
# ---------------------------------------------------------------------------

SCALAR_DTYPES = {
    "fp16": MIL_pb2.FLOAT16,
    "fp32": MIL_pb2.FLOAT32,
    "fp64": MIL_pb2.FLOAT64,
    "bfloat16": MIL_pb2.BFLOAT16,
    "fp8e4m3fn": MIL_pb2.FLOAT8E4M3FN,
    "fp8e5m2": MIL_pb2.FLOAT8E5M2,
    "int8": MIL_pb2.INT8,
    "int16": MIL_pb2.INT16,
    "int32": MIL_pb2.INT32,
    "int64": MIL_pb2.INT64,
    "int4": MIL_pb2.INT4,
    "uint8": MIL_pb2.UINT8,
    "uint16": MIL_pb2.UINT16,
    "uint32": MIL_pb2.UINT32,
    "uint64": MIL_pb2.UINT64,
    "uint1": MIL_pb2.UINT1,
    "uint2": MIL_pb2.UINT2,
    "uint3": MIL_pb2.UINT3,
    "uint4": MIL_pb2.UINT4,
    "uint6": MIL_pb2.UINT6,
    "bool": MIL_pb2.BOOL,
    "string": MIL_pb2.STRING,
}

# dtypes whose immediate values are stored in the TensorValue.bytes field
# (packed / raw little-endian), per coremltools' MIL serializer.
_BYTES_DTYPES = {
    MIL_pb2.FLOAT16,
    MIL_pb2.INT8,
    MIL_pb2.UINT8,
    MIL_pb2.UINT32,
    MIL_pb2.INT4,
    MIL_pb2.UINT1,
    MIL_pb2.UINT2,
    MIL_pb2.UINT3,
    MIL_pb2.UINT4,
    MIL_pb2.UINT6,
}
_BIT_WIDTH = {
    MIL_pb2.INT4: 4,
    MIL_pb2.UINT1: 1,
    MIL_pb2.UINT2: 2,
    MIL_pb2.UINT3: 3,
    MIL_pb2.UINT4: 4,
    MIL_pb2.UINT6: 6,
}

# pixel buffer format → tensor dtype used for the MIL program input type
_PIXEL_FMT_DTYPES = {
    "FMT_32BGRA": MIL_pb2.UINT8,  # BGRA8
    "FMT_32ARGB": MIL_pb2.UINT8,
    "FMT_32RGBA": MIL_pb2.UINT8,
    "FMT_32ABGR": MIL_pb2.UINT8,
    "FMT_64RGBAHalf": MIL_pb2.FLOAT16,
    "FMT_128RGBAFloat": MIL_pb2.FLOAT32,
    "FMT_L8": MIL_pb2.UINT8,
    "FMT_16LA": MIL_pb2.UINT16,
    "FMT_24BGR": MIL_pb2.UINT8,
    "FMT_32BGRX": MIL_pb2.UINT8,
    "FMT_32XRGB": MIL_pb2.UINT8,
    "FMT_32XBGR": MIL_pb2.UINT8,
    "FMT_16Gray": MIL_pb2.UINT16,
    "FMT_8Gray": MIL_pb2.UINT8,
    "FMT_RGB565": MIL_pb2.UINT16,
    "FMT_10RGBX": MIL_pb2.UINT16,
    "FMT_10XRGB": MIL_pb2.UINT16,
    "FMT_12BGR": MIL_pb2.UINT16,
    "FMT_30RGB": MIL_pb2.UINT32,
    "FMT_30RGBX": MIL_pb2.UINT32,
    "FMT_30XRGB": MIL_pb2.UINT32,
}

_TOKEN_EOF = ("eof", "", -1)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when the MIL text cannot be parsed."""


def _tokenize(text: str) -> List[Tuple[str, str, int]]:
    tokens: List[Tuple[str, str, int]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        # arrow (must be checked before '-' as a number sign)
        if ch == "-" and i + 1 < n and text[i + 1] == ">":
            tokens.append(("arrow", "->", i))
            i += 2
            continue
        # string literal
        if ch == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':
                c = text[j]
                if c == "\\" and j + 1 < n:
                    nxt = text[j + 1]
                    esc = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'"}
                    if nxt in esc:
                        buf.append(esc[nxt])
                        j += 2
                    elif nxt == "u" and j + 5 < n:
                        buf.append(chr(int(text[j + 2 : j + 6], 16)))
                        j += 6
                    else:
                        buf.append(c)
                        j += 1
                else:
                    buf.append(c)
                    j += 1
            if j >= n:
                raise ParseError(f"unterminated string at offset {i}")
            tokens.append(("string", "".join(buf), i))
            i = j + 1
            continue
        # number: optional sign, hex float / hex int / decimal
        if ch.isdigit() or ch in "+-" or ch == ".":
            j = i
            if ch in "+-" and j + 1 < n and (text[j + 1].isdigit() or text[j + 1] == "."):
                j += 1
            elif ch in "+-" and j + 1 < n and text[j + 1 : j + 4] in (
                "inf",
                "nan",
            ):
                # signed special literal: -inf / -nan / +inf
                j += 4
                if text[i + 4 : i + 4 + 8] == "infinity":
                    j += 5
                tokens.append(("num", text[i:j], i))
                i = j
                continue
            # hex?
            if j + 1 < n and text[j] == "0" and text[j + 1] in "xX":
                j += 2
                while j < n and (text[j] in "0123456789abcdefABCDEF._pP+-"):
                    j += 1
            else:
                while j < n and (text[j].isdigit() or text[j] in "._eE+-"):
                    # don't swallow a '+'/'-' that begins the next token
                    if text[j] in "+-" and j > i and not (j > i and text[j - 1] in "eEpP"):
                        break
                    j += 1
            # guard: '.' alone is not a number
            tok = text[i:j]
            if tok.strip("."):
                tokens.append(("num", tok, i))
                i = j
                continue
        # identifiers (also catches `true` / `false`)
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(("ident", text[i:j], i))
            i = j
            continue
        if ch in "()[]{}<>,=;":
            tokens.append((ch, ch, i))
            i += 1
            continue
        if ch == "?":
            tokens.append(("?", "?", i))
            i += 1
            continue
        raise ParseError(f"unexpected character {ch!r} at offset {i}")
    tokens.append(_TOKEN_EOF)
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class MilTextParser:
    """Recursive-descent parser: MIL text → ``coremltools.proto.MIL_pb2.Program``."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.pos = 0

    # -- token helpers ------------------------------------------------------

    def peek(self) -> Tuple[str, str, int]:
        return self.tokens[self.pos]

    def advance(self) -> Tuple[str, str, int]:
        tok = self.tokens[self.pos]
        if tok[0] != "eof":
            self.pos += 1
        return tok

    def expect(self, kind: str, value: Optional[str] = None) -> Tuple[str, str, int]:
        tok = self.peek()
        if tok[0] != kind or (value is not None and tok[1] != value):
            raise ParseError(
                f"expected {value or kind!r}, got {tok[1]!r} at offset {tok[2]}"
            )
        return self.advance()

    def at(self, kind: str, value: Optional[str] = None) -> bool:
        tok = self.peek()
        return tok[0] == kind and (value is None or tok[1] == value)

    # -- entry point --------------------------------------------------------

    def parse_program(self) -> MIL_pb2.Program:
        self.expect("ident", "program")
        self.expect("(")
        # The proto stores the MIL program version as an int64; coremltools'
        # own exporter writes 1 for every deployment target, so the legacy
        # float in the text (1.0/1.2/1.3) collapses to 1.
        ver = self.expect("num")[1]
        self.expect(")")
        prog = MIL_pb2.Program(version=int(float(ver)))
        if self.at("["):
            self.advance()
            while not self.at("]"):
                key = self.expect("ident")[1]
                self.expect("=")
                prog.attributes[key].CopyFrom(self.parse_value())
                if self.at(","):
                    self.advance()
            self.expect("]")
        self.expect("{")
        while not self.at("}"):
            self.parse_function(prog)
        self.expect("}")
        self.expect("eof")
        return prog

    # -- functions ----------------------------------------------------------

    def parse_function(self, prog: MIL_pb2.Program) -> None:
        self.expect("ident", "func")
        fname = self.expect("ident")[1]
        self.expect("<")
        opset = self.expect("ident")[1]
        self.expect(">")
        self.expect("(")
        fn = MIL_pb2.Function(opset=opset)
        if not self.at(")"):
            while True:
                fn.inputs.append(self.parse_named_value_type())
                if not self.at(","):
                    break
                self.advance()
        self.expect(")")
        if self.at("["):
            self.parse_attrs(fn.attributes)
        # function body block: { ops } -> (outputs);
        block = MIL_pb2.Block()
        self.expect("{")
        self.parse_operations(block)
        self.expect("}")
        self.expect("arrow")
        self.expect("(")
        while not self.at(")"):
            block.outputs.append(self.expect("ident")[1])
            if self.at(","):
                self.advance()
        self.expect(")")
        self.expect(";")
        fn.block_specializations[opset].CopyFrom(block)
        prog.functions[fname].CopyFrom(fn)

    # -- typed params -------------------------------------------------------

    def parse_named_value_type(self) -> MIL_pb2.NamedValueType:
        vtype = self.parse_type()
        name = self.expect("ident")[1]
        nvt = MIL_pb2.NamedValueType(name=name)
        nvt.type.CopyFrom(vtype)
        return nvt

    # -- types --------------------------------------------------------------

    def parse_type(self) -> MIL_pb2.ValueType:
        name = self.peek()
        if name[0] != "ident":
            raise ParseError(f"expected a type, got {name[1]!r} at offset {name[2]}")
        if name[1] == "tensor":
            self.advance()
            self.expect("<")
            dtype = self._parse_dtype_name()
            self.expect(",")
            self.expect("[")
            dims = self.parse_dimensions()
            self.expect("]")
            self.expect(">")
            return self._make_tensor_type(dtype, dims)
        if name[1] == "list":
            self.advance()
            self.expect("<")
            elem = self.parse_type()
            self.expect(",")
            length = self.parse_dimension()
            self.expect(">")
            vt = MIL_pb2.ValueType()
            lt = vt.listType
            lt.type.CopyFrom(elem)
            self._set_dim(lt.length, length)
            return vt
        if name[1] == "tuple":
            self.advance()
            self.expect("<")
            vt = MIL_pb2.ValueType()
            while True:
                vt.tupleType.types.add().CopyFrom(self.parse_type())
                if not self.at(","):
                    break
                self.advance()
            self.expect(">")
            return vt
        if name[1] == "dict":
            self.advance()
            self.expect("<")
            vt = MIL_pb2.ValueType()
            vt.dictionaryType.keyType.CopyFrom(self.parse_type())
            self.expect(",")
            vt.dictionaryType.valueType.CopyFrom(self.parse_type())
            self.expect(">")
            return vt
        if name[1] == "state":
            self.advance()
            self.expect("<")
            vt = MIL_pb2.ValueType()
            vt.stateType.wrappedType.CopyFrom(self.parse_type())
            self.expect(">")
            return vt
        if name[1] == "pixel_buffer":
            # image input: pixel_buffer<FMT_32BGRA, shape=[512, 512]>.
            # The MIL program proto has no image type, so this is represented
            # as a tensor whose dtype is derived from the format and whose
            # dimensions come from the shape= parameter.  The CoreML
            # DESCRIPTION (built separately from metadata.json / inference)
            # carries the true imageType semantics.
            self.advance()
            self.expect("<")
            fmt = self.expect("ident")[1]
            self.expect(",")
            self.expect("ident", "shape")
            self.expect("=")
            self.expect("[")
            dims = self.parse_dimensions()
            self.expect("]")
            self.expect(">")
            dtype = _PIXEL_FMT_DTYPES.get(fmt)
            if dtype is None:
                raise ParseError(f"unknown pixel format {fmt!r} at offset {name[2]}")
            return self._make_tensor_type(dtype, dims)
        # bare scalar dtype
        self.advance()
        if name[1] not in SCALAR_DTYPES:
            raise ParseError(f"unknown type name {name[1]!r} at offset {name[2]}")
        return self._make_tensor_type(SCALAR_DTYPES[name[1]], [])

    def _parse_dtype_name(self) -> int:
        tok = self.expect("ident")
        if tok[1] not in SCALAR_DTYPES:
            raise ParseError(f"unknown dtype {tok[1]!r} at offset {tok[2]}")
        return SCALAR_DTYPES[tok[1]]

    def _make_tensor_type(self, dtype: int, dims: List[object]) -> MIL_pb2.ValueType:
        vt = MIL_pb2.ValueType()
        tt = vt.tensorType
        tt.dataType = dtype
        tt.rank = len(dims)
        for d in dims:
            self._set_dim(tt.dimensions.add(), d)
        return vt

    def _set_dim(self, dim: MIL_pb2.Dimension, d: object) -> None:
        if isinstance(d, int):
            dim.constant.size = d
        elif d == "?":
            dim.unknown.variadic = False
        elif d in ("...", "*"):
            dim.unknown.variadic = True
        else:
            raise ParseError(f"cannot encode dimension {d!r}")

    def parse_dimension(self) -> object:
        tok = self.peek()
        if tok[0] == "num":
            self.advance()
            try:
                return int(tok[1], 0)
            except ValueError:
                raise ParseError(f"bad integer {tok[1]!r} at offset {tok[2]}")
        if tok[0] == "?" or (tok[0] == "ident" and tok[1] == "?"):
            self.advance()
            return "?"
        if tok[0] == "ident" and tok[1] in ("...",):
            self.advance()
            return "..."
        if tok[0] == "ident" and tok[1] == "*":
            self.advance()
            return "..."
        raise ParseError(f"expected a dimension, got {tok[1]!r} at offset {tok[2]}")

    def parse_dimensions(self) -> List[object]:
        dims = []
        while not self.at("]"):
            dims.append(self.parse_dimension())
            if self.at(","):
                self.advance()
        return dims

    # -- operations ---------------------------------------------------------

    def parse_operations(self, block: MIL_pb2.Block) -> None:
        while not self.at("}"):
            block.operations.append(self.parse_operation())

    def parse_operation(self) -> MIL_pb2.Operation:
        # output list: type name, type name, ...
        outputs: List[MIL_pb2.NamedValueType] = []
        while True:
            outputs.append(self.parse_named_value_type())
            if not self.at(","):
                break
            self.advance()
        self.expect("=")
        op_type = self.expect("ident")[1]
        self.expect("(")
        op = MIL_pb2.Operation(type=op_type)
        if not self.at(")"):
            while True:
                pname = self.expect("ident")[1]
                self.expect("=")
                arg = self.parse_input_arg()
                op.inputs[pname].CopyFrom(arg)
                if not self.at(","):
                    break
                self.advance()
        self.expect(")")
        if self.at("["):
            self.parse_attrs(op.attributes)
        # control-flow blocks: (params) { ops } -> (outs)   (repeat)
        while self.at("("):
            block = MIL_pb2.Block()
            self.advance()
            if not self.at(")"):
                while True:
                    block.inputs.append(self.parse_named_value_type())
                    if not self.at(","):
                        break
                    self.advance()
            self.expect(")")
            if self.at("["):
                self.parse_attrs(block.attributes)
            self.expect("{")
            self.parse_operations(block)
            self.expect("}")
            self.expect("arrow")
            self.expect("(")
            while not self.at(")"):
                block.outputs.append(self.expect("ident")[1])
                if self.at(","):
                    self.advance()
            self.expect(")")
            op.blocks.append(block)
        self.expect(";")
        for o in outputs:
            op.outputs.append(o)
        return op

    def parse_input_arg(self) -> MIL_pb2.Argument:
        arg = MIL_pb2.Argument()
        if self.at("("):
            # multiple bindings (variadic / tuple-typed inputs)
            self.advance()
            while not self.at(")"):
                arg.arguments.append(self.parse_binding())
                if self.at(","):
                    self.advance()
            self.expect(")")
            return arg
        arg.arguments.append(self.parse_binding())
        return arg

    def parse_binding(self) -> MIL_pb2.Argument.Binding:
        tok = self.peek()
        if tok[0] == "ident" and not self._looks_like_value(tok):
            self.advance()
            return MIL_pb2.Argument.Binding(name=tok[1])
        b = MIL_pb2.Argument.Binding()
        b.value.CopyFrom(self.parse_value())
        return b

    def _looks_like_value(self, tok: Tuple[str, str, int]) -> bool:
        """An ident starts a value if it names a type constructor or a literal."""
        nxt = self.tokens[self.pos + 1]
        return (
            tok[1] in SCALAR_DTYPES
            or tok[1] in ("tensor", "list", "tuple", "dict", "state", "BLOBFILE")
            or (nxt[0] == "(" and tok[1] in ("true", "false"))
            or nxt[0] in ("<", "(")
        )

    # -- attributes ---------------------------------------------------------

    def parse_attrs(self, dst) -> None:
        self.expect("[")
        while True:
            key = self.expect("ident")[1]
            self.expect("=")
            dst[key].CopyFrom(self.parse_value())
            if not self.at(","):
                break
            self.advance()
        self.expect("]")

    # -- values -------------------------------------------------------------

    def parse_value(self) -> MIL_pb2.Value:
        """Parse any value form (typed or shorthand).  No expected-type context
        is required: shorthands only appear inside typed containers, which call
        the specialized content parsers below."""
        tok = self.peek()
        if tok[0] == "ident":
            nxt = self.tokens[self.pos + 1]
            if tok[1] in ("tensor", "list", "tuple", "dict", "state"):
                return self.parse_typed_value()
            if tok[1] in SCALAR_DTYPES and nxt[0] == "(":
                return self.parse_scalar_constructor()
            if tok[1] == "BLOBFILE":
                return self.parse_blobfile_value()
            raise ParseError(f"expected a value, got {tok[1]!r} at offset {tok[2]}")
        raise ParseError(f"expected a value, got {tok[1]!r} at offset {tok[2]}")

    def parse_typed_value(self) -> MIL_pb2.Value:
        tname = self.peek()[1]
        vtype = self.parse_type()
        val = MIL_pb2.Value()
        val.type.CopyFrom(vtype)
        self.expect("(")
        if tname == "tensor":
            tt = vtype.tensorType
            contents = self.parse_tensor_contents(tt.dataType, tt.dimensions)
            if contents[0] == "blob":
                val.blobFileValue.CopyFrom(contents[1])
            else:
                val.immediateValue.tensor.CopyFrom(contents[1])
            self.expect(")")
        elif tname == "tuple":
            # a tuple VALUE is written `tuple<...>( (v1, v2) )` — the literal
            # `(v1, v2)` is consumed by parse_tuple_contents, then the value's
            # closing paren follows.
            val.immediateValue.tuple.CopyFrom(self.parse_tuple_contents(vtype))
            self.expect(")")
        elif tname == "list":
            # parse_list_contents consumes its own ']'
            val.immediateValue.list.CopyFrom(self.parse_list_contents(vtype.listType))
            self.expect(")")
        elif tname == "dict":
            # parse_dict_contents consumes its own final '}'
            val.immediateValue.dictionary.CopyFrom(self.parse_dict_contents(vtype.dictionaryType))
            self.expect(")")
        elif tname == "state":
            # state values wrap the inner tensor
            val.immediateValue.tensor.CopyFrom(self.parse_state_contents(vtype.stateType))
            self.expect(")")
        else:  # pragma: no cover
            raise ParseError(f"unhandled typed value {tname!r}")
        return val

    def parse_scalar_constructor(self) -> MIL_pb2.Value:
        dtype = self._parse_dtype_name()
        val = MIL_pb2.Value()
        val.type.CopyFrom(self._make_tensor_type(dtype, []))
        self.expect("(")
        val.immediateValue.tensor.CopyFrom(self.parse_scalar_immediate(dtype))
        self.expect(")")
        return val

    # -- tensor contents ----------------------------------------------------

    def parse_tensor_contents(self, dtype: int, dims):
        """Parse the inside of ``tensor<dtype, dims>(...)``.  Returns
        ``("blob", BlobFileValue)`` or ``("imm", TensorValue)``."""
        if self.at("ident", "BLOBFILE"):
            return ("blob", self.parse_blobfile_value().blobFileValue)
        if self.at("["):
            # parse_scalar_list_immediate consumes its own ']'
            self.advance()
            tv = self.parse_scalar_list_immediate(dtype, len(dims) and self._total_elems(dims))
            return ("imm", tv)
        tv = self.parse_scalar_immediate(dtype)
        return ("imm", tv)

    def _total_elems(self, dims) -> int:
        n = 1
        for d in dims:
            if isinstance(d, int):
                n *= d
        return n

    def parse_blobfile_value(self) -> MIL_pb2.Value:
        self.expect("ident", "BLOBFILE")
        self.expect("(")
        self.expect("ident", "path")
        self.expect("=")
        path_val = self.parse_value()
        path = self._read_string_value(path_val)
        self.expect(",")
        self.expect("ident", "offset")
        self.expect("=")
        off_val = self.parse_value()
        offset = self._read_int_value(off_val)
        self.expect(")")
        val = MIL_pb2.Value()
        val.blobFileValue.fileName = path
        val.blobFileValue.offset = offset
        return val

    @staticmethod
    def _read_string_value(val: MIL_pb2.Value) -> str:
        tv = val.immediateValue.tensor
        if len(tv.strings.values) == 1:
            return tv.strings.values[0]
        raise ParseError("expected a single string value")

    @staticmethod
    def _read_int_value(val: MIL_pb2.Value) -> int:
        tv = val.immediateValue.tensor
        for field in ("ints", "longInts"):
            if len(getattr(tv, field).values) == 1:
                return int(getattr(tv, field).values[0])
        raise ParseError("expected a single integer value")

    # -- immediate scalar encoding ------------------------------------------

    def parse_scalar_immediate(self, dtype: int) -> MIL_pb2.TensorValue:
        tv = MIL_pb2.TensorValue()
        lit = self.parse_literal(dtype)
        self._append_literal(tv, dtype, lit)
        return tv

    def parse_scalar_list_immediate(self, dtype: int, max_elems: int) -> MIL_pb2.TensorValue:
        """Parse a flat or nested ``[...]`` literal list, flattening nested
        brackets (``[[1, 1], [1, 1]]``) into one scalar stream.  The caller
        has already consumed the opening ``[``."""
        tv = MIL_pb2.TensorValue()
        self.parse_nested_scalar_list_immediate(dtype, tv)
        self.expect("]")
        return tv

    def parse_nested_scalar_list_immediate(self, dtype: int, tv: MIL_pb2.TensorValue) -> None:
        """Append scalars from ``scalar | '[' nested ']'`` runs into *tv*."""
        while True:
            if self.at("["):
                self.advance()
                self.parse_nested_scalar_list_immediate(dtype, tv)
                self.expect("]")
            else:
                self._append_literal(tv, dtype, self.parse_literal(dtype))
            if self.at(","):
                self.advance()
                continue
            break

    def parse_literal(self, dtype: int):
        """Parse one literal (number / string / bool) for the given dtype."""
        tok = self.peek()
        if dtype == MIL_pb2.STRING:
            if tok[0] != "string":
                raise ParseError(f"expected a string literal, got {tok[1]!r}")
            self.advance()
            return tok[1]
        if dtype == MIL_pb2.BOOL:
            if tok[0] != "ident" or tok[1] not in ("true", "false"):
                raise ParseError(f"expected true/false, got {tok[1]!r}")
            self.advance()
            return tok[1] == "true"
        if tok[0] == "num":
            self.advance()
            return tok[1]
        if tok[0] == "ident" and tok[1] in ("inf", "-inf", "nan", "infinity"):
            self.advance()
            return tok[1]
        raise ParseError(f"expected a numeric literal, got {tok[1]!r} at offset {tok[2]}")

    def _append_literal(self, tv: MIL_pb2.TensorValue, dtype: int, lit) -> None:
        if dtype == MIL_pb2.STRING:
            tv.strings.values.append(lit)
        elif dtype == MIL_pb2.BOOL:
            tv.bools.values.append(lit)
        elif dtype == MIL_pb2.FLOAT32:
            tv.floats.values.append(self._to_float(lit))
        elif dtype == MIL_pb2.FLOAT64:
            tv.doubles.values.append(self._to_float(lit))
        elif dtype == MIL_pb2.FLOAT16:
            tv.bytes.values += struct.pack("<e", self._to_float(lit))
        elif dtype in (MIL_pb2.INT64, MIL_pb2.UINT64):
            tv.longInts.values.append(self._to_int(lit))
        elif dtype in (MIL_pb2.INT32, MIL_pb2.INT16, MIL_pb2.UINT16):
            tv.ints.values.append(self._to_int(lit))
        elif dtype == MIL_pb2.UINT32:
            tv.bytes.values += struct.pack("<I", self._to_int(lit))
        elif dtype == MIL_pb2.INT8:
            tv.bytes.values += struct.pack("<b", self._to_int(lit))
        elif dtype == MIL_pb2.UINT8:
            tv.bytes.values += struct.pack("<B", self._to_int(lit))
        elif dtype in _BIT_WIDTH:
            # sub-byte types are packed LSB-first into bytes
            bits = _BIT_WIDTH[dtype]
            v = self._to_int(lit)
            if v < 0 or v >= (1 << bits):
                raise ParseError(f"value {v} does not fit in {bits} bits")
            # pack into the stream's last byte
            if not tv.bytes.values or self._cur_byte_bits >= 8 - bits:
                tv.bytes.values += b"\x00"
            tv.bytes.values = tv.bytes.values[:-1] + bytes([
                tv.bytes.values[-1] | (v << self._cur_byte_bits)
            ])
            self._cur_byte_bits = (self._cur_byte_bits + bits) % 8
        else:  # pragma: no cover
            raise ParseError(f"unsupported immediate dtype {dtype}")

    def _to_float(self, lit) -> float:
        if isinstance(lit, str):
            s = lit.lower()
            if "x" in s and ("p" in s or "0x" in s):
                # hex float, possibly signed: -0x1.f38p+9
                try:
                    return float.fromhex(lit)
                except ValueError:
                    raise ParseError(f"bad hex float {lit!r}")
            if s in ("inf", "infinity"):
                return float("inf")
            if s == "-inf":
                return float("-inf")
            if s == "nan":
                return float("nan")
            return float(lit)
        return float(lit)

    def _to_int(self, lit) -> int:
        if isinstance(lit, bool):
            return int(lit)
        if isinstance(lit, int):
            return lit
        s = lit.strip()
        sign = 1
        if s.startswith(("-", "+")):
            if s[0] == "-":
                sign = -1
            s = s[1:]
        try:
            return sign * int(s, 0)
        except ValueError:
            # maybe a hex float with integer value, e.g. 0x1p+2
            return sign * int(float.fromhex(s))
    def _count_scalars(self, tv: MIL_pb2.TensorValue, dtype: int) -> Optional[int]:
        """Element count of an immediate tensor, or None if not recoverable.

        Byte-packed dtypes need the dtype to derive the count from the byte
        length; sub-byte (packed-bit) dtypes are not recoverable here.
        """
        if len(tv.strings.values):
            return len(tv.strings.values)
        if len(tv.bools.values):
            return len(tv.bools.values)
        if len(tv.ints.values):
            return len(tv.ints.values)
        if len(tv.longInts.values):
            return len(tv.longInts.values)
        if len(tv.floats.values):
            return len(tv.floats.values)
        if len(tv.doubles.values):
            return len(tv.doubles.values)
        width = {
            MIL_pb2.FLOAT16: 2,
            MIL_pb2.INT8: 1,
            MIL_pb2.UINT8: 1,
            MIL_pb2.UINT32: 4,
        }.get(dtype)
        if width and len(tv.bytes.values) % width == 0:
            return len(tv.bytes.values) // width
        return None

    def _literal_depth(self) -> int:
        """Bracket nesting depth of the value literal at the current token.

        Pure lookahead over the token stream (does not consume).  Called
        only when the stream starts with ``[``; stops at the matching close.
        """
        depth = 0
        max_depth = 0
        i = self.pos
        while i < len(self.tokens):
            kind = self.tokens[i][0]
            if kind == "[":
                depth += 1
                max_depth = max(max_depth, depth)
            elif kind == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        return max_depth
    # end of _append_literal helpers

    # sub-byte packing state (reset per immediate)
    _cur_byte_bits = 0

    # -- tuple / list / dict contents ---------------------------------------

    def parse_tuple_contents(self, vtype: MIL_pb2.ValueType) -> MIL_pb2.TupleValue:
        """Parse the tuple literal ``(v1, v2, ...)``, consuming both parens.

        A tuple VALUE is written ``tuple<...>( literal )`` — the outer parens
        belong to the value, the literal keeps its own pair.  A tuple ELEMENT
        (inside another container) is just the literal ``(v1, v2, ...)``.
        Both callers rely on this function to consume the literal's parens.
        """
        self.expect("(")
        tv = MIL_pb2.TupleValue()
        while not self.at(")"):
            tv.values.append(self.parse_element_value(vtype.tupleType.types))
            if self.at(","):
                self.advance()
        self.expect(")")
        return tv

    def parse_list_contents(self, ltype: MIL_pb2.ListType) -> MIL_pb2.ListValue:
        """Parse the list literal ``[v1, v2, ...]``, consuming both brackets.

        A list VALUE is written ``list<...>( literal )``; a list ELEMENT is
        just the literal.  Both callers rely on this to consume the brackets.
        """
        self.expect("[")
        lv = MIL_pb2.ListValue()
        while not self.at("]"):
            lv.values.append(self.parse_element_value([ltype.type]))
            if self.at(","):
                self.advance()
        self.expect("]")
        return lv

    def parse_dict_contents(self, dtype: MIL_pb2.DictionaryType) -> MIL_pb2.DictionaryValue:
        dv = MIL_pb2.DictionaryValue()
        # form: {{k1, v1}, {k2, v2}} — outer `{` is the entry list opener,
        # each entry is its own `{ key, value }` pair.
        self.expect("{")
        while not self.at("}"):
            self.expect("{")
            entry = dv.values.add()
            entry.key.CopyFrom(self.parse_element_value([dtype.keyType]))
            self.expect(",")
            entry.value.CopyFrom(self.parse_element_value([dtype.valueType]))
            self.expect("}")
            if self.at(","):
                self.advance()
        self.expect("}")  # entry-list closer
        return dv

    def parse_element_value(self, allowed_types) -> MIL_pb2.Value:
        """Parse one element of a container, which may be written in full typed
        form or as a shorthand matching the (single) allowed type."""
        tok = self.peek()
        # full typed form
        if tok[0] == "ident" and tok[1] in ("tensor", "list", "tuple", "dict", "state"):
            return self.parse_typed_value()
        if tok[0] == "ident" and tok[1] in SCALAR_DTYPES and self.tokens[self.pos + 1][0] == "(":
            return self.parse_scalar_constructor()
        # shorthand per the expected type
        expect = allowed_types[0]
        if len(allowed_types) > 1:
            # tuple element with multiple candidate types: try each in order
            for t in allowed_types:
                v = self._try_shorthand(t)
                if v is not None:
                    return v
            raise ParseError(f"cannot parse element at offset {tok[2]}")
        v = self._try_shorthand(expect)
        if v is None:
            raise ParseError(f"cannot parse element at offset {tok[2]}")
        return v

    def _try_shorthand(self, vtype: MIL_pb2.ValueType) -> Optional[MIL_pb2.Value]:
        """Parse a shorthand value for the expected ValueType, or None if the
        token stream does not match.  Does not consume tokens on failure."""
        save = self.pos
        try:
            which = vtype.WhichOneof("type")
            if which == "tensorType":
                tt = vtype.tensorType
                if self.at("["):
                    # A tensor literal's bracket nesting equals its rank.
                    # Enforcing this resolves container-element ambiguity:
                    # a nested literal like [[1,1],[1,1]] is a LIST of
                    # rank-1 tensors, never a rank-1 tensor (which would
                    # flatten it) — without this check the first of several
                    # candidate element types (e.g. FlexibleShapeInformation
                    # DefaultShapes vs RangeDims entries) greedily consumes
                    # the wrong value.
                    if self._literal_depth() != tt.rank:
                        raise ParseError(
                            f"literal depth != tensor rank {tt.rank}"
                        )
                    # parse_scalar_list_immediate consumes its own ']'
                    self.advance()
                    tv = self.parse_scalar_list_immediate(tt.dataType, 0)
                else:
                    if tt.rank != 0:
                        raise ParseError(
                            f"scalar shorthand for rank-{tt.rank} tensor"
                        )
                    tv = self.parse_scalar_immediate(tt.dataType)
                n = self._count_scalars(tv, tt.dataType)
                val = MIL_pb2.Value()
                # the shorthand literal has a CONCRETE element count —
                # replace any UNKNOWN dims in the declared type with it
                # (the runtime needs known dims, e.g. for DefaultShapes).
                val.type.CopyFrom(vtype)
                if (
                    n is not None
                    and tt.rank != 0
                    and any(d.HasField("unknown") for d in tt.dimensions)
                ):
                    vt2 = val.type.tensorType
                    fixed = False
                    for d in vt2.dimensions:
                        if d.HasField("unknown") and not fixed:
                            d.Clear()
                            d.constant.size = n
                            fixed = True
                val.immediateValue.tensor.CopyFrom(tv)
                return val
            if which == "tupleType":
                # parse_tuple_contents consumes the literal's own parens
                tup = self.parse_tuple_contents(vtype)
                val = MIL_pb2.Value()
                val.type.CopyFrom(vtype)
                val.immediateValue.tuple.CopyFrom(tup)
                return val
            if which == "listType":
                # parse_list_contents consumes the literal's own brackets
                lst = self.parse_list_contents(vtype.listType)
                val = MIL_pb2.Value()
                val.type.CopyFrom(vtype)
                # the literal has a CONCRETE element count — resolve an
                # unknown/variadic declared length to it (the runtime needs
                # known lengths for immediate list values, e.g. RangeDims).
                if vtype.listType.length.HasField("unknown"):
                    val.type.listType.length.Clear()
                    val.type.listType.length.constant.size = len(lst.values)
                val.immediateValue.list.CopyFrom(lst)
                return val
            if which == "dictionaryType":
                val = MIL_pb2.Value()
                val.type.CopyFrom(vtype)
                val.immediateValue.dictionary.CopyFrom(self.parse_dict_contents(vtype.dictionaryType))
                return val
            if which == "stateType":
                self.expect("(")
                val = MIL_pb2.Value()
                val.type.CopyFrom(vtype)
                val.immediateValue.tensor.CopyFrom(self.parse_state_contents(vtype.stateType))
                self.expect(")")
                return val
            return None
        except ParseError:
            self.pos = save
            return None

    def parse_state_contents(self, stype: MIL_pb2.StateType) -> MIL_pb2.TensorValue:
        # state<wrapped> value is a single tensor of the wrapped type
        wt = stype.wrappedType
        if wt.WhichOneof("type") != "tensorType":
            raise ParseError("state values must wrap a tensor")
        tt = wt.tensorType
        contents = self.parse_tensor_contents(tt.dataType, tt.dimensions)
        if contents[0] != "imm":
            raise ParseError("state values cannot be blob files")
        return contents[1]


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def parse_mil_text(text: str) -> MIL_pb2.Program:
    """Parse MIL text and return the ``Program`` proto."""
    return MilTextParser(text).parse_program()


def parse_mil_file(path: str) -> MIL_pb2.Program:
    """Parse a ``model.mil`` file and return the ``Program`` proto."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_mil_text(f.read())


def extract_weights_refs(program: MIL_pb2.Program) -> List[Tuple[str, int]]:
    """Return the distinct ``(fileName, maxOffset+size)`` weight files referenced
    by the program (fileName keeps the ``@model_path/`` prefix)."""
    refs: dict = {}

    def visit_value(val: MIL_pb2.Value) -> None:
        if val.HasField("blobFileValue"):
            bf = val.blobFileValue
            prev = refs.get(bf.fileName, 0)
            if bf.offset > prev:
                refs[bf.fileName] = bf.offset

    def visit_ops(ops) -> None:
        for op in ops:
            for arg in op.inputs.values():
                for b in arg.arguments:
                    if b.HasField("value"):
                        visit_value(b.value)
            for attr in op.attributes.values():
                visit_value(attr)
            for blk in op.blocks:
                visit_ops(blk.operations)

    for fn in program.functions.values():
        for blk in fn.block_specializations.values():
            visit_ops(blk.operations)
    return sorted(refs.items())


def strip_model_path(file_name: str) -> str:
    """Map a ``@model_path/weights/weight.bin`` reference to a plain path."""
    if file_name.startswith("@model_path/"):
        return file_name[len("@model_path/") :]
    return file_name
