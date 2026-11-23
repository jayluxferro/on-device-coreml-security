# Reproducibility Package — On-Device CoreML Security Analysis

Companion repository for the paper *"Harvesting Apple's Orchard: A Security
Analysis of On-Device CoreML Models Across the Apple Ecosystem."* This
repository contains the tooling and measurement records needed to reproduce the
paper's claims. It contains **no model weights, tokenizer vocabularies, or
prompt assets** — those are the platform vendor's intellectual property and are
excluded; the records reference models by name and path so a researcher with
their own device or firmware image can rebuild the corpus.

## Contents

| Directory | What it is |
|---|---|
| `harvest/` | model enumeration, the firmware-image extractor, the `.mlmodelc`→`.mlpackage` conversion pipeline, the static analyzer, the adversarial harness, the macOS-27 surface inventory, the DMG encryption sweep, and the MIL mutation harness |
| `datasets/` | the adversarial evaluation records (82,094 JSONL rows in two schemas), the static-analysis records, the dated family census, the localization-layer index, the template and prompt indexes, and the crash/mutation logs |
| `poc/` | the crash-reproduction loader and the metadata-read proofs (no bundled model) |

The manuscript source and figure generator are not duplicated here; they are
available with the paper.

## Reproduce the headline results

1. **Census / static analysis** — `datasets/STATIC_SECURITY_DATA.jsonl` is the
   1,137-record artifact inventory; the plaintext/quantized/provenance
   predicates are in `harvest/static_security.py`.
2. **Adversarial campaign** — `datasets/EXHAUSTIVE_SECURITY_DATA.jsonl` holds the
   82,094 records (77,396 `predict()` calls with a comparable output). The
   harness is `harvest/security_exhaustive.py`.
3. **Conversion** — `harvest/mlmodelc_loader.py` turns a native `.mlmodelc`
   bundle into a loadable `.mlpackage` via the public tooling;
   `harvest/mil_text_parser.py` parses the native MIL text.
4. **Crash** — `poc/load_model.swift` loads a user-supplied `.mlmodelc` through
   the public CoreML API; the `hand_gesture` bundle faults in `libBNNS` on
   affected systems. No model is bundled here.
5. **Encryption sweep** — `harvest/dmg_encryption_sweep.py` records
   `hdiutil isencrypted` over every model disk image under
   `/System/Library/AssetsV2`; results are in
   `datasets/dmg_encryption_*.json`.

## What is deliberately excluded

- Model weights (`.weights`, `.bin`, `.hwx`), MIL text, compiled `coremldata.bin`
- Tokenizer vocabularies and system-prompt texts
- IPSW firmware images and any harvested `.mlmodelc` bundles

These are the vendor's intellectual property. The datasets reference them by
name and path only.

## License

This artifact is licensed under the Creative Commons Attribution 4.0
International license (CC BY 4.0). See `LICENSE` for the full legal text. In
short: you may reuse, adapt, and redistribute it for any purpose provided you
give appropriate credit.

## Citation

Please cite the paper. A machine-readable citation (`CITATION.cff`) is
included; the arXiv DOI and URL are added to it once the preprint is
published.
