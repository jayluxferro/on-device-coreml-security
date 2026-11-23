"""
Regenerate ALL eight Paper-1 figures at print size.

Why this file was rewritten (review round 1, findings M7/m13 and the figure
data divergences): the combined-draft generator produced several figures at
12-in-wide canvas sizes whose 8--12 pt fonts scaled down to ~4--7 pt when
typeset at \\columnwidth or \\textwidth, reviewers reported legends covering
annotated values (fig2/fig4/fig7), a stale macOS count (fig6 said 10, the
verified count is 17), a greyed 0% bar in fig1 that the paper text no longer
discusses, and fig8 still carried the 47,371-evaluation subtitle after the
campaign was unified to a single 82,092-row campaign.  Figures are now rendered at
their final typeset width (7.0 in for figure*, 3.35 in for single column)
with effective font sizes >= ~6.5 pt, legends moved outside the axes, and
every data value asserted against its source file below.

Data sources (re-verified against the files, not the old draft):
  fig2_latency.pdf            docs/ONDEVICE_LATENCY_SMS_iOS26_1.md (medians
                              of 50 predictions, iOS 26.1 virtual test
                              device; load times section 2.2)
  fig3_census.pdf             docs/STATIC_SECURITY_DATA.jsonl (1,137 records;
                              platform/kind counts asserted)
  fig6_architecture_landscape paper_combined/data/firmware_model_metadata.json
                              (iOS/iPadOS 22-model sets) + model_inventory.csv
                              (macOS 17 WORKING models) -- both asserted
  fig8_adversarial_flip_rates datasets/EXHAUSTIVE_SECURITY_DATA.jsonl
                              (82,092 campaign rows / 82,085 predict()
                              calls, 77,396 with an attacked output;
                              worst-case rates asserted
                              per model, including the three models whose
                              record names carry extraction path prefixes;
                              the SourceEditor annotation is DERIVED from the
                              dataset and asserted -- R2-M4: the records show
                              18 of 21 token substitutions flipping, not the
                              stale "7/7" narrative the code/docs carried)
  fig1/fig4/fig5/fig7         benchmark values as reported in the paper's
                              Tables 3/5 (unchanged data, re-plotted for size)

Round 2 (review R2-M1): font sizes raised so every text element lands
>= ~7 pt at print scale -- for the column-width figures the tight-bbox
width was the culprit (long y-labels/titles/legends exceeded the canvas),
so those canvases were narrowed, titles/legend entries wrapped, and fonts
raised together; log-axis exponents on fig2 are replaced by plain tick
labels (the mathtext exponents were the <5 pt offenders); fig2(a)'s
value labels are coloured per condition and de-collided in display space
with connectors (see the function comment at the call site).

Run:  .venv/bin/python paper1/figures/generate_figures.py
"""
import collections
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(OUT, "..", ".."))
COLW = 3.35   # single-column render width (in); prints at \columnwidth
FULLW = 7.0   # full-text-width render width (in); prints at \textwidth

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 8.5,
    "axes.labelsize": 9.2,
    "axes.titlesize": 9.6,
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "legend.fontsize": 8.3,
    "axes.facecolor": "#fafafa",
    "figure.facecolor": "white",
})

CLR_BLUE = "#2e6b9e"
CLR_ORANGE = "#d4752e"
CLR_GREEN = "#3c9e3c"
CLR_RED = "#c0392b"
CLR_GRAY = "#888888"
CLR_PURPLE = "#7b3f9e"
CLR_TOKEN = "#1f5f8b"   # tokenized-USEN perturbations


def savef(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print(name, "saved")


# ---------------------------------------------------------------------------
# Fig 1: SMS benchmark accuracy.  Four bars only: the paper text quotes
# smsMLModel 100, basicModel_BRPT 85, mainModel_BRPT 55, mainModel_INEN 100.
# (subClassModel_INEN ran a different 4-label scheme and is not part of the
# quoted accuracies; the old greyed 0% bar is removed rather than explained.)
# ---------------------------------------------------------------------------
# R2-M1: "smsMLModel" wrapped to two lines and the axes widened (default
# margins) so the four category labels stop touching at the raised font size.
models = ["smsML\nModel", "basicModel\nBRPT", "mainModel\nBRPT", "mainModel\nINEN"]
accuracies = [100.0, 85.0, 55.0, 100.0]

fig, ax = plt.subplots(figsize=(COLW, 1.72))
fig.subplots_adjust(left=0.11, right=0.985)
bars = ax.bar(models, accuracies, color=[CLR_GREEN, CLR_ORANGE, CLR_RED, CLR_GREEN],
              edgecolor="black", linewidth=0.5, width=0.6)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{acc:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.8)
ax.set_ylabel("Accuracy (%)", fontsize=9.5)
ax.set_title("SMS classification accuracy (20-message benchmark)", fontsize=9.8)
ax.set_ylim(0, 118)
ax.tick_params(axis="x", labelsize=8.6)
ax.tick_params(axis="y", labelsize=8.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
savef(fig, "fig1_model_accuracy.pdf")

# ---------------------------------------------------------------------------
# Fig 2: on-device inference latency + load times (iOS 26.1 virtual test
# device; medians of 50 predictions at MLComputeUnitsAll).  Legends moved
# below the axes -- "lower right" overlapped value labels (review fig2).
# ---------------------------------------------------------------------------
conds = ["normal", "phishing", "phishing+ZWSP"]
lat_models = ["mainModel_INEN", "subClassModel_INEN", "mainModel_USEN",
              "subClassModel_USEN", "basicModel_BRPT"]
lat = [
    [13.060, 31.132, 52.044],   # mainModel_INEN (textClassifier)
    [15.817, 24.973, 27.414],   # subClassModel_INEN
    [3.641, 3.546, 2.435],      # mainModel_USEN (raw net, tensor input)
    [2.802, 2.390, 2.322],      # subClassModel_USEN
    [0.093, 0.158, 0.178],      # basicModel_BRPT
]
cond_colors = [CLR_GREEN, CLR_ORANGE, CLR_RED]

fig, axes = plt.subplots(1, 2, figsize=(FULLW, 2.22), gridspec_kw={"width_ratios": [2.0, 1]})
ax = axes[0]
y = np.arange(len(lat_models))
h = 0.24
for j, (cond, c) in enumerate(zip(conds, cond_colors)):
    vals = [lat[i][j] for i in range(len(lat_models))]
    bars = ax.barh(y + (j - 1) * h, vals, height=h, color=c,
                   edgecolor="black", linewidth=0.4, label=cond)
# R2-M1: value labels are coloured per condition (matching the legend) and
# de-collided in display space.  Neighbouring rows share label x-ranges
# (24.97 vs 27.41, 3.55 vs 3.64, 2.39 vs 2.32, ...), which the earlier
# fixed-offset scheme did not clear; here any pair of overlapping label
# boxes is spread half-and-half up/down, iterated to convergence, and any
# label that moved more than 2.5 pt gets a thin connector line from its bar
# tip so the association survives the shift.  (The old in-axes note on the
# NLP core was removed here -- it was the element colliding with panel (b)'s
# labels; the sentence lives in the figure caption instead.)
value_notes = []
for i in range(len(lat_models)):
    for j in range(3):
        v = lat[i][j]
        yc = i + (j - 1) * h
        t = ax.text(v * 1.07, yc, f"{v:.2f}", va="center", ha="left",
                    fontsize=8.4, color=cond_colors[j], fontweight="bold")
        value_notes.append([t, v, yc])
ax.set_xscale("log")
ax.set_xlim(0.06, 1500)
ax.set_ylim(-1.15, 5.25)


def _resolve_value_labels(ax, notes, renderer, ylim_span):
    ax.figure.canvas.draw()
    axh = ax.get_window_extent(renderer).height
    u_px = axh / ylim_span
    for _ in range(150):
        moved = False
        for a in range(len(notes)):
            for b in range(a + 1, len(notes)):
                ba = notes[a][0].get_window_extent(renderer)
                bb = notes[b][0].get_window_extent(renderer)
                # 5 px slack on the x-overlap test, and the y test also
                # treats a near-gap (< 3 px) as still needing a push: the
                # plain "boxes must overlap" test converged with 0.4-0.6 px
                # clearances (row0 of fig2a), which is ~0.3 pt printed.
                # History: 1.2 px slack once missed the "15.82|52.04"
                # cross-row pair -- after each label was pushed away from
                # its own row neighbours the two ended with a 4.0 px x-gap
                # and 13 px y-overlap, so the strict test saw "no overlap"
                # and the pair stayed merged at print size.
                if (ba.x1 > bb.x0 - 5.0 and bb.x1 > ba.x0 - 5.0
                        and ba.y1 > bb.y0 - 3.0 and bb.y1 > ba.y0 - 3.0):
                    dy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0) + 3.0
                    hi, lo = (a, b) if ba.y0 >= bb.y0 else (b, a)
                    for k, sgn in ((hi, +1.0), (lo, -1.0)):
                        t = notes[k][0]
                        t.set_y(t.get_position()[1] + sgn * dy / 2.0 / u_px)
                    moved = True
        if not moved:
            break
    for t, v, yc in notes:
        if abs(t.get_position()[1] - yc) * u_px > 2.5:
            ax.plot([v, v * 1.07], [yc, t.get_position()[1]],
                    color=t.get_color(), lw=0.6, zorder=0.5, solid_capstyle="butt")


_resolve_value_labels(ax, value_notes, fig.canvas.get_renderer(), 6.4)

if os.environ.get("DEBUG_LAYOUT"):
    _r = fig.canvas.get_renderer()
    fig.canvas.draw()
    for t, v, yc in value_notes:
        bb = t.get_window_extent(_r)
        print(f"DBG v={v:8.3f} yc={yc:5.2f} y={t.get_position()[1]:6.3f} "
              f"x0={bb.x0:7.1f} x1={bb.x1:7.1f} y0={bb.y0:7.1f} y1={bb.y1:7.1f}")
    worst = []
    for a in range(len(value_notes)):
        for b in range(a + 1, len(value_notes)):
            ba = value_notes[a][0].get_window_extent(_r)
            bb = value_notes[b][0].get_window_extent(_r)
            xg = max(ba.x0, bb.x0) - min(ba.x1, bb.x1)
            yov = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if xg < 5.5 and yov > -2.5:
                worst.append((xg, value_notes[a][1], value_notes[b][1], yov))
    for xg, va, vb, yov in sorted(worst):
        print(f"DBG-TOUCH {va} vs {vb}: xgap={xg:.2f} yoverlap={yov:.1f}")

# Plain tick labels: the default mathtext exponents (10^-1 ...) printed at
# <5 pt after inclusion scaling (review R2-M1).
ax.set_xticks([0.1, 1, 10, 100, 1000])
ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks(y)
ax.set_yticklabels(lat_models, fontsize=9)
ax.set_xlabel("Median inference latency (ms, log scale)", fontsize=9.5)
ax.set_title("(a) Inference latency, test device (median of 50)", fontsize=10)
# Anchored slightly left of centre so the three-entry legend cannot touch
# panel (b)'s legend (they overlapped after the font bump, review R2-M1).
ax.legend(loc="upper center", bbox_to_anchor=(0.46, -0.26), ncol=3,
          framealpha=0.95, fontsize=8.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
load_models = ["mainModel_INEN", "subClassModel_INEN", "basicModel_BRPT"]
cold = [327.4, 383.6, 21.1]
warm = [6.2, 9.8, 2.6]
yy = np.arange(len(load_models))
h2 = 0.32
b1 = ax.barh(yy + h2 / 1.6, cold, height=h2, color=CLR_PURPLE,
             edgecolor="black", linewidth=0.4, label="cold load")
b2 = ax.barh(yy - h2 / 1.6, warm, height=h2, color=CLR_BLUE,
             edgecolor="black", linewidth=0.4, label="warm reload")
for bars, vals, c in ((b1, cold, CLR_PURPLE), (b2, warm, CLR_BLUE)):
    for bar, v in zip(bars, vals):
        ax.text(v * 1.09, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                va="center", fontsize=8.4, color=c, fontweight="bold")
ax.set_xscale("log")
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(["1", "10", "100", "1000"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks(yy)
ax.set_yticklabels(load_models, fontsize=9)
ax.set_xlim(1, 2000)
ax.set_xlabel("Model load time (ms, log scale)", fontsize=9.5)
ax.set_title("(b) Cold vs. warm load", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
          framealpha=0.95, fontsize=8.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(wspace=0.35, bottom=0.3)
savef(fig, "fig2_latency.pdf")

# ---------------------------------------------------------------------------
# Fig 3: static-corpus census + specification-version histogram, straight
# from docs/STATIC_SECURITY_DATA.jsonl with count assertions (the old
# combined-draft census carried stale platform totals).
# ---------------------------------------------------------------------------
recs = [json.loads(l) for l in open(os.path.join(ROOT, "docs", "STATIC_SECURITY_DATA.jsonl"))]
assert len(recs) == 1137, len(recs)

# R2-M1 follow-up: with the 8.3 pt labels, the earlier two-line tick
# strings ("iOS 26.1\ntest device", ...) overlapped vertically -- 8 rows in
# a 1.9 in canvas leave ~18 px of pitch for ~23 px of two-line text.  The
# labels are single-line now; the "iPhone X" detail lives in the caption.
plat_order = [
    ("ios261_live", "iOS 26.1 test device"),
    ("ios", "iOS 26.0 firmware"),
    ("ipados", "iPadOS 26.0.1 firmware"),
    ("ios261", "iOS 26.1 firmware adds"),
    ("tvos", "tvOS 26.6 firmware"),
    ("iOS12legacy", "iOS 16.7-era device harvest"),
    ("ios12", "iOS 12.5.8 firmware (in-depth)"),
    ("macos", "macOS 26.5 firmware"),
]
kind_order = ["mlmodel", "mlmodelc_dir", "mlmodelc_zip", "loose_dat"]
kind_label = {
    "mlmodel": ".mlmodel protobuf",
    "mlmodelc_dir": ".mlmodelc bundle (dir)",
    "mlmodelc_zip": ".mlmodelc bundle (zip)",
    "loose_dat": "loose .dat (GBDT)",
}
kind_color = {
    "mlmodel": CLR_GREEN,
    "mlmodelc_dir": CLR_ORANGE,
    "mlmodelc_zip": CLR_RED,
    "loose_dat": CLR_PURPLE,
}

counts = {p: collections.Counter() for p, _ in plat_order}
for r in recs:
    counts[r["platform"]][r["kind"]] += 1

plat_totals = {p: sum(counts[p].values()) for p, _ in plat_order}
assert plat_totals == {"ios261_live": 463, "ios": 248, "ipados": 253,
                       "ios261": 19, "tvos": 102, "iOS12legacy": 30,
                       "ios12": 5, "macos": 17}, plat_totals
kind_totals = collections.Counter(r["kind"] for r in recs)
assert kind_totals == {"mlmodel": 98, "mlmodelc_dir": 887,
                       "mlmodelc_zip": 22, "loose_dat": 130}, kind_totals

fig, axes = plt.subplots(1, 2, figsize=(FULLW, 1.9),
                         gridspec_kw={"width_ratios": [1.55, 1]})
ax = axes[0]
y = np.arange(len(plat_order))
left = np.zeros(len(plat_order))
for k in kind_order:
    vals = np.array([counts[p].get(k, 0) for p, _ in plat_order], dtype=float)
    ax.barh(y, vals, left=left, height=0.6, color=kind_color[k],
            edgecolor="black", linewidth=0.4, label=kind_label[k])
    left += vals
for i, total in enumerate(left):
    ax.text(total + 8, i, f"{int(total)}", va="center", fontsize=8.3, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([lbl for _, lbl in plat_order], fontsize=8.3)
ax.set_xlabel("CoreML artifacts cataloged (static corpus, n = 1,137)", fontsize=9.2)
ax.set_title("(a) Corpus by platform and artifact kind", fontsize=9.6)
ax.set_xlim(0, 560)
ax.legend(loc="upper right", framealpha=0.95, fontsize=8.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

spec_hist = collections.Counter(("v%d" % r["spec_version"]) if r.get("spec_version") else "unk"
                                for r in recs)
spec_keys = ["v%d" % i for i in range(1, 10)] + ["unk"]
spec_vals = [spec_hist.get(k, 0) for k in spec_keys]
ax = axes[1]
# The "unk" bucket gets a 0.6-slot gap: at the tick-label font size the
# "v9" and "unk" tick strings sat ~1.4 pt apart on the page.  The gap marks
# it as "not a version" rather than a tenth value.
spec_pos = list(range(9)) + [9.6]
bars = ax.bar(spec_pos, spec_vals, color=[CLR_BLUE] * 9 + [CLR_GRAY],
              edgecolor="black", linewidth=0.4)
for bar, v in zip(bars, spec_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 5, f"{v}", ha="center",
            fontsize=7.8, fontweight="bold")
ax.set_xticks(spec_pos)
ax.set_xticklabels(spec_keys)
ax.set_ylabel("Artifacts", fontsize=9.2)
ax.set_xlabel("CoreML specification version", fontsize=9.2)
ax.set_title("(b) Specification versions", fontsize=9.6)
ax.set_ylim(0, 400)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(wspace=0.32, bottom=0.13)
savef(fig, "fig3_census.pdf")

# ---------------------------------------------------------------------------
# Fig 4: per-class accuracy of the locale variants (20-message benchmark).
# Legend moved below the axes (review: it covered mainModel_BRPT's 100%
# transactional bar).
# ---------------------------------------------------------------------------
locale_labels = ["Promotional", "Transactional"]
brpt_basic = [90.0, 80.0]
brpt_main = [10.0, 100.0]
inen_main = [100.0, 100.0]

# R2-M1: canvas narrowed from 3.35 in; with the raised fonts the one-line
# 50-char title alone was wider than the canvas, its tight-bbox overflow
# downscaled the whole figure below \columnwidth (effective min 5.9 pt).
fig, ax = plt.subplots(figsize=(3.0, 1.58))
x = np.arange(len(locale_labels))
width = 0.22
b1 = ax.bar(x - width, brpt_basic, width, label="basicModel_BRPT",
            color=CLR_ORANGE, edgecolor="black", linewidth=0.5)
b2 = ax.bar(x, brpt_main, width, label="mainModel_BRPT",
            color=CLR_RED, edgecolor="black", linewidth=0.5)
b3 = ax.bar(x + width, inen_main, width, label="mainModel_INEN",
            color=CLR_GREEN, edgecolor="black", linewidth=0.5)
# R2-M1: the two 100% labels in the Transactional group collided ("100%/100%")
# at the raised font size; equal-height neighbours are separated horizontally
# in display space.
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_lbls = []
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        t = ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=8.7,
                    fontweight="bold")
        _lbls.append(t)
_px_per_unit = (ax.get_window_extent(_rend).width
                / (ax.get_xlim()[1] - ax.get_xlim()[0]))
for _ in range(20):
    _moved = False
    for _a in range(len(_lbls)):
        for _b in range(_a + 1, len(_lbls)):
            _ba = _lbls[_a].get_window_extent(_rend)
            _bb = _lbls[_b].get_window_extent(_rend)
            # 1.2 px slack: after a spread pass the boxes can end exactly
            # abutting ("100%100%" merged at print size with the old strict
            # test); the extra 3.0 px below leaves a printable gap.
            if (_ba.x1 > _bb.x0 - 1.2 and _bb.x1 > _ba.x0 - 1.2
                    and _ba.y1 > _bb.y0 and _bb.y1 > _ba.y0):
                _dx = min(_ba.x1, _bb.x1) - max(_ba.x0, _bb.x0) + 3.0
                _left, _right = ((_a, _b) if _ba.x0 <= _bb.x0 else (_b, _a))
                for _k, _sgn in ((_left, -1.0), (_right, +1.0)):
                    _t = _lbls[_k]
                    _t.set_x(_t.get_position()[0] + _sgn * _dx / 2.0 / _px_per_unit)
                _moved = True
    if not _moved:
        break
ax.set_xticks(x)
ax.set_xticklabels(locale_labels, fontsize=8.5)
ax.set_ylabel("Accuracy (%)", fontsize=9.5)
ax.set_title("Per-class accuracy:\nBRPT vs. INEN locale variants", fontsize=9.8)
# R2-M1: ncol 3 -> 2.  At 8.3 pt the three-across legend box was 317 pt
# wide (wider than the 216 pt canvas), and its two-sided overflow forced a
# 0.71 inclusion scale that printed the 8.3 pt text at 5.9 pt.
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
          framealpha=0.95, fontsize=8.3)
ax.set_ylim(0, 118)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
savef(fig, "fig4_locale_comparison.pdf")

# ---------------------------------------------------------------------------
# Fig 5: macOS landscape.  (a) the 17 fully inspected .mlmodel protobufs by
# family (PowerUI 7, TuriCore 4, MCCKit 3, CreateML 2, SourceEditor 1);
# (b) the 209 .mlmodelc bundles in the released catalog (read live from
# models/firmware/macos_mlmodelc_catalog.csv, asserted, so this panel can
# never drift from the record again).  Y labels are single-line to stop
# the collisions the review flagged.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(FULLW, 1.76))
families = ["PowerUI (power mgmt)", "TuriCore (style transfer)", "MCCKit (mail)",
            "CreateML (audio+skeleton)", "SourceEditor (code)"]
family_counts = [7, 4, 3, 2, 1]
ax = axes[0]
bars = ax.barh(families, family_counts, height=0.55,
               color=[CLR_ORANGE, CLR_PURPLE, CLR_BLUE, CLR_GREEN, CLR_RED],
               edgecolor="black", linewidth=0.5)
for bar, count in zip(bars, family_counts):
    yc = bar.get_y() + bar.get_height() / 2
    if bar.get_width() / 8.7 > 0.6:   # PowerUI (7/8.7): outside label would
        # sit in the inter-panel gap, where panel (b)'s unclipped y labels
        # ("treeEnsemble regressor") reach -- caught printing on top of it
        # at 600 dpi.  Draw it inside the bar instead, white and right-
        # aligned, clear of anything in panel (b).
        ax.text(bar.get_width() - 0.15, yc, f"{count}", va="center",
                ha="right", fontsize=8.3, fontweight="bold", color="white")
    else:
        ax.text(bar.get_width() + 0.12, yc, f"{count}", va="center",
                fontsize=8.3, fontweight="bold")
ax.set_xlim(0, 8.7)
ax.set_xlabel("Fully-inspected models", fontsize=9.2)
ax.set_title("(a) macOS .mlmodel protobufs (17, all loadable)", fontsize=9.6)
ax.tick_params(axis="y", labelsize=8.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ml_types = ["mlProgram (MIL)", "neuralNetwork", "pipeline classifier",
            "treeEnsemble classifier", "glmClassifier", "pipeline regressor",
            "textClassifier", "pipeline (other)", "neuralNetwork classifier",
            "scaler", "treeEnsemble regressor", "imageClassifier"]
# Read the released catalog directly; assert before plotting.
import csv as _csv
CATALOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "models", "firmware", "macos_mlmodelc_catalog.csv")
_rows = list(_csv.DictReader(open(CATALOG)))
_assert_msg = f"catalog rows changed: {len(_rows)} != 209"
assert len(_rows) == 209, _assert_msg
_short = {
    "MLModelType_mlProgram": "mlProgram (MIL)",
    "MLModelType_neuralNetwork": "neuralNetwork",
    "MLModelType_pipelineClassifier": "pipeline classifier",
    "MLModelType_treeEnsembleClassifier": "treeEnsemble classifier",
    "MLModelType_glmClassifier": "glmClassifier",
    "MLModelType_pipelineRegressor": "pipeline regressor",
    "MLModelType_textClassifier": "textClassifier",
    "MLModelType_pipeline": "pipeline (other)",
    "MLModelType_neuralNetworkClassifier": "neuralNetwork classifier",
    "MLModelType_scaler": "scaler",
    "MLModelType_treeEnsembleRegressor": "treeEnsemble regressor",
    "MLModelType_imageClassifier": "imageClassifier",
}
_type_counts = collections.Counter(r["model_type"] for r in _rows)
ml_types = [_short[k] for k in _short]
ml_counts = [_type_counts.get(k, 0) for k in _short]
_expected = [76, 54, 18, 16, 15, 10, 7, 5, 3, 2, 2, 1]
assert ml_counts == _expected, f"fig5 counts drifted: {ml_counts} != {_expected}"
ax = axes[1]
colors = [CLR_PURPLE if "mlProgram" in t else (CLR_BLUE if "neuralNetwork" in t else CLR_GRAY)
          for t in ml_types]
bars = ax.barh(ml_types[::-1], ml_counts[::-1], height=0.6, color=colors[::-1],
               edgecolor="black", linewidth=0.4)
for bar, count in zip(bars, ml_counts[::-1]):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{count}", va="center", fontsize=8.3)
ax.set_xlim(0, 86)
ax.set_xlabel("mlmodelc bundles", fontsize=9.2)
ax.set_title("(b) macOS .mlmodelc types (n = 209)", fontsize=9.6)
ax.tick_params(axis="y", labelsize=8.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# wspace 0.42 -> 0.48: panel (b)'s longest y label ("treeEnsemble
# regressor") is not clipped and reaches toward panel (a).  It collided
# with PowerUI's value label ("7"); that label now sits inside its bar, so
# the remaining clearance only needs to keep the label off panel (a)'s
# right spine.
fig.subplots_adjust(wspace=0.48)
savef(fig, "fig5_macos_models.pdf")

# ---------------------------------------------------------------------------
# Fig 6: architecture landscape of the 61 loadable .mlmodel files from the
# initial firmware/macOS harvest.  Counts asserted against
# paper_combined/data/firmware_model_metadata.json (iOS/iPadOS, 22 each) and
# model_inventory.csv (macOS WORKING, 17).  The old combined-draft figure
# plotted macOS = 10.
# ---------------------------------------------------------------------------
fw = json.load(open(os.path.join(ROOT, "paper_combined", "data", "firmware_model_metadata.json")))


def arch_short(mt):
    # case-insensitive: the inventory CSV stores "NeuralNetwork"/"pipeline*"
    # in mixed case while the firmware metadata uses camelCase prototypes.
    m = str(mt).lower()
    if "mlprogram" in m:
        return "mlProgram"
    if "textclassifier" in m:
        return "textClassifier"
    if "treeensemble" in m:
        return "treeEnsemble"
    if "pipeline" in m:
        return "pipeline"
    if "neuralnetwork" in m:
        return "neuralNetwork"
    return mt


ios_rows = [r for r in fw if r["platform_key"] == "ios"]
ipad_rows = [r for r in fw if r["platform_key"] == "ipados"]
assert len(ios_rows) == 22 and len(ipad_rows) == 22, (len(ios_rows), len(ipad_rows))

mac_working = [r for r in csv.DictReader(open(os.path.join(
    ROOT, "paper_combined", "data", "model_inventory.csv")))
    if r["platform"] == "macOS" and r["status"] == "WORKING"]
assert len(mac_working) == 17, len(mac_working)


def dist(rows, key="model_type"):
    d = collections.Counter(arch_short(r[key]) for r in rows)
    return d


d_ios = dist(ios_rows)
d_ipad = dist(ipad_rows)
d_mac = dist([{"model_type": r["model_type"]} for r in mac_working])
assert d_ios == {"treeEnsemble": 7, "neuralNetwork": 6, "textClassifier": 5,
                 "pipeline": 3, "mlProgram": 1}, d_ios
assert d_ipad == d_ios, d_ipad
assert d_mac == {"treeEnsemble": 5, "neuralNetwork": 7, "mlProgram": 3,
                 "pipeline": 2}, d_mac

platform_labels = ["iOS 26.0", "iPadOS 26.0.1", "macOS 26.5"]
arch_order = ["neuralNetwork", "treeEnsemble", "textClassifier", "pipeline", "mlProgram"]
arch_colors = {"textClassifier": CLR_RED, "neuralNetwork": CLR_BLUE,
               "treeEnsemble": CLR_ORANGE, "pipeline": CLR_GREEN, "mlProgram": CLR_PURPLE}
dists = [d_ios, d_ipad, d_mac]

fig, ax = plt.subplots(figsize=(COLW, 1.72))
bottom = np.zeros(3)
for a in arch_order:
    vals = [dists[i].get(a, 0) for i in range(3)]
    ax.bar(platform_labels, vals, bottom=bottom, color=arch_colors[a],
           edgecolor="black", linewidth=0.4, label=a)
    bottom += np.array(vals)
for i, total in enumerate(bottom):
    ax.text(i, total + 0.6, f"{int(total)}", ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("Loadable .mlmodel files", fontsize=9.5)
# Title trimmed: the 9.8 pt one-line version was wider than the axes, and
# its first word overlapped the rotated ylabel's top at print size (found
# at 250 dpi; "Architecture" printed across "files").  The full phrasing
# stays in the paper caption.
ax.set_title("Architecture of the 61 loadable models", fontsize=9.8)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
          framealpha=0.95, fontsize=8.3)
ax.set_ylim(0, 27)
ax.tick_params(axis="x", labelsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
savef(fig, "fig6_architecture_landscape.pdf")

# ---------------------------------------------------------------------------
# Fig 7: SMS locale variants, architectural divergence.  Short bar-end
# labels only ("640-d"/"256-d"); the old right-side annotations and the
# side legend are replaced by a below-axes legend.
# ---------------------------------------------------------------------------
sms_models = [
    ("mainModel_INEN", "text", 640),
    ("subClassModel_INEN", "text", 640),
    ("mainModel_BRPT", "text", 640),
    ("basicModel_BRPT", "text", 640),
    ("mainModel_USEN", "raw", 256),
    ("subClassModel_USEN", "raw", 256),
]
# R2-M1: canvas narrowed from 3.35 in and the title wrapped to two lines.
# At 3.35 in the six long y-labels (9 pt) and the one-line 52-char title
# pushed the tight-bbox width to 344 pt, downscaling the 8.5 pt legend to
# 5.8 pt when typeset (review's effective-size measurement).
fig, ax = plt.subplots(figsize=(2.7, 1.70))
y_pos = np.arange(len(sms_models))
colors = [CLR_RED if fam == "text" else CLR_TOKEN for _, fam, _ in sms_models]
bars = ax.barh(y_pos, [d for *_, d in sms_models], height=0.55, color=colors,
               edgecolor="black", linewidth=0.5)
for (*_, dim), bar in zip(sms_models, bars):
    ax.text(dim + 14, bar.get_y() + bar.get_height() / 2, f"{dim}-d",
            va="center", fontsize=8.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([n for n, *_ in sms_models], fontsize=9)
ax.set_xlabel("Feature dimensionality", fontsize=9.5)
ax.set_title("SMS filter locale variants:\narchitectural divergence", fontsize=10)
ax.set_xlim(0, 820)
ax.tick_params(axis="x", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

from matplotlib.patches import Patch
# Anchor pushed from -0.16 to -0.24 after the height shave: at -0.16 the
# legend box top cut through the xlabel ("Feature dimensionality").
# R2-M1: legend font 5.8 -> 8.5 with the entries wrapped to two lines -- the
# one-line versions at 8.5 pt were wider than the canvas and would have been
# downscaled below \columnwidth in LaTeX.
ax.legend(handles=[
    Patch(color=CLR_RED, label="textClassifier (640-d, 9-layer TCN)"),
    Patch(color=CLR_TOKEN, label="neuralNetwork (256-d, 21 layers)"),
], loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=1, framealpha=0.95, fontsize=8.5)
savef(fig, "fig7_locale_architecture.pdf")

# ---------------------------------------------------------------------------
# Fig 8: worst-case adversarial flip rate per model across the COMBINED
# campaign (82,092 campaign rows: 82,085 predict() calls, of which 77,396
# carry an attacked output).  Every
# value is asserted against the worst attack found in
# datasets/EXHAUSTIVE_SECURITY_DATA.jsonl -- including the three models
# whose record names carry extraction-directory prefixes (SourceEditor,
# mac_routine_eng, main_nyc), which the old figure's manual table had
# drifted on (main_nyc plotted 90, true worst case 100 under all-zero/
# all-one feature probes).
# ---------------------------------------------------------------------------
# The file carries 82,094 JSONL records in two schemas (paper appendix
# (iii)): 82,092 campaign rows -- distinguished by the `flipped` key, which
# the 2 supplementary resnet-16 style-0 rows lack -- plus those 2
# supplementary rows.  Within the campaign rows, 7 are `__population__`
# per-path summary rows and 4,689 carry no attacked output (4,680
# crema/TFModel battery calls, 5 clean-repeat sanity rows, 4 failed input-
# validation runs).  Only output-carrying records can enter a flip rate;
# the asserts pin the accounting so regeneration cannot count summary,
# outputless, or supplementary rows as evaluations (review round 5, F1).
_adv_raw = [json.loads(l) for l in open(os.path.join(ROOT, "datasets", "EXHAUSTIVE_SECURITY_DATA.jsonl"))]
adv_recs = [r for r in _adv_raw if "flipped" in r]
_pop_rows = [r for r in adv_recs if r["attack_type"] == "__population__"]
_eval_rows = [r for r in adv_recs if r["attack_type"] != "__population__"]
_scored_rows = [r for r in _eval_rows if r.get("attacked_output") not in (None, "", "None")]
assert len(_adv_raw) == 82094, len(_adv_raw)          # file lines, two schemas
assert len(adv_recs) == 82092, len(adv_recs)          # campaign rows
assert len(_pop_rows) == 7, len(_pop_rows)            # distribution summaries, not evaluations
assert len(_eval_rows) == 82085, len(_eval_rows)      # predict() calls
assert len(_scored_rows) == 77396, len(_scored_rows)  # calls with an attacked output

# R2-M4 / N1: the SourceEditor annotation is derived from the records, not
# copied from the old "7/7" narrative.  The dataset holds 21 single-token
# substitution probes (18 flip) and 6 sequence-length probes (all 6 flip);
# the label below is built from those counts and asserted, so regeneration
# cannot reintroduce "7/7".
SE_NAME = "System_Library_AssetsV2_com_apple_MobileAsset_SourceEditorAssets_110f1"
_se = [r for r in adv_recs if r["name"] == SE_NAME]
_se_tok = [r for r in _se if r["attack_type"].startswith("token_replace")]
_se_seq = [r for r in _se if r["attack_type"].startswith("sequence_len")]
_se_tflip = sum(1 for r in _se_tok if r["flipped"])
assert (_se_tflip, len(_se_tok)) == (18, 21), (_se_tflip, len(_se_tok))
assert len(_se_seq) == 6 and all(r["flipped"] for r in _se_seq), len(_se_seq)
SE_LABEL = "SourceEditor (%d/%d)" % (_se_tflip, len(_se_tok))

# figure label -> exact record name in the dataset
RECORD_NAME = {
    SE_LABEL: SE_NAME,
    "crepe (pitch)": "crepe",
    "main_nyc": "main_nyc2d88ffc_44000-V53-V54",
    "ghent_classifier": "ghent_classifier",
    "BlackPearlTS (mail)": "BlackPearlTS",
    "mainModel_USEN": "mainModel_USEN",
    "test_ghent_classifier": "test_ghent_classifier",
    "mac_routine_eng": "assets_COREOS_CHARGE_PREDICTION_mac_routine_eng",
    "mainModel_INEN": "mainModel_INEN",
    "deoc_model_iphone": "assets_COREOS_CHARGE_PREDICTION_deoc_model_iphone",
    "deoc_series": "assets_COREOS_CHARGE_PREDICTION_deoc_series",
    "mainModel_BRPT": "mainModel_BRPT",
    "subClassModel_INEN": "subClassModel_INEN",
}
adv_expected = {
    SE_LABEL: 100.0,
    "crepe (pitch)": 100.0,
    "main_nyc": 100.0,
    "ghent_classifier": 80.0,
    "BlackPearlTS (mail)": 77.5,
    "mainModel_USEN": 70.0,
    "test_ghent_classifier": 67.0,
    "mac_routine_eng": 67.0,
    "mainModel_INEN": 65.0,
    "deoc_model_iphone": 61.0,
    "deoc_series": 52.0,
    "mainModel_BRPT": 55.0,
    "subClassModel_INEN": 50.0,
}

campaign = collections.defaultdict(lambda: [0, 0])   # (name, attack) -> [n, flips]
for r in adv_recs:
    if r["flipped"] is None or r["error"]:
        continue
    k = (r["name"], r["attack_type"])
    campaign[k][0] += 1
    campaign[k][1] += 1 if r["flipped"] else 0

worst = {}
for label in RECORD_NAME:
    rec_name = RECORD_NAME[label]
    rates = [100.0 * fl / tot for (n, _), (tot, fl) in campaign.items()
             if n == rec_name and tot > 0]
    assert rates, label
    worst[label] = max(rates)
mismatch = {k: (worst[k], v) for k, v in adv_expected.items() if abs(worst[k] - v) > 1e-9}
assert not mismatch, mismatch

FAM_TEXT = CLR_RED       # raw-text Unicode attacks
FAM_TOKEN = CLR_TOKEN    # tokenized USEN
FAM_FEAT = CLR_ORANGE    # feature-space attacks on tabular models
FAM_CODE = CLR_GREEN     # code completion
FAM_AUDIO = CLR_PURPLE   # audio

families = {
    SE_LABEL: FAM_CODE,
    "crepe (pitch)": FAM_AUDIO,
    "main_nyc": FAM_FEAT,
    "ghent_classifier": FAM_FEAT,
    "BlackPearlTS (mail)": FAM_FEAT,
    "mainModel_USEN": FAM_TOKEN,
    "test_ghent_classifier": FAM_FEAT,
    "mac_routine_eng": FAM_FEAT,
    "mainModel_INEN": FAM_TEXT,
    "deoc_model_iphone": FAM_FEAT,
    "deoc_series": FAM_FEAT,
    "mainModel_BRPT": FAM_TEXT,
    "subClassModel_INEN": FAM_TEXT,
}
order = sorted(adv_expected.items(), key=lambda kv: -kv[1])
# stable order within equal rates: keep the sorted() instance order (by label)

# R2-M1: canvas narrowed from 3.4 in and the title/legend entries wrapped.
# The 20-char y-labels ("test_ghent_classifier", 9 pt) overflow the left
# canvas edge; at 3.4 in that overflow made the tight width 324 pt and the
# 8.3 pt legend printed at 6.15 pt.
fig, ax = plt.subplots(figsize=(2.6, 2.0))
y = np.arange(len(order))
bars = ax.barh(y, [v for _, v in order], height=0.62,
               color=[families[k] for k, _ in order],
               edgecolor="black", linewidth=0.5)
for bar, (_, v) in zip(bars, order):
    ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=9.2, fontweight="bold")
# main_nyc's worst case is a degenerate-model result (constant-feature probes
# flip every prediction), not a stealthy perturbation; say so in the bar.
nyc_idx = [k for k, _ in order].index("main_nyc")
ax.text(2.0, nyc_idx, "all-zero / all-one features", va="center", ha="left",
        fontsize=9, color="white", fontweight="bold")
ax.invert_yaxis()
ax.set_yticks(y)
ax.set_yticklabels([k for k, _ in order], fontsize=9)
ax.set_xlabel("Worst-case prediction-flip rate (%)", fontsize=9.5)
# R5-F1: title wrapped to three lines so the longer accounting
# ("77,396 with output") does not widen the tight bbox past the
# round-2 budget (the typeset scale, and with it every font on the
# canvas, shrinks with the tight width).
ax.set_title("Worst-case adversarial flip rate\n"
             "per model (77,396 calls\n"
             "with output)", fontsize=10)
ax.set_xlim(0, 112)
ax.tick_params(axis="x", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Deeper anchor than the original -0.10: the legend top edge overlapped the
# xlabel after the height shave.  ncol stays 1 -- a 2-column legend gets
# wider than the axes and would be downscaled below \columnwidth in LaTeX.
# R2-M1: the two longest entries are wrapped so the legend box stays inside
# the canvas.
ax.legend(handles=[
    Patch(color=FAM_CODE, label="code completion (token substitution)"),
    Patch(color=FAM_AUDIO, label="audio, extreme feature values"),
    Patch(color=FAM_FEAT, label="tabular: dropout / scaling / extremes"),
    Patch(color=FAM_TOKEN, label="tokenized SMS classifier"),
    Patch(color=FAM_TEXT, label="raw-text Unicode injection"),
], loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1,
    framealpha=0.95, fontsize=8.5)
savef(fig, "fig8_adversarial_flip_rates.pdf")

print("\nAll figures regenerated.")
