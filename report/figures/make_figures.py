"""Figures for the AI6102 team-project report.

Every number is copied from EXPERIMENTS.md (nested CV, both partitions) or the Kaggle
submission list; the confusion matrix is read from the 8B run's saved confusion CSVs.

    uv run --with matplotlib --with pandas python report/figures/make_figures.py \
        outputs/tc2/qwen3-8b-l24-full-001
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def progress() -> None:
    """CV (mean of the two partitions) and public LB for each milestone."""
    steps = [
        ("Linear\ntext+style", 0.355, None),
        ("Ettin-400M\nofficial only", 0.388, None),
        ("3-model blend\nofficial only", 0.425, None),
        ("Ettin-400M\n+ pool 45k", 0.532, 0.546),
        ("4 encoders", 0.565, 0.566),
        ("4 enc. +\nQwen3-8B", 0.575, 0.582),
        ("5-model, 400M\npool 87k", 0.590, 0.590),
        ("Qwen3-8B\nLoRA24, 212k", 0.6235, 0.599),
        ("+ test-time\ncentring", None, 0.619),
    ]
    x = np.arange(len(steps))
    cv = [s[1] for s in steps]
    lb = [s[2] for s in steps]
    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    xc = [i for i, v in enumerate(cv) if v is not None]
    ax.plot(xc, [cv[i] for i in xc], color=BLUE, lw=2, marker="o", ms=5, label="Nested CV BA (120 dev prompts)")
    xl = [i for i, v in enumerate(lb) if v is not None]
    ax.plot(xl, [lb[i] for i in xl], color=ORANGE, lw=2, marker="s", ms=5, label="Public leaderboard")
    ax.plot([7], [0.633], marker="^", ms=7, color=AQUA, ls="none", label="Locked confirmation set (30 prompts)")
    for i in range(len(steps)):
        c, l = cv[i], lb[i]
        if i == 7:  # 8B: CV above, LB below, confirmation to the right
            ax.annotate(f"{c:.3f}", (i, c), textcoords="offset points", xytext=(-14, 3), ha="center", color=INK, fontsize=6.5)
            ax.annotate(f"{l:.3f}", (i, l), textcoords="offset points", xytext=(0, -11), ha="center", color=INK, fontsize=6.5)
            ax.annotate("0.633", (i, 0.633), textcoords="offset points", xytext=(13, -2), ha="center", color=INK, fontsize=6.5)
            continue
        if c is not None and l is not None:
            hi, lo = max(c, l), min(c, l)
            ax.annotate(f"{hi:.3f}", (i, hi), textcoords="offset points", xytext=(0, 6), ha="center", color=INK, fontsize=6.5)
            ax.annotate(f"{lo:.3f}", (i, lo), textcoords="offset points", xytext=(0, -11), ha="center", color=INK, fontsize=6.5)
        elif c is not None or l is not None:
            v = c if c is not None else l
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 6), ha="center", color=INK, fontsize=6.5)
    for y, txt, dy, va in [(0.684, "unchanged public notebook (not ours): 0.684", -0.004, "top"),
                           (0.702, "top-10% cut-off: 0.702", 0.004, "bottom"),
                           (0.765, "leaderboard rank 10: 0.765", 0.004, "bottom")]:
        ax.axhline(y, color=INK2, lw=0.7, ls=(0, (4, 3)))
        ax.text(-0.35, y + dy, txt, color=INK2, fontsize=6.5, va=va)
    ax.set_xticks(x, [s[0] for s in steps], fontsize=6.5)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.32, 0.79)
    ax.set_xlim(-0.45, len(steps) - 0.55)
    ax.legend(loc="lower right", fontsize=6.5, ncols=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_progress.pdf")
    plt.close(fig)


def scaling() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.35))
    # (a) official training prompts
    n = [24, 48, 72, 96]
    a.plot(n, [0.284, 0.318, 0.342, 0.356], color=BLUE, lw=2, marker="o", ms=5)
    a.text(48, 0.318 + 0.03, "linear text+style", color=INK, fontsize=6.5, ha="center")
    a.plot([64, 96], [0.613, 0.6235], color=ORANGE, lw=2, marker="s", ms=5)
    a.axhline(0.588, color=ORANGE, lw=1, ls=(0, (4, 3)))
    a.text(25, 0.578, "Qwen3-8B, pool stage 1 only (0 official prompts)", color=INK2, fontsize=6.5, va="top")
    a.text(96, 0.6235 + 0.015, "Qwen3-8B LoRA24\n+ official stage 2", color=INK, fontsize=6.5, ha="right")
    a.set_xscale("log", base=2)
    a.set_xticks(n, [str(v) for v in n])
    a.minorticks_off()
    a.set_xlabel("Official training prompts (18 passages each)")
    a.set_ylabel("Balanced accuracy (CV)")
    a.set_ylim(0.25, 0.68)
    a.set_title("(a) Official data", fontsize=8, loc="left")
    # (b) community pool rows in stage 1
    b.plot([45_000, 87_549], [0.532, 0.5675], color=BLUE, lw=2, marker="o", ms=5)
    b.text(87_549, 0.5675 - 0.03, "Ettin-400M", color=INK, fontsize=6.5, ha="center")
    b.plot([36_000, 211_755], [0.550, 0.6235], color=ORANGE, lw=2, marker="s", ms=5)
    b.text(211_755, 0.6235 + 0.012, "Qwen3-8B LoRA24", color=INK, fontsize=6.5, ha="right")
    b.plot([36_000], [0.5075], color=AQUA, marker="^", ms=6, ls="none")
    b.text(36_000 * 1.1, 0.5075 - 0.005, "Qwen3-8B last-token", color=INK, fontsize=6.5, va="top")
    for y, txt, c in [(0.388, "Ettin-400M, no pool", BLUE), (0.396, "Qwen3-8B, no pool", ORANGE)]:
        b.axhline(y, color=c, lw=1, ls=(0, (4, 3)))
    b.text(30_000, 0.405, "both models without pool data: 0.39-0.40", color=INK2, fontsize=6.5)
    b.set_xscale("log")
    b.set_xticks([30_000, 50_000, 100_000, 200_000], ["30k", "50k", "100k", "200k"])
    b.minorticks_off()
    b.set_xlabel("Community pool rows in stage 1")
    b.set_ylim(0.36, 0.66)
    b.set_title("(b) Community pool data", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scaling.pdf")
    plt.close(fig)


def confusion(run: Path) -> None:
    m = sum(pd.read_csv(run / f"confusion_p{p}.csv", index_col=0, encoding="utf-8") for p in (0, 1)) / 2
    labels = list(m.index)
    fig, ax = plt.subplots(figsize=(3.4, 3.25))
    ax.grid(False)
    im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=0.9)
    short = [s.replace("claude-", "c-").replace("gemini-", "gem-").replace("-preview", "") for s in labels]
    ax.set_xticks(range(len(labels)), short, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(labels)), short, fontsize=5.5)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("True", fontsize=7)
    for start, size in [(0, 5), (5, 4), (9, 2), (11, 5), (16, 1), (17, 1)]:
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), size, size, fill=False, ec=ORANGE, lw=1.0))
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = m.values[i, j]
            if v >= 0.10:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=4.5,
                        color="white" if v > 0.45 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_confusion.pdf")
    plt.close(fig)


def headroom() -> None:
    rows = [
        ("Qwen3-8B LoRA24 as is", 0.6235),
        ("Claude siblings fixed", 0.672),
        ("Gemini siblings fixed", 0.697),
        ("GPT siblings fixed", 0.6995),
        ("GLM pair fixed", 0.713),
        ("GLM pair + Kimi fixed", 0.752),
        ("every sibling fixed", 0.8465),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    y = np.arange(len(rows))[::-1]
    colors = [BLUE] + [AQUA] * 3 + [ORANGE] * 2 + [INK2]
    ax.barh(y, [r[1] for r in rows], color=colors, height=0.62)
    for yi, (_, v) in zip(y, rows):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.5)
    ax.set_xlim(0.5, 0.92)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.axvline(0.765, color=INK2, lw=0.7, ls=(0, (4, 3)))
    ax.text(0.762, y[0] + 0.25, "LB rank 10", fontsize=6, color=INK2, ha="right", va="bottom")
    ax.set_xlabel("Balanced accuracy, CV oracle")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_headroom.pdf")
    plt.close(fig)


if __name__ == "__main__":
    progress()
    scaling()
    headroom()
    confusion(Path(sys.argv[1]))
    print("figures written to", OUT)
