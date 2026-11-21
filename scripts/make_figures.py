"""Figures for the README and the site. Run after run_worlds.py and run_images.py."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from luotain2.worlds import make_world, causal_ar_texture
from luotain2.stream import Stream
from luotain2.discover import Discoverer
from luotain2.grammar import term_name

OUT = "site/fig"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#b9b7b0", "#6250d6"
SEQ = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cfe0f6", "#7fb0ea", "#2a78d6", "#0d3f80"])
DIV = LinearSegmentedColormap.from_list("div", ["#c2410c", "#f4c3a8", "#f2f1ee", "#a9c8ef", "#1d5fb3"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "font.size": 10, "font.family": "DejaVu Sans"})


def offs(v, n):
    return v if v <= n // 2 else v - n


def steps_figure(world, seed=101, max_rows=6):
    trajs, law = make_world(world, seed=seed)
    tr, va, te = [Stream(x, mode="dynamics", max_lag=3) for x in trajs]
    L = Discoverer(mode="dynamics", max_lag=3, margin=32).run(tr, va)
    steps = [s for s in L.history if "corr_map" in s][:max_rows]
    fig, axes = plt.subplots(len(steps), 2, figsize=(6.4, 2.9 * len(steps)))
    axes = np.atleast_2d(axes)
    for row, s in zip(axes, steps):
        r = s["resid_frame"]
        v = np.percentile(np.abs(r), 99) or 1
        row[0].imshow(r, cmap=DIV, norm=TwoSlopeNorm(0, -v, v))
        row[0].set_title("what the law could not predict", fontsize=9, color=INK2)
        S = np.fft.fftshift(s["corr_map"])
        n = S.shape[0]
        row[1].imshow(np.sqrt(np.maximum(S, 0)), cmap=SEQ, extent=(-n // 2 - .5, n // 2 - .5, n // 2 - .5, -n // 2 - .5))
        dy, dx = s["peak"]
        row[1].scatter([dx], [dy], s=260, facecolors="none", edgecolors=ORANGE, linewidths=2)
        row[1].set_title("where it points (all offsets at once)", fontsize=9, color=INK2)
        row[1].set_xlabel("dx", fontsize=8)
        row[1].set_ylabel("dy", fontsize=8)
        row[0].set_xticks([]); row[0].set_yticks([])
        row[0].text(0.02, -0.08, f"added: {s['name']}   ({s['move']})", transform=row[0].transAxes,
                    fontsize=9, color=INK, va="top")
    truth = " ".join(f"{c:+.2f}·{term_name(t)}" for t, c in law.items())
    fig.suptitle(f"{world}: residue → next term", fontsize=12, x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.005, f"found: {L.describe(3)}\ntruth: {truth}", fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(f"{OUT}/steps_{world}.png", dpi=110)
    plt.close(fig)


def benchmark_figure(path="results/worlds.json"):
    if not os.path.exists(path):
        return
    rows = json.load(open(path))
    if os.path.exists("results/stress.json"):
        rows = rows + json.load(open("results/stress.json"))
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    methods = [("residue", "residue-guided (Luotain2)"), ("stlsq", "fixed library + STLSQ"), ("random", "random proposals")]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.0 * len(conds), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        for i, (m, label) in enumerate(methods):
            sub = [r for r in rows if r["condition"] == cond and r["method"] == m]
            if not sub:
                continue
            q = [r["test_nmse"] / r["oracle_nmse"] for r in sub]
            ex = sum(r["verdict"] == "exact" for r in sub)
            eq = sum(r["verdict"] != "exact" and 0.99 <= x <= 1.01 for r, x in zip(sub, q))
            bt = sum(r["verdict"] != "exact" and x < 0.99 for r, x in zip(sub, q))
            mi = len(sub) - ex - eq - bt
            left = 0
            for val, col in ((ex, BLUE), (eq, AQUA), (bt, VIOLET), (mi, GRAY)):
                if val:
                    ax.barh(i, val, left=left, color=col, height=0.62, edgecolor=SURF, linewidth=2)
                left += val
            ax.text(left + 0.4, i, f"{ex} exact", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([l for _, l in methods])
        ax.invert_yaxis()
        ax.set_title(cond, fontsize=10, color=INK)
        ax.set_xlabel("runs (worlds × seeds)")
        ax.set_xlim(0, 52)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, AQUA, VIOLET, GRAY)]
    fig.legend(handles, ["exact law", "equivalent law", "predicts better than the true law (not the law)", "miss"],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{OUT}/benchmark.png", dpi=120)
    plt.close(fig)


def images_figure(path="results/images.json"):
    if not os.path.exists(path):
        return
    res = json.load(open(path))
    names = [k for k, o in res.items() if "bpp_med" in o]
    names.sort(key=lambda k: res[k]["bpp_law"] - min(res[k]["bpp_med"], res[k]["bpp_ls_r2"]))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 1.2))
    for i, k in enumerate(names):
        o = res[k]
        vals = [o["bpp_med"], o["bpp_ls_r2"], o["bpp_law"]]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=2, zorder=1)
        for v, c in zip(vals, (ORANGE, GRAY, BLUE)):
            ax.scatter(v, i, s=70, color=c, zorder=3, edgecolors=SURF, linewidths=2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("bits per pixel on the held-out bottom band (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8) for c in (ORANGE, GRAY, BLUE)]
    ax.legend(handles, ["JPEG-LS median predictor", "fixed radius-2 stencil (LS)", "discovered law"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/images_bpp.png", dpi=120)
    plt.close(fig)


def stencil_figure(path="results/images.json", which=("synthetic AR texture", "page", "text", "brick")):
    if not os.path.exists(path):
        return
    res = json.load(open(path))
    which = [w for w in which if w in res]
    fig, axes = plt.subplots(1, len(which), figsize=(3.2 * len(which), 3.0))
    for ax, w in zip(np.atleast_1d(axes), which):
        o = res[w]
        R = max(4, o["reach"])
        grid = np.full((R + 1, 2 * R + 1), np.nan)
        law_terms = o["terms"]
        # recover coefficients from the law string in order of o["terms"]
        coefs = {}
        for part in o["law"].split():
            if "*" in part:
                c, t = part.split("*", 1)
                coefs[t] = float(eval(c.replace("+", "")) if "/" in c else float(c))
        for t, (dy, dx) in zip(law_terms, o["offsets"]):
            grid[R + dy, R + dx] = coefs.get(t, 0)
        v = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
        ax.imshow(grid, cmap=DIV, norm=TwoSlopeNorm(0, -v, v), extent=(-R - .5, R + .5, .5, -R - .5))
        ax.scatter([0], [0], marker="s", s=40, color=INK)
        ax.set_title(f"{w}\n{o['n_terms']} terms, reach {o['reach']} px", fontsize=9)
        ax.set_xlabel("dx"); ax.set_ylabel("dy")
    fig.suptitle("Where each image's law reads from (black square = the pixel being predicted)", fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/stencils.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1:] or ["steps", "benchmark", "images"]
    if "steps" in what:
        for w in ("wormhole", "wave", "allen_cahn", "drift"):
            steps_figure(w)
    if "benchmark" in what:
        benchmark_figure()
    if "images" in what:
        images_figure()
        stencil_figure()
