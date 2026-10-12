"""Static HTML report for an experiment: one self-contained file with embedded figures."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical slots (light mode) and neutrals.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRAY, INK, INK2 = "#9a9893", "#0b0b0b", "#52514e"
SERIES = [ORANGE, AQUA, VIOLET, "#e87ba4", "#008300", "#e34948"]
SEQ_BLUES = ["#b9d4f3", "#86b4ea", "#5594df", "#2a78d6", "#1d5aa6", "#123d73"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 110,
})


def _grid(n: int, ncols: int, w: float, h: float, **kw):
    """Small multiples wrapped into rows; unused axes are hidden."""
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False, **kw)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_effects(res: dict) -> str:
    """Accuracy drop per perturbation: targeted vs size-matched random control, per network."""
    names = [p["name"] for p in res["perturbations"]]
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(names), 3.2))
    for i, name in enumerate(names):
        d_t, d_c = [], []
        for s in res["seeds"].values():
            base = s["intact"]["accuracy"]
            r = s["perturbations"][name]
            d_t.append(100 * (base - r["perturbed"]["accuracy"]))
            d_c.append(100 * (base - r["control"]["accuracy"]))
        for x0, vals, col, lab in ((i - 0.18, d_t, ORANGE, "Targeted"), (i + 0.18, d_c, GRAY, "Random control")):
            ax.bar(x0, np.mean(vals), width=0.32, color=col, label=lab if i == 0 else None, zorder=2)
            ax.scatter(np.full(len(vals), x0), vals, s=14, color=INK, alpha=0.6, zorder=3, linewidths=0)
        for a, b in zip(d_t, d_c):
            ax.plot([i - 0.18, i + 0.18], [a, b], color=INK, alpha=0.2, linewidth=1, zorder=3)
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set_xticks(range(len(names)), names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy drop (percentage points)")
    ax.legend(loc="upper left")
    ax.set_title("Effect of each perturbation, one dot per network", loc="left", color=INK)
    return _png(fig)


def fig_psychometric(res: dict) -> str:
    names = [p["name"] for p in res["perturbations"]]
    fig, axs = _grid(len(names), 3, 3.4, 3.0, sharey=True)
    seeds = list(res["seeds"].values())
    x = np.array(seeds[0]["intact"]["psychometric"]["signed_coh"])
    intact = np.mean([s["intact"]["psychometric"]["p_choice_2"] for s in seeds], 0)
    for ax, name in zip(axs, names):
        pert = np.mean([s["perturbations"][name]["perturbed"]["psychometric"]["p_choice_2"] for s in seeds], 0)
        ctrl = np.mean([s["perturbations"][name]["control"]["psychometric"]["p_choice_2"] for s in seeds], 0)
        ax.plot(x, intact, color=BLUE, marker="o", markersize=4, label="Intact")
        ax.plot(x, ctrl, color=GRAY, marker="o", markersize=4, label="Random control")
        ax.plot(x, pert, color=ORANGE, marker="o", markersize=4, label="Targeted")
        ax.set_title(name, loc="left", color=INK)
        ax.set_xlabel("Signed coherence (%)")
        ax.set_ylim(-0.03, 1.03)
    for ax in axs[::3]:
        ax.set_ylabel("P(choose right)")
    axs[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _png(fig)


def fig_rates(res: dict) -> str:
    """Population-mean E and I rates, intact vs each perturbation (example network).

    One panel per perturbation; solid = excitatory mean, dashed = inhibitory mean."""
    ex = res["example"]
    is_exc = np.array(ex["is_exc"])
    names = [k for k in ex["rates"] if k != "intact"]
    dt = res["dt"]
    intact = np.array(ex["rates"]["intact"])
    t = np.arange(intact.shape[0]) * dt
    fig, axs = _grid(len(names), 3, 3.4, 2.8, sharex=True, sharey=True)
    for ax, name in zip(axs, names):
        r = np.array(ex["rates"][name])
        for data, col, lab in ((intact, BLUE, "Intact"), (r, ORANGE, "Perturbed")):
            ax.plot(t, data[:, is_exc].mean(1), color=col, label=f"{lab}, E")
            ax.plot(t, data[:, ~is_exc].mean(1), color=col, linestyle="--", linewidth=1.5, label=f"{lab}, I")
        for per in ("stimulus", "decision"):
            s0, e0 = res["periods"][per]
            ax.axvspan(s0 * dt, e0 * dt, color="#f1f0ed", zorder=0)
        ax.set_title(name, loc="left", color=INK)
        ax.set_xlabel("Time in trial (ms)")
    for ax in axs[::3]:
        ax.set_ylabel("Mean rate (a.u.)")
    axs[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return _png(fig)


STATE_COLORS = {"hungry": ORANGE, "thirsty": BLUE, "sated": GRAY}
CUE_STYLES = {"food": "-", "water": "--", "neutral": ":"}


def fig_pca(res: dict) -> str:
    """Condition-averaged trajectories on the intact network's top 2 PCs."""
    pca = res["example"]["pca"]
    keys = [tuple(k) for k in pca["keys"]]
    names = list(pca["proj"].keys())
    need_task = isinstance(keys[0][0], str)
    if not need_task:
        cohs = sorted({k[1] for k in keys})
    fig, axs = _grid(len(names), 4, 2.8, 2.8, sharex=True, sharey=True)
    for ax, name in zip(axs, names):
        proj = np.array(pca["proj"][name])  # (K, T, 2)
        for k, key in enumerate(keys):
            if need_task:
                col, ls = STATE_COLORS.get(key[0], INK2), CUE_STYLES.get(key[1], "-")
            else:
                gt, coh = key
                ramp = SEQ_BLUES if gt == 1 else ["#fbd3c2", "#f6ab8b", "#f08a60", ORANGE, "#b84a1f", "#7d3013"]
                col, ls = ramp[min(cohs.index(coh) + 1, len(ramp) - 1)], "-"
            ax.plot(proj[k, :, 0], proj[k, :, 1], color=col, linestyle=ls, linewidth=1.5)
            ax.scatter(proj[k, -1, 0], proj[k, -1, 1], color=col, s=16, zorder=3)
        ax.set_title(name, loc="left", color=INK)
        ax.set_xlabel(f"PC1 ({100 * pca['var_explained'][0]:.0f}% var.)")
    fig.tight_layout()
    axs[0].set_ylabel(f"PC2 ({100 * pca['var_explained'][1]:.0f}% var.)")
    note = ("Color = need state (orange hungry, blue thirsty, gray sated); line = cue (solid food, dashed water, "
            "dotted neutral). Dots mark trial end." if need_task else
            "Blue = right-choice trials, orange = left; darker = stronger evidence. Dots mark trial end.")
    fig.text(0.01, -0.04, note, fontsize=8, color=INK2)
    return _png(fig)


def fig_lick(res: dict) -> str:
    """Lick probability per need state x cue: intact, then change caused by each perturbation."""
    seeds = list(res["seeds"].values())
    lick0 = seeds[0]["intact"]["lick"]
    states, cues = lick0["states"], lick0["cues"]
    intact = np.mean([s["intact"]["lick"]["p_lick"] for s in seeds], 0)
    names = [p["name"] for p in res["perturbations"]]
    fig, axs = _grid(len(names) + 1, 4, 2.6, 2.4)
    diverging = matplotlib.colors.LinearSegmentedColormap.from_list("div", [ORANGE, "#f1f0ed", BLUE])
    def draw(ax, mat, cmap, vmin, vmax, title, fmt, light):
        # light: value the colormap draws pale; text turns white only on strong colors.
        ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt(mat[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if abs(mat[i, j] - light) > 0.6 * (vmax - light if mat[i, j] > light else light - vmin) else INK)
        ax.set_xticks(range(len(cues)), cues, fontsize=8)
        ax.set_yticks(range(len(states)), states, fontsize=8)
        ax.grid(False)
        ax.set_title(title, loc="left", color=INK, fontsize=9)
    seq = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f1f0ed", BLUE])
    draw(axs[0], intact, seq, 0, 1, "Intact: P(lick)", lambda v: f"{v:.2f}", light=0)
    for ax, name in zip(axs[1:], names):
        pert = np.mean([s["perturbations"][name]["perturbed"]["lick"]["p_lick"] for s in seeds], 0)
        draw(ax, pert - intact, diverging, -1, 1, f"{name}: change", lambda v: f"{v:+.2f}", light=0)
    fig.tight_layout()
    return _png(fig)


def _summary_table(res: dict) -> str:
    rows = []
    for p in res["perturbations"]:
        s = res["summary"][p["name"]]
        tgt = p["target"]
        who = ", ".join(f"{k}={v}" for k, v in tgt.items())
        win = p["window"] if p["kind"] != "lesion" else "whole trial"
        first = next(iter(res["seeds"].values()))["perturbations"][p["name"]]
        pool = {"E": "random E", "I": "random I", "all": "random, any unit"}.get(first.get("control_population"), "random")
        rows.append(
            f"<tr><td>{html.escape(p['name'])}</td><td>{p['kind']}</td><td>{html.escape(who)}</td>"
            f"<td>{html.escape(str(win))}</td><td>{pool}</td>"
            f"<td>{100 * s['drop_perturbed_mean']:.1f} ± {100 * s['drop_perturbed_sem']:.1f}</td>"
            f"<td>{100 * s['drop_control_mean']:.1f} ± {100 * s['drop_control_sem']:.1f}</td>"
            f"<td>{100 * s['targeted_minus_control_mean']:.1f}</td><td>{s['p_value']:.3f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Perturbation</th><th>Kind</th><th>Target</th><th>Window</th><th>Control</th>"
        "<th>Drop, targeted (pp)</th><th>Drop, control (pp)</th><th>Targeted − control (pp)</th>"
        "<th>p (sign-flip)</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--card:#fff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3936;--card:#fcfcfb}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3936;--card:#fcfcfb}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px 16px}
main{max-width:1000px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}
.sub{color:var(--ink2);margin:0 0 16px}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink2);font-weight:600}
figure{margin:12px 0;background:var(--card);border-radius:8px;padding:8px;overflow-x:auto}
figure img{max-width:100%;height:auto;display:block}
figcaption{color:#52514e;font-size:13px;padding:4px 4px 0}
code{font-size:13px}
"""


def build_report(exp_dir: Path) -> Path:
    exp_dir = Path(exp_dir)
    res = json.loads((exp_dir / "results.json").read_text())
    seeds = res["seeds"]
    base_acc = [s["intact"]["accuracy"] for s in seeds.values()]
    m = res["model"]
    n_exc = round(m["frac_exc"] * m["n_units"]) if m["dale"] else m["n_units"]
    intact0 = next(iter(seeds.values()))["intact"]
    behavior_fig = []
    if "psychometric" in intact0:
        behavior_fig = [("Psychometric curves", fig_psychometric(res), "Mean across networks.")]
    elif "lick" in intact0:
        behavior_fig = [("Lick behavior by need state and cue", fig_lick(res),
                         "Mean across networks. First panel: intact lick probability. Other panels: perturbed minus "
                         "intact (blue = more licking, orange = less).")]
    figs = [
        ("Effect size vs. random control", fig_effects(res),
         "Accuracy drop from intact for each network. Lines pair a network's targeted and control results."),
        *behavior_fig,
        ("Population activity", fig_rates(res), f"Network seed {res['example']['seed']}. Solid: mean excitatory rate; dashed: mean inhibitory rate. Shaded: stimulus, then decision period."),
        ("State-space trajectories", fig_pca(res), "Projected on the intact network's top two principal components."),
    ]
    body = "".join(
        f"<h2>{t}</h2><figure><img alt='{html.escape(t)}' src='data:image/png;base64,{b}'>"
        f"<figcaption>{html.escape(c)}</figcaption></figure>" for t, b, c in figs
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Perturbation report</title>
<style>{CSS}</style></head><body><main>
<h1>{html.escape(res['name'])}</h1>
<p class="sub">Task {html.escape(res['task']['name'])} · {len(seeds)} networks · {m['n_units']} units
({n_exc} E / {m['n_units'] - n_exc} I, Dale's law {'on' if m['dale'] else 'off'}) ·
intact accuracy {100 * np.mean(base_acc):.1f}% (range {100 * min(base_acc):.1f}–{100 * max(base_acc):.1f}%)</p>
<h2>Summary</h2><div class="tablewrap">{_summary_table(res)}</div>
<p class="sub">{"Accuracy includes 0% coherence trials, where chance is 50%. " if "psychometric" in intact0 else ""}Control = the same number of randomly chosen
units (from the target's own population for selected targets, from the whole network for population targets),
same window and kind, averaged over 10 random draws; random targets are also averaged over 10 draws. p-value from an exact sign-flip test on the
per-network targeted − control differences.</p>
{body}
</main></body></html>"""
    out = exp_dir / "report.html"
    out.write_text(page)
    return out
