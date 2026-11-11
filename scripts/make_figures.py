#!/usr/bin/env python3
"""Figures for the fault-injection results chapter.

Reads the dataset exported by scripts/export_dataset.py and the coverage inputs:
    python3 make_figures.py <runs.csv> <coverage_inputs.json> <out_dir>
Writes one PDF per figure (vector, for LaTeX) plus a PNG preview.
Rates use the same filters as the SUMMARY files: valid_for_rates == 1 and
outcome_class_for_rates; memory AVF figures use for_avf == 1.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated palette (dataviz reference, light mode): slots 2, 1, 3.
C_SDC, C_CRASH, C_TIMEOUT = "#eb6834", "#2a78d6", "#1baf7a"
C_MASKED = "#c9c8c3"
GRAYS = ["#8f8e89", "#b3b2ad", "#d6d5d0", "#e8e7e3"]   # Masked subclasses, dark -> light
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WIDTH = 6.3   # inches, \textwidth of a typical thesis

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})

APPS = ["gemm", "cnn", "ptr_chase", "shm_stencil", "sfu_math"]
OUTC = [("sdc", "SDC", C_SDC), ("crash", "Crash (DUE)", C_CRASH), ("timeout", "Timeout (DUE)", C_TIMEOUT),
        ("masked", "Masked", C_MASKED)]


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["valid_for_rates"] == 1].copy()
    df["oc"] = df["outcome_class_for_rates"].fillna(df["outcome_class"])
    return df


def shares(sub):
    n = len(sub)
    return {k: (sub["oc"] == k).sum() / n for k, _, _ in OUTC}, n


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def stacked_hbar(ax, labels, rows, ns, colors_keys):
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for key, lab, col in colors_keys:
        v = np.array([r.get(key, 0) for r in rows]) * 100
        ax.barh(y, v, left=left, color=col, edgecolor="white", linewidth=1.0, height=0.72, label=lab)
        for yi, li, vi in zip(y, left, v):   # selective direct labels: segments >= 8 %
            if vi >= 8:
                ax.text(li + vi / 2, yi, f"{vi:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if col not in (C_MASKED, *GRAYS[2:]) else INK)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}  (n={n})" for l, n in zip(labels, ns)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of runs (%)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


# --- Figure 1: registers, single bit -------------------------------------------------
def fig_registers(df, out):
    reg = df[(df["injector"] == "reg") & (df["fault_model"] == "single_bit") &
             (~df["group"].isin(["addr", "addr_store"]))]
    order = [("gemm", "tc"), ("gemm", "fp32"), ("gemm", "pr"), ("gemm", "ld"), ("gemm", "gp"),
             ("gemm", "uniform"), ("sfu_math", "sfu"), ("sfu_math", "gp"), ("sfu_math", "pr"),
             ("sfu_math", "uniform"), ("ptr_chase", "gp"), ("ptr_chase", "pr"), ("ptr_chase", "uniform"),
             ("shm_stencil", "gp"), ("shm_stencil", "pr"), ("shm_stencil", "uniform"),
             ("cnn", "fp32"), ("cnn", "ld"), ("cnn", "gp"), ("cnn", "uniform")]
    labels, rows, ns = [], [], []
    for app, grp in order:
        sub = reg[(reg["app"] == app) & (reg["group"] == grp)]
        if grp == "uniform" and app == "gemm":   # keep only the re-run (block 3), not final-v1
            sub = sub[sub["campaign"] != "final-v1"]
        if len(sub) == 0:
            continue
        r, n = shares(sub)
        labels.append(f"{app} · {grp}"); rows.append(r); ns.append(n)
    fig, ax = plt.subplots(figsize=(WIDTH, 0.27 * len(labels) + 0.9))
    stacked_hbar(ax, labels, rows, ns, OUTC)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False)
    save(fig, out, "fig_res_registers")
    return pd.DataFrame(rows, index=labels).assign(n=ns)


# --- Figure 2: memory, single bit, with Masked subclasses ---------------------------
def fig_memory(df, out):
    mem = df[(df["injector"] == "mem") & (df["fault_model"] == "single_bit") & (df["for_avf"] == 1)]
    keys = [("sdc", "SDC", C_SDC), ("crash", "Crash (DUE)", C_CRASH), ("timeout", "Timeout (DUE)", C_TIMEOUT),
            ("read_masked", "Masked: read, tolerated", GRAYS[0]), ("overwritten", "Masked: overwritten", GRAYS[1]),
            ("no_read", "Masked: never read", GRAYS[2]), ("other_m", "Masked: other", GRAYS[3])]
    labels, rows, ns = [], [], []
    for app in APPS:
        for mode in ["a", "b", "u"]:
            sub = mem[(mem["app"] == app) & (mem["mem_mode"] == mode)]
            if len(sub) == 0:
                continue
            n = len(sub)
            r = {k: (sub["oc"] == k).sum() / n for k in ["sdc", "crash", "timeout"]}
            m = sub[sub["oc"] == "masked"]["masked_subclass"].fillna("other")
            for k in ["read_masked", "overwritten", "no_read"]:
                r[k] = (m == k).sum() / n
            r["other_m"] = (~m.isin(["read_masked", "overwritten", "no_read"])).sum() / n
            name = {"a": "mode a (between kernels)", "b": "mode b (in L2, during kernel)",
                    "u": "mode u (shared, uniform)"}[mode]
            labels.append(f"{app} · {name}"); rows.append(r); ns.append(n)
    fig, ax = plt.subplots(figsize=(WIDTH, 0.3 * len(labels) + 1.2))
    stacked_hbar(ax, labels, rows, ns, keys)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.35, 1.0), frameon=False)
    save(fig, out, "fig_res_memory")
    return pd.DataFrame(rows, index=labels).assign(n=ns)


# --- Figure 3: outcome per bit position, fp32 ---------------------------------------
def fig_fp32_bits(df, out):
    reg = df[(df["injector"] == "reg") & (df["fault_model"] == "single_bit") & (df["group"] == "fp32")].copy()
    reg["maskv"] = reg["mask"].map(lambda m: int(str(m), 0) if isinstance(m, str) and m.strip() else 0)
    reg = reg[reg["maskv"] > 0]
    reg["bitpos"] = np.log2(reg["maskv"]).round().astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True)
    for ax, app in zip(axes, ["gemm", "cnn"]):
        sub = reg[reg["app"] == app]
        bits = np.arange(32)
        bottom = np.zeros(32)
        for key, lab, col in OUTC:
            v = np.array([((sub["bitpos"] == b) & (sub["oc"] == key)).sum() /
                          max((sub["bitpos"] == b).sum(), 1) for b in bits]) * 100
            ax.bar(bits, v, bottom=bottom, width=0.85, color=col, edgecolor="white", linewidth=0.5, label=lab)
            bottom += v
        for x0, x1, name in [(-0.5, 22.5, "mantissa"), (22.5, 30.5, "exponent"), (30.5, 31.5, "sign")]:
            ax.annotate("", xy=(x0, 104), xytext=(x1, 104), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
                        annotation_clip=False)
            ax.text((x0 + x1) / 2, 106, name if name != "exponent" else "exp.", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_title(f"{app} · fp32 (n={len(sub)})", pad=18)
        ax.set_xlabel("Bit flipped (0 = LSB)")
        ax.set_xlim(-0.6, 31.6); ax.set_ylim(0, 100)
        ax.set_xticks([0, 8, 16, 23, 31])
    axes[0].set_ylabel("Share of runs (%)")
    axes[1].legend(loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False)
    save(fig, out, "fig_res_fp32_bits")


# --- Figure 4: MCU, failure rate vs k (GPR, group gp) -------------------------------
def fig_mcu(df, out):
    reg = df[(df["injector"] == "reg") & (df["group"] == "gp")]
    models = [("adj_bits", "Adjacent bits", C_SDC), ("adj_regs", "Adjacent registers", C_CRASH),
              ("adj_lanes", "Adjacent lanes", C_TIMEOUT)]
    fig, axes = plt.subplots(1, 5, figsize=(WIDTH, 2.0), sharey=True)
    for ax, app in zip(axes, APPS):
        sb = reg[(reg["app"] == app) & (reg["fault_model"] == "single_bit")]
        base = (sb["oc"] != "masked").mean() * 100 if len(sb) else np.nan
        for mdl, lab, col in models:
            ys = [base]
            for k in (2, 3, 4):
                s = reg[(reg["app"] == app) & (reg["fault_model"] == mdl) & (reg["k"] == k)]
                ys.append((s["oc"] != "masked").mean() * 100 if len(s) else np.nan)
            ax.plot([1, 2, 3, 4], ys, color=col, lw=2, marker="o", ms=4, label=lab,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.set_title(app, fontsize=8.5)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("k (cells)")
        ax.set_ylim(0, 100); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("SDC + DUE (%)")
    axes[2].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.42), frameon=False)
    save(fig, out, "fig_res_mcu_k")


# --- Figure 5: addresses, address register vs descriptor ----------------------------
def fig_addr(df, out):
    a = df[(df["injector"] == "reg") & (df["group"] == "addr")]
    labels, rows, ns = [], [], []
    for app in APPS:
        for part, name in [("ra", "address register"), ("desc", "descriptor")]:
            sub = a[(a["app"] == app) & (a["addr_part"] == part)]
            if len(sub) == 0:
                continue
            r, n = shares(sub)
            labels.append(f"{app} · {name}"); rows.append(r); ns.append(n)
    fig, ax = plt.subplots(figsize=(WIDTH, 0.27 * len(labels) + 0.9))
    stacked_hbar(ax, labels, rows, ns, OUTC)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False)
    save(fig, out, "fig_res_addr")


# --- Figure 6: live share of the register file per kernel ---------------------------
def fig_rf(cov, out):
    labels, alloc, live = [], [], []
    for app in APPS:
        for k in cov["exposure"][app]["register_file"]:
            labels.append(f"{app} · {k['kernel']}")
            alloc.append(k.get("rf_share_achieved", k["rf_share_at_mean"]) * 100)
            live.append(k.get("rf_live_share_achieved", k["rf_live_share_at_mean"]) * 100)
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(WIDTH, 0.24 * len(labels) + 0.8))
    ax.barh(y, alloc, color="#a9c6ec", height=0.72, label="Allocated (resident warps)")
    ax.barh(y, live, color=C_CRASH, height=0.72, label="Live (written, still to be read)")
    for yi, a_, l_ in zip(y, alloc, live):
        ax.text(a_ + 1, yi, f"{l_:.0f} / {a_:.0f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100); ax.set_xlabel("Share of the register file of an SM (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    save(fig, out, "fig_res_rf_share")


if __name__ == "__main__":
    runs, covp, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    df = load(runs)
    cov = json.load(open(covp))
    t1 = fig_registers(df, out)
    t2 = fig_memory(df, out)
    fig_fp32_bits(df, out)
    fig_mcu(df, out)
    fig_addr(df, out)
    fig_rf(cov, out)
    t1.round(3).to_csv(os.path.join(out, "check_registers.csv"))
    t2.round(3).to_csv(os.path.join(out, "check_memory.csv"))
    print(t1.round(3).to_string()); print(t2.round(3).to_string())
