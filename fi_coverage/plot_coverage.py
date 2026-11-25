#!/usr/bin/env python3
"""Figures of the coverage chapter from coverage_final_results.json.
Usage: plot_coverage.py <coverage_final_results.json> <out_dir>"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
C1, C2, C3, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})
d = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
 
def save(fig, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight") 
# scaling of the coverage with the size of gemm
labs = ["gemm", "gemm N=256", "gemm N=512", "gemm N=1024", "gemm N=2048", "gemm N=4096"]
x = np.arange(len(labs))
fig, ax = plt.subplots(figsize=(6.3, 2.6))
for key, sc, col, name in [("Csdc", "no_ecc", C2, "SDC, no ECC"), ("C", "no_ecc", C1, "Total, no ECC"),
                           ("C", "ecc", C3, "Total, with ECC")]:
    q = np.array([d[l][sc][key] for l in labs]) * 100
    ax.fill_between(x, q[:, 0], q[:, 2], color=col, alpha=0.15, lw=0)
    ax.plot(x, q[:, 1], color=col, lw=2, marker="o", ms=4, label=name, markeredgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(["N=128", "256", "512", "1024", "2048", "4096"])
ax.set_xlabel("gemm matrix size N"); ax.set_ylabel("FI coverage (%)"); ax.set_ylim(0, 100)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
save(fig, "fig_cov_scaling")
 
# tornado: rank correlation of the inputs with C (gemm N=128)
# labels as in the equations: symbol of the term, structure as subscript
SYM = {"N": "N", "sigma": r"\sigma", "rho": r"\rho", "P": "P", "kappa": r"\kappa", "eta": r"\bar\eta"}
SHORT = {"GPR": "GPR", "Predicate register": "PR", "Uniform register": "UR", "L1 data / shared": "L1",
         "Cache tags": "tags", "L2 data": "L2", "VRAM cells": "VRAM", "Instruction/constant cache": "icache",
         "Datapath latches": "latch", "Control logic": "ctrl", "GSP / PCIe / microcontrollers": "GSP",
         "DRAM SEFI": "SEFI"}
 
def label(key):
    struct, term = key.split(": ")
    if struct == "Combinational":
        return r"$f_{\mathrm{comb}}$"
    return rf"${SYM[term]}_{{\mathrm{{{SHORT[struct]}}}}}$"
 
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharex=True)
for ax, sc, title in zip(axes, ["no_ecc", "ecc"], ["gemm N=128, no ECC", "gemm N=128, with ECC"]):
    s = list(d["gemm"][sc]["sensitivity"].items())[:7][::-1]
    v = [b for _, b in s]
    ax.barh(range(len(s)), v, color=[C1 if b > 0 else C2 for b in v], height=0.65)
    ax.set_yticks(range(len(s))); ax.set_yticklabels([label(a) for a, _ in s], fontsize=9)
    ax.axvline(0, color=INK2, lw=0.8); ax.set_xlim(-0.7, 0.7); ax.set_title(title, fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
axes[0].set_xlabel(r"Rank correlation with $C_{\mathrm{FI}}$"); axes[1].set_xlabel(r"Rank correlation with $C_{\mathrm{FI}}$")
fig.tight_layout(); save(fig, "fig_cov_tornado")
 