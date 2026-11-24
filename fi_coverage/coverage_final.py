"""Coverage of the fault injection: final model (one script for every workload).

    F^o_s = N_s sigma_s eta_s rho_s P^o_s ;   C^o = sum_s kappa_s F^o_s / sum_s F^o_s
  - steady state: T = kernel time; kernels weighted by their GPU duration (ncu, no tool);
  - rho of GPR, PR, UR from allocation x liveness (coverage_inputs / scaling.json);
  - rho of L1, datapath latches and control logic scale with the achieved RF share a;
  - L2 / VRAM: rho = working set / capacity; P from memory modes b / a;
  - MCU: AVF(k) ~ AVF(1) (within 3 points, block 2-4), so sum_k p(k) eta(k) AVF(k) = eta_bar AVF;
  - ECC scenario: eta_bar = sum_k p(k) pi(k) eta_word(k) ~ U(0, 0.04)
      (0 = full interleaving; 0.022 = no interleaving, Hsiao (72,64); 0.04 = no interleaving, eta3 = 1).
Usage: coverage_final.py <coverage_inputs.json> <scaling.json> <out.json>
"""
import json
import sys

import numpy as np

RNG = np.random.default_rng(20261123)
NMC = 100_000
KB = 1024 * 8
SM = 20
N_RF, N_PR, N_UR = SM * 256 * KB, SM * 1536 * 7, SM * 48 * (64 * 32 + 7)
N_L1, N_L2, N_VRAM = SM * 128 * KB, 32 * 1024 * KB, 8 * 1024**3 * 8
SRAM, FF, DRAM = ("l", 1e-15, 3e-15), ("l", 5e-16, 2e-15), ("l", 1e-17, 1e-16)
ETA_ECC = ("u", 0.0, 0.04)
L1_FACT, SEQ_FACT, UR_USED = ("u", .3, .9), ("u", .3, 1.2), ("u", .25, .75)

# Measured P: (fail min, fail max), SDC share of the failures. Ranges = per write vs liveness-weighted.
P = {
 "gemm":        dict(GPR=((.918, .937), .68), PR=((.249, .943), .94), UR=((.233, .296), .45),
                     L2=((.281, .338), 1.), VR=((.341, .400), 1.)),
 "cnn":         dict(GPR=((.348, .376), .35), PR=((.25, 1.0), .9), UR=((.242, .269), .2),
                     L2=((.141, .186), 1.), VR=((.205, .257), 1.)),
 "ptr_chase":   dict(GPR=((.929, .978), .78), PR=((.974, 1.0), .974), UR=((.034, .362), .34),
                     L2=((.50, .55), .61), VR=((.34, .385), .56)),
 "shm_stencil": dict(GPR=((.940, .975), .58), PR=((.996, 1.0), 1.), UR=((.069, .713), .18),
                     L2=((.15, .55), .8), VR=((.15, .55), .8)),
 "sfu_math":    dict(GPR=((.754, .882), .58), PR=((.30, .405), 1.), UR=((.11, .444), .49),
                     L2=((.15, .55), .8), VR=((.15, .55), .8)),
}
# block 6: mode b measured on the large variants (N=2048, 4096 taken equal to N=1024)
PB6 = {"gemm_n1024": (.516, .560), "gemm_n2048": (.516, .560), "gemm_large": (.516, .560),
       "cnn_large": (.214, .251)}


def draw(s):
    if not isinstance(s, tuple):
        return np.full(NMC, float(s))
    k, lo, hi = s
    if lo == hi:
        return np.full(NMC, float(lo))
    return np.exp(RNG.uniform(np.log(lo), np.log(hi), NMC)) if k == "l" else RNG.uniform(lo, hi, NMC)


def stats_small(cov, app):
    rf = cov["exposure"][app]["register_file"]
    kd = cov["exposure"][app]["kernel_durations"]["kernels"]
    w = np.array([kd[r["kernel"]]["gpu_us"] for r in rf]); w = w / w.sum()
    g = lambda key, f=lambda v: v: float(sum(wi * f(r[key]) for wi, r in zip(w, rf)))
    warps = g("warps_per_sm_achieved")
    return dict(a=g("rf_share_achieved"), rf_live=g("rf_live_share_achieved"), warps=warps,
                pr=float(sum(wi * r["warps_per_sm_achieved"] / 48 * r["PR_live_of_7"] for wi, r in zip(w, rf))),
                ur=float(sum(wi * r["warps_per_sm_achieved"] / 48 * r["UR_live_of_used"] for wi, r in zip(w, rf))),
                ws=cov["exposure"][app]["global_memory"]["bytes_peak_live"])


def stats_large(rows, name, small):
    ks = [r for r in rows if r["variant"] == name]
    w = np.array([r["gpu_us"] for r in ks]); w = w / w.sum()
    live = [r.get("rf_live_share") if r.get("rf_live_share") is not None
            else r["rf_share_achieved"] * (0.642 if r["kernel"] == "gemm_core" else 0.621) for r in ks]
    warps = float(sum(wi * r["warps_active_per_sm"] for wi, r in zip(w, ks)))
    scale = warps / max(small["warps"], 1e-9)
    return dict(a=float(sum(wi * r["rf_share_achieved"] for wi, r in zip(w, ks))),
                rf_live=float(sum(wi * l for wi, l in zip(w, live))), warps=warps,
                pr=small["pr"] * scale, ur=small["ur"] * scale, ws=None)


def run(fam, st, ecc, pl2=None):
    p = dict(P[fam])
    if pl2:
        p["L2"] = (pl2, 1.)
    a, ws = st["a"], st["ws"]
    l2 = min(ws, N_L2 / 8) / (N_L2 / 8); vr = ws / (N_VRAM / 8)
    S = [  # name, N, sigma, rho, (P range, SDC share), kappa, protected by ECC
     ("GPR", N_RF, SRAM, ("u", .9 * st["rf_live"], 1.1 * st["rf_live"]), p["GPR"], 1, True),
     ("Predicate register", N_PR, FF, ("u", .5 * st["pr"] + 1e-4, 1.5 * st["pr"] + 2e-4), p["PR"], 1, False),
     ("Uniform register", N_UR, SRAM, "UR", p["UR"], 1, False),
     ("L1 data / shared", N_L1, SRAM, "L1", ((.2, .6), .8), ("u", .6, 1.), True),
     ("Cache tags", ("l", .03 * (N_L1 + N_L2), .08 * (N_L1 + N_L2)), SRAM, ("u", .01, .10), ((.3, .8), .5), ("u", 0, .5), True),
     ("L2 data", N_L2, SRAM, ("u", .9 * l2, l2), p["L2"], 1, True),
     ("VRAM cells", N_VRAM, DRAM, ("u", .9 * vr, vr), p["VR"], 1, True),
     ("Instruction/constant cache", ("u", SM * 64 * KB, SM * 192 * KB), SRAM, ("u", .02, .10), ((.5, .9), .3), 0, "parity"),
     ("Datapath latches", ("l", .05 * N_RF, .20 * N_RF), FF, "SEQ", p["GPR"], ("u", .7, 1.), False),
     ("Control logic", ("l", .05 * N_RF, .20 * N_RF), FF, "SEQ", ((.5, .9), .2), ("u", 0, .2), False),
     ("GSP / PCIe / microcontrollers", ("l", 2e6, 1e7), SRAM, ("u", .05, .30), ((.3, .8), .05), 0, False),
    ]
    names, F, Fs, K, inputs = [], [], [], [], {}
    for n, N, sig, rho, (prng, sdc), kap, e in S:
        N, sig, kap = draw(N), draw(sig), draw(kap)
        if rho == "UR":
            r = st["ur"] * draw(UR_USED)
        elif rho == "L1":
            r = np.minimum(draw(L1_FACT) * a, 1)
        elif rho == "SEQ":
            r = np.minimum(draw(SEQ_FACT) * a, 1)
        else:
            r = draw(rho)
        Pv = draw(("u", *prng))
        eta = np.ones(NMC)
        if ecc and e is True:
            eta = draw(ETA_ECC)
        if ecc and e == "parity":
            eta = np.zeros(NMC)
        f = N * sig * eta * r * Pv
        names.append(n); F.append(f); Fs.append(f * sdc); K.append(kap)
        for lab, v in (("N", N), ("sigma", sig), ("rho", r), ("P", Pv), ("kappa", kap), ("eta", eta)):
            if np.ptp(v) > 0:
                inputs[f"{n}: {lab}"] = v
        if n == "Datapath latches":
            lat = (N * sig, r, Pv, kap)
        if n == "GPR":
            rf_raw = N * sig
    cfrac = draw(("u", .05, .15)); inputs["Combinational: fraction"] = cfrac
    fc = cfrac * (rf_raw + lat[0]) * lat[1] * lat[2]
    names.append("Combinational logic"); F.append(fc); Fs.append(fc * .7); K.append(lat[3])
    sefi = draw(("l", 1e-11, 1e-9)) * 4; inputs["DRAM SEFI: sigma"] = sefi
    names.append("DRAM SEFI / control"); F.append(sefi); Fs.append(sefi * 0); K.append(np.zeros(NMC))
    F, Fs, K = map(np.array, (F, Fs, K)); Fd = F - Fs
    C, Cs, Cd = (K * F).sum(0) / F.sum(0), (K * Fs).sum(0) / Fs.sum(0), (K * Fd).sum(0) / Fd.sum(0)
    q = lambda x: [round(float(np.percentile(x, v)), 3) for v in (5, 50, 95)]
    rank = lambda x: np.argsort(np.argsort(x))
    rc = rank(C)
    sens = {k: float(np.corrcoef(rank(v), rc)[0, 1]) for k, v in inputs.items()}
    sens = dict(sorted(sens.items(), key=lambda kv: -abs(kv[1]))[:8])
    return dict(C=q(C), Csdc=q(Cs), Cdue=q(Cd),
                share={n: round(float(np.median(F[i] / F.sum(0))), 3) for i, n in enumerate(names)},
                sensitivity=sens)


if __name__ == "__main__":
    cov = json.load(open(sys.argv[1]))
    rows = json.load(open(sys.argv[2]))["rows"]
    WS_LARGE = {"gemm_n256": 1.31e6, "gemm_n512": 5.24e6, "gemm_n1024": 2.1e7, "gemm_n2048": 8.39e7,
                "gemm_large": 3.36e8, "cnn_large": 5.69e6}
    out = {}
    small = {}
    for app in ["gemm", "cnn", "ptr_chase", "shm_stencil", "sfu_math"]:
        st = stats_small(cov, app); small[app] = st
        out[app] = dict(stats=st, no_ecc=run(app, st, False), ecc=run(app, st, True))
    for v, fam, lab in [("gemm_n256", "gemm", "gemm N=256"), ("gemm_n512", "gemm", "gemm N=512"),
                        ("gemm_n1024", "gemm", "gemm N=1024"), ("gemm_n2048", "gemm", "gemm N=2048"),
                        ("gemm_large", "gemm", "gemm N=4096"), ("cnn_large", "cnn", "cnn batch 256")]:
        st = stats_large(rows, v, small[fam]); st["ws"] = WS_LARGE[v]
        out[lab] = dict(stats=st, no_ecc=run(fam, st, False, PB6.get(v)), ecc=run(fam, st, True, PB6.get(v)))
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    for k, r in out.items():
        s = r["stats"]
        print(f"{k:14s} a={s['a']:.2f} live={s['rf_live']:.3f} ws={s['ws']/1e6:6.2f}MB | "
              f"noECC C {r['no_ecc']['C']} SDC {r['no_ecc']['Csdc']} DUE {r['no_ecc']['Cdue']} | "
              f"ECC C {r['ecc']['C']} SDC {r['ecc']['Csdc']}")
