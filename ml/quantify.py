#!/usr/bin/env python3
# Quantification: estimate the class PROPORTIONS of a set of events (a beam campaign) from
# the out-of-fold probabilities of the A+xid model (ml/reports/oof_Axid.parquet, written by
# ml/train_xid.py). CPU only, no retraining.
#
# Fake beam campaigns per workload, separately for SDC and crash/timeout events: N events
# drawn from the single-bit visible runs of the small app with known class proportions
# (uniform, or "physical" = exposed bits x liveness x P(outcome | class)). Estimators:
#   CC    classify and count (argmax)
#   PCC   probabilistic classify and count (mean posterior)
#   ACC   adjusted CC (hard confusion matrix)
#   PACC  probabilistic adjusted CC (soft confusion matrix = mean posterior per true class)
#   PRIOR reference without a classifier: the class shares of the FI calibration runs
#   EM    Saerens, Latinne & Decaestecker (2002), on posteriors calibrated by a multinomial
#         logistic regression on the log-probabilities
# The posteriors are out-of-fold (site-grouped folds of ml/train_gbdt.py). The rows of each
# (app, outcome) are split at random, half of each class, into a calibration half (confusion
# matrices, calibrator, training prior) and a pool the campaign events are drawn from (with
# replacement): the beam hits the same program, so the same static sites, as the FI runs.
#
#   ~/thesis/venv-ml/bin/python ml/quantify.py [date]  -> ml/reports/quantify-<date>.json

import csv, datetime, json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import labels as L  # noqa: E402

REPORTS = os.path.join(HERE, "reports")
OOF = os.path.join(REPORTS, "oof_Axid.parquet")
COV = os.path.join(ROOT, "logs/final-2026-11-05/coverage_inputs.json")

APPS = ["gemm", "cnn", "ptr_chase", "shm_stencil", "sfu_math"]
OUTCOMES = {"sdc": 1, "crash": 0}
NS = [100, 300, 1000]
REPS = 200
SCENARIOS = ["uniform", "physical"]
METHODS = ["PRIOR", "CC", "PCC", "ACC", "PACC", "EM"]
COVERAGE = dict(methods=["PACC", "EM"], ns=[300, 1000], campaigns=20, boot=200)
THRESHOLDS = [10, 30]
MIN_ROWS = 20                      # rows of a class in (app, outcome) to enter the support
N_SPLITS = 20                      # calibration / pool splits, cycled over the repetitions
SEED = 0
EPS = 1e-6

# Propagation families (how the corruption spreads), for comparison with the merged groups.
FAMILY = {"fp32": "single_value", "fp64": "single_value", "fp16": "single_value", "tc": "single_value",
	"sfu": "single_value", "int": "single_value", "move": "single_value", "ld_data": "single_value",
	"ld_const": "single_value", "ur": "warp_uniform", "pr": "control_flow", "up": "control_flow",
	"addr": "pointer", "mem_global": "memory", "mem_shared": "memory"}

# Register file sizes per SM (bits) and assumptions, as in fi_coverage/coverage_final.py.
N_SM = 20
GPR_BITS = 256 * 1024 * 8          # 64K x 32-bit registers
PR_BITS = 48 * 32 * 7              # 48 warps x 32 lanes x 7 predicates
UR_BITS = 48 * 64 * 32             # 48 warps x 64 uniform registers x 32 bits
UP_BITS = 48 * 7                   # 48 warps x 7 uniform predicates
UR_USED = 0.5                      # share of the 64 uniform registers a kernel uses (coverage: U(.25, .75))


# ---------------------------------------------------------------- physical proportions

def _runs_by_app():
	"""Single-bit runs of the small apps with any outcome (Masked included): label, outcome,
	group, liveness weight, dynamic instructions of the group."""
	rows = []
	for r in csv.DictReader(open(L.RUNS)):
		if r["app"] not in APPS or r["fault_model"] != "single_bit" or r["valid_for_rates"] != "1":
			continue
		if any(L._matches(r, e) for e in L.MAP["selection"]["exclude"]):
			continue
		c = L.label(r)
		if c is None or r["outcome_class_for_rates"] not in ("sdc", "crash", "timeout", "masked"):
			continue
		rows.append(dict(app=r["app"], label=c, outcome=r["outcome_class_for_rates"], group=r["group"],
			lw=float(r["liveness_weight"]) if r["liveness_weight"] not in ("", "nan") else np.nan,
			gi=float(r["group_instrs"]) if r["group_instrs"] not in ("", "-1") else np.nan))
	return pd.DataFrame(rows)


def physical(cov, runs):
	"""{app: {"exposure": {structure: bits}, "weight": {class: bits}, "p_outcome": {class: {sdc, crash}}}}"""
	out = {}
	for app in APPS:
		ex = cov["exposure"][app]
		rf = ex["register_file"]
		kd = ex["kernel_durations"]["kernels"]
		w = np.array([kd[r["kernel"]]["gpu_us"] for r in rf])
		w = w / w.sum()
		tw = lambda f: float(sum(wi * f(r) for wi, r in zip(w, rf)))  # kernel-time weighted mean
		occ = lambda r: r["warps_per_sm_achieved"] / 48
		e = {"GPR": N_SM * GPR_BITS * tw(lambda r: r["rf_live_share_achieved"]),
			"PR": N_SM * PR_BITS * tw(lambda r: occ(r) * r["PR_live_of_7"]),
			"UR": N_SM * UR_BITS * UR_USED * tw(lambda r: occ(r) * r["UR_live_of_used"]),
			"UP": N_SM * UP_BITS * tw(lambda r: occ(r) * r["UP_live_of_7"]),
			"global": 8.0 * ex["global_memory"]["bytes_peak_live"]}
		sh = ex.get("shared_memory", {}).get("gpu")
		e["shared"] = sh["exposure_bit_s"]["windows"] / sh["exposure_bit_s"]["run_seconds"] if sh else 0.0
		g = runs[runs.app == app]
		# GPR split by functional unit: liveness-weighted labels of the gp group (all GPR writers)
		gp = g[g.group == "gp"].dropna(subset=["lw"])
		share = gp.groupby("label").lw.sum() / gp.lw.sum()
		# addresses: dynamic address uses (group addr) against GPR writes (group gp)
		n_addr, n_gp = g[g.group == "addr"].gi.median(), g[g.group == "gp"].gi.median()
		f_addr = float(n_addr / (n_addr + n_gp)) if n_addr == n_addr and n_gp == n_gp else 0.0
		wt = {c: e["GPR"] * (1 - f_addr) * float(s) for c, s in share.items()}
		wt["addr"] = e["GPR"] * f_addr
		wt.update({"pr": e["PR"], "ur": e["UR"], "up": e["UP"], "mem_global": e["global"], "mem_shared": e["shared"]})
		po = {}
		for c, x in g.groupby("label"):
			n = len(x)
			po[c] = {"sdc": float((x.outcome == "sdc").sum() / n), "crash": float(x.outcome.isin(["crash", "timeout"]).sum() / n), "n": n}
		out[app] = {"exposure": e, "gpr_split": {c: float(s) for c, s in share.items()}, "f_addr": f_addr,
			"weight": wt, "p_outcome": po}
	return out


def physical_props(phys, app, oc, support):
	w = np.array([phys[app]["weight"].get(c, 0.0) * phys[app]["p_outcome"].get(c, {}).get(oc, 0.0) for c in support])
	dropped = {c: phys[app]["weight"].get(c, 0.0) * phys[app]["p_outcome"].get(c, {}).get(oc, 0.0)
		for c in phys[app]["weight"] if c not in support}
	tot = w.sum() + sum(dropped.values())
	return w / w.sum(), {c: v / tot for c, v in dropped.items() if v > 0}


# ---------------------------------------------------------------- estimators

def simplex_lsq(A, b, lam=100.0):
	"""argmin ||A p - b|| with p >= 0, sum p = 1 (sum enforced by a heavy row)."""
	k = A.shape[1]
	p, _ = nnls(np.vstack([A, lam * np.ones((1, k))]), np.concatenate([b, [lam]]))
	s = p.sum()
	return p / s if s > 0 else np.full(k, 1 / k)


def em(post, prior, iters=1000, tol=1e-8):
	"""Saerens et al. (2002): re-weight calibrated posteriors to the unknown test prior."""
	q = prior.copy()
	for _ in range(iters):
		r = post * (q / prior)
		r /= r.sum(1, keepdims=True)
		qn = r.mean(0)
		if np.abs(qn - q).max() < tol:
			return qn
		q = qn
	return q


def estimate(P, C, ctx):
	"""P: posteriors (restricted to the support), C: calibrated posteriors; ctx: fold context."""
	k = P.shape[1]
	cc = np.bincount(P.argmax(1), minlength=k) / len(P)
	pcc = P.mean(0)
	return {"PRIOR": ctx["prior"], "CC": cc, "PCC": pcc, "ACC": simplex_lsq(ctx["M_hard"].T, cc), "PACC": simplex_lsq(ctx["M_soft"].T, pcc),
		"EM": em(C, ctx["prior"])}


# ---------------------------------------------------------------- per (app, outcome) setup

def group_probs(P, groups, k):
	G = np.zeros((k, len(groups)))
	for j, g in enumerate(groups):
		G[g, j] = 1
	return P @ G, G


def soft_matrix(P, y, g):
	return np.vstack([P[y == i].mean(0) for i in range(g)])


def hard_matrix(P, y, g):
	pred = P.argmax(1)
	return np.vstack([np.bincount(pred[y == i], minlength=g) / max((y == i).sum(), 1) for i in range(g)])


def context(P, y, o, g):
	"""Matrices, prior and calibrator from the calibration rows o; calibrated posteriors of
	every row."""
	if g == 1:  # everything merged: the proportion is 1 by construction
		return {"M_soft": np.ones((1, 1)), "M_hard": np.ones((1, 1)), "prior": np.ones(1)}, np.ones((len(P), 1))
	logp = np.log(P + EPS)
	lr = LogisticRegression(C=1.0, max_iter=2000)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		lr.fit(logp[o], y[o])
	C = np.zeros((len(P), g))
	C[:, lr.classes_] = lr.predict_proba(logp)
	prior = np.bincount(y[o], minlength=g) / o.sum()
	return {"M_soft": soft_matrix(P[o], y[o], g), "M_hard": hard_matrix(P[o], y[o], g), "prior": prior}, C


def half_splits(y, k, n, rng):
	"""n random splits of the rows into calibration (half of each class) and campaign pool."""
	out = []
	for _ in range(n):
		o = np.zeros(len(y), bool)
		for i in range(k):
			w = rng.permutation(np.where(y == i)[0])
			o[w[:len(w) // 2]] = True
		out.append(o)
	return out


def greedy_groups(P, y, k, thr):
	"""Merge the most confused pair (soft matrix, M_ij + M_ji) until cond(M) < thr."""
	groups = [[i] for i in range(k)]
	trace = []
	while True:
		lab = np.zeros(k, int)
		for j, g in enumerate(groups):
			lab[g] = j
		Pg, _ = group_probs(P, groups, k)
		M = soft_matrix(Pg, lab[y], len(groups))
		cond = float(np.linalg.cond(M))
		trace.append({"groups": len(groups), "cond": cond})
		if cond < thr or len(groups) == 1:
			return groups, trace
		S = M + M.T
		np.fill_diagonal(S, -1)
		a, b = np.unravel_index(S.argmax(), S.shape)
		a, b = min(a, b), max(a, b)
		trace[-1]["merge"] = [int(a), int(b)]
		groups = [g for j, g in enumerate(groups) if j not in (a, b)] + [groups[a] + groups[b]]


def counts_exact(props, n):
	raw = props * n
	c = np.floor(raw).astype(int)
	c[np.argsort(raw - c)[::-1][:n - c.sum()]] += 1
	return c


def interval(a):
	a = np.asarray(a)
	return [float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def bootstrap_coverage(setup, cnts, true, pools, rng):
	"""Percentile bootstrap of ONE campaign (events resampled, calibration fixed): share of
	(campaign, class) pairs whose 95% interval contains the true proportion, and mean width."""
	Pg, _, ctxs = setup
	hit = {m: [] for m in COVERAGE["methods"]}
	width = {m: [] for m in COVERAGE["methods"]}
	for r in range(COVERAGE["campaigns"]):
		s = r % N_SPLITS
		ctx, C = ctxs[s]
		idx = np.concatenate([rng.choice(pools[s][i], c, replace=True) for i, c in enumerate(cnts) if c > 0])
		boot = {m: [] for m in COVERAGE["methods"]}
		for _ in range(COVERAGE["boot"]):
			b = rng.choice(idx, len(idx), replace=True)
			est = estimate(Pg[b], C[b], ctx)
			for m in boot:
				boot[m].append(est[m])
		for m, v in boot.items():
			lo, hi = np.percentile(np.array(v), [2.5, 97.5], axis=0)
			hit[m] += list((lo <= true + 1e-12) & (true <= hi + 1e-12))
			width[m] += list(hi - lo)
	return {m: {"coverage": float(np.mean(hit[m])), "mean_width": float(np.mean(width[m]))} for m in hit}


# ---------------------------------------------------------------- main

def main():
	date = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()
	oof = pd.read_parquet(OOF)
	all_classes = [c[2:] for c in oof.columns if c.startswith("p_")]
	cov = json.load(open(COV))
	phys = physical(cov, _runs_by_app())
	rng = np.random.default_rng(SEED)
	R = {"date": date, "reps": REPS, "ns": NS, "methods": METHODS, "thresholds": THRESHOLDS, "physical": phys,
		"families": FAMILY, "cells": {}}
	for app in APPS:
		for oc, sdc in OUTCOMES.items():
			d = oof[(oof.meta_app == app) & (oof.f_out_sdc == sdc)].reset_index(drop=True)
			cnt = d.label.value_counts()
			support = [c for c in all_classes if cnt.get(c, 0) >= MIN_ROWS]
			excluded = {c: int(n) for c, n in cnt.items() if c not in support}
			k = len(support)
			d = d[d.label.isin(support)].reset_index(drop=True)
			P = d[["p_" + c for c in support]].to_numpy()
			P = P / P.sum(1, keepdims=True)
			y = d.label.map({c: i for i, c in enumerate(support)}).to_numpy()
			cell = {"support": support, "excluded_rows": excluded, "rows": {c: int(cnt[c]) for c in support}}
			# groupings: fine (singletons) and greedy at each threshold (matrix on all rows)
			groupings = {"fine": [[i] for i in range(k)]}
			cell["cond_fine"] = float(np.linalg.cond(soft_matrix(P, y, k)))
			cell["cond_fine_hard"] = float(np.linalg.cond(hard_matrix(P, y, k)))
			cell["groupings"] = {}
			for thr in THRESHOLDS:
				gs, trace = greedy_groups(P, y, k, thr)
				groupings["cond<%d" % thr] = gs
				names = [[support[i] for i in g] for g in gs]
				fams = [sorted({FAMILY[c] for c in g}) for g in names]
				cell["groupings"]["cond<%d" % thr] = {"groups": names, "families": fams, "trace": trace,
					"pure": int(sum(len(f) == 1 for f in fams)), "n_groups": len(gs)}
			splits = half_splits(y, k, N_SPLITS, rng)
			setups = {}
			for gname, gs in groupings.items():
				lab = np.zeros(k, int)
				for j, g in enumerate(gs):
					lab[g] = j
				Pg, G = group_probs(P, gs, k)
				setups[gname] = (Pg, G, [context(Pg, lab[y], o, len(gs)) for o in splits])
			pools = [[np.where(~o & (y == i))[0] for i in range(k)] for o in splits]
			pphys, dropped = physical_props(phys, app, oc, support)
			cell["physical_props"] = dict(zip(support, map(float, pphys)))
			cell["physical_dropped"] = dropped
			cell["results"] = {}
			for sc in SCENARIOS:
				props = np.full(k, 1 / k) if sc == "uniform" else pphys
				for n in NS:
					cnts = counts_exact(props, n)
					true = cnts / n
					err = {gname: {m: [] for m in METHODS} for gname in groupings}
					agg = {gname: {m: [] for m in METHODS} for gname in groupings if gname != "fine"}
					for r in range(REPS):
						s = r % N_SPLITS
						idx = np.concatenate([rng.choice(pools[s][i], c, replace=True) for i, c in enumerate(cnts) if c > 0])
						fine_est = None
						for gname, (Pg, G, ctxs) in setups.items():
							ctx, C = ctxs[s]
							est = estimate(Pg[idx], C[idx], ctx)
							tg = true @ G
							for m in METHODS:
								err[gname][m].append(np.abs(est[m] - tg))
							if gname == "fine":
								fine_est = est
							else:
								for m in METHODS:
									agg[gname][m].append(np.abs(fine_est[m] @ G - tg))
					res = {}
					for gname in groupings:
						names = ["+".join(support[i] for i in g) for g in groupings[gname]]
						res[gname] = {}
						for m in METHODS:
							e = np.array(err[gname][m])
							res[gname][m] = {"tv": interval(0.5 * e.sum(1)), "mae": interval(e.mean(1)),
								"per_class": {nm: interval(e[:, j]) for j, nm in enumerate(names)}}
							if gname != "fine":
								a = np.array(agg[gname][m])
								res[gname][m]["fine_aggregated_tv"] = interval(0.5 * a.sum(1))
					res["true"] = dict(zip(support, map(float, true)))
					if sc == "physical" and n in COVERAGE["ns"]:
						res["coverage"] = bootstrap_coverage(setups["fine"], cnts, true, pools, rng)
					cell["results"]["%s|%d" % (sc, n)] = res
			R["cells"]["%s|%s" % (app, oc)] = cell
			print(app, oc, "k=%d" % k, "cond=%.1f" % cell["cond_fine"],
				{g: v["n_groups"] for g, v in cell["groupings"].items()},
				{m: round(cell["results"]["physical|1000"]["fine"][m]["tv"][0], 3) for m in METHODS}, flush=True)
	json.dump(R, open(os.path.join(REPORTS, "quantify-%s.json" % date), "w"), indent=1)


if __name__ == "__main__":
	main()
