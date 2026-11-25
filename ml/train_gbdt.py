#!/usr/bin/env python3
# Stage-1 diagnostic baseline: LightGBM multiclass on ml/data/stage1_v0.parquet.
#
#   ~/thesis/venv-ml/bin/python ml/train_gbdt.py [date]
#
# Variant A (operational): all f_ + fw_ features, StratifiedGroupKFold by static site.
# Variant B (physics test): no fw_ and no workload-revealing f_, StratifiedGroupKFold by
#   site and leave-one-workload-out (LOWO) on the classes present in >= 2 workloads.
# Writes ml/reports/baseline-<date>.md, ml/reports/oof_A.parquet (out-of-fold
# probabilities) and ml/reports/baseline-<date>.json (all numbers of the report).
# Near-default hyperparameters, balanced class weights, early stopping on a site-grouped
# part of each training fold. No tuning: it is a diagnostic baseline.

import datetime, json, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data/stage1_v0.parquet")
REPORTS = os.path.join(HERE, "reports")
SEED = 0
N_FOLDS = 5
INNER_FOLDS = 6  # one sixth of each training fold, grouped by site, for early stopping
PARAMS = dict(objective="multiclass", learning_rate=0.05, num_leaves=31, min_child_samples=20,
	feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, seed=SEED, deterministic=True,
	num_threads=8, verbose=-1)
MAX_ROUNDS, EARLY_STOP = 3000, 50
B_DROP = ["f_frac_eq_init", "f_rank_log2", "f_cnn_frac_images", "f_partial_na",
	"f_frac_kernels_completed", "f_frac_outputs_wrong"]
# B2: B without f_frac_completed_correct too (NaN by construction on the apps without one
# output file per kernel; on SDCs it is 1 - fraction of outputs wrong, i.e. the output count)
B2_DROP = B_DROP + ["f_frac_completed_correct"]
SHAP_ROWS = 4000
SUBSETS = {"sdc": lambda d: d.f_out_sdc == 1, "crash_timeout": lambda d: d.f_out_sdc == 0, "all": lambda d: d.f_out_sdc >= 0}


def balanced_weights(y, k):
	counts = np.bincount(y, minlength=k).astype(float)
	w = np.zeros(k)
	present = counts > 0
	w[present] = len(y) / (present.sum() * counts[present])
	return w[y]


def fit(X, y, groups, k):
	"""Fit with early stopping on a site-grouped, stratified sixth of (X, y)."""
	inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
	tr, va = next(inner.split(X, y, groups))
	dtr = lgb.Dataset(X.iloc[tr], y[tr], weight=balanced_weights(y[tr], k))
	dva = lgb.Dataset(X.iloc[va], y[va], weight=balanced_weights(y[va], k), reference=dtr)
	m = lgb.train(dict(PARAMS, num_class=k), dtr, MAX_ROUNDS, valid_sets=[dva],
		callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
	return m


def fit_rounds(X, y, k, rounds):
	return lgb.train(dict(PARAMS, num_class=k), lgb.Dataset(X, y, weight=balanced_weights(y, k)), rounds)


def cross_val(d, feats, classes, splitter, groups):
	"""Out-of-fold probabilities; returns (proba, fold, best iterations)."""
	k = len(classes)
	y = d.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
	X = d[feats]
	proba = np.full((len(d), k), np.nan)
	fold = np.full(len(d), -1)
	iters = []
	for f, (tr, te) in enumerate(splitter):
		m = fit(X.iloc[tr], y[tr], groups[tr], k)
		proba[te] = m.predict(X.iloc[te], num_iteration=m.best_iteration)
		fold[te] = f
		iters.append(m.best_iteration)
	return proba, fold, iters


def metrics(y_true, y_pred, classes):
	"""Macro-F1 over the classes present in y_true, accuracy, row-normalised confusion."""
	present = [c for c in classes if (y_true == c).any()]
	cm = confusion_matrix(y_true, y_pred, labels=classes).astype(float)
	rows = cm.sum(1, keepdims=True)
	cmn = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
	return {"n": int(len(y_true)), "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"f1_per_class": dict(zip(present, map(float, f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)))),
		"confusion": cm.astype(int).tolist(), "confusion_norm": cmn.tolist(), "labels": classes}


def by_subset(d, pred, classes):
	return {s: metrics(d.label[f(d)].to_numpy(), pred[f(d).to_numpy()], classes) for s, f in SUBSETS.items()}


def confused_pairs(m, top=10):
	"""Unordered pairs by mean of the two row-normalised off-diagonal rates."""
	labels, cm, cmn = m["labels"], np.array(m["confusion"]), np.array(m["confusion_norm"])
	out = []
	for i in range(len(labels)):
		for j in range(i + 1, len(labels)):
			if cm[i].sum() == 0 and cm[j].sum() == 0:
				continue
			if cm[i, j] + cm[j, i] == 0:
				continue
			out.append({"a": labels[i], "b": labels[j], "a_as_b": float(cmn[i, j]), "b_as_a": float(cmn[j, i]),
				"n_a_as_b": int(cm[i, j]), "n_b_as_a": int(cm[j, i]), "score": float(cmn[i, j] + cmn[j, i]) / 2})
	return sorted(out, key=lambda p: -p["score"])[:top]


def per_fold_f1(d, pred, fold, classes):
	"""F1 per class in each fold, mean and std (classes absent from a fold are skipped)."""
	res = {}
	for c in classes:
		vals = []
		for f in sorted(set(fold)):
			mk = fold == f
			yt, yp = d.label.to_numpy()[mk], pred[mk]
			if (yt == c).any():
				vals.append(float(f1_score(yt == c, yp == c, zero_division=0)))
		res[c] = {"folds": len(vals), "mean": float(np.mean(vals)) if vals else None, "std": float(np.std(vals)) if vals else None, "values": vals}
	return res


def shap_importance(model, X, classes):
	rng = np.random.default_rng(SEED)
	idx = rng.choice(len(X), min(SHAP_ROWS, len(X)), replace=False)
	sv = shap.TreeExplainer(model).shap_values(X.iloc[idx])
	sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)  # (rows, features, classes)
	mean_abs = np.abs(sv).mean(0)  # (features, classes)
	feats = list(X.columns)
	glob = sorted(zip(feats, mean_abs.sum(1)), key=lambda t: -t[1])
	per = {c: sorted(zip(feats, mean_abs[:, i]), key=lambda t: -t[1])[:8] for i, c in enumerate(classes)}
	return {"global": [(f, float(v)) for f, v in glob], "per_class": {c: [(f, float(v)) for f, v in l] for c, l in per.items()}}


# ---------------- report helpers
def fmt_cm(m, decimals=2):
	labels, cmn, cm = m["labels"], np.array(m["confusion_norm"]), np.array(m["confusion"])
	keep = [i for i in range(len(labels)) if cm[i].sum() > 0]
	head = "| true \\ pred | " + " | ".join(labels) + " | n |"
	L = [head, "|" + "---|" * (len(labels) + 2)]
	for i in keep:
		cells = []
		for j in range(len(labels)):
			v = cmn[i, j]
			s = ("%.*f" % (decimals, v)) if v > 0 else "."
			cells.append("**%s**" % s if i == j else s)
		L.append("| %s | %s | %d |" % (labels[i], " | ".join(cells), cm[i].sum()))
	return "\n".join(L)


def fmt_pairs(pairs):
	L = ["| pair | a→b | b→a | runs a→b | runs b→a |", "|---|---|---|---|---|"]
	for p in pairs:
		L.append("| %s ↔ %s | %.2f | %.2f | %d | %d |" % (p["a"], p["b"], p["a_as_b"], p["b_as_a"], p["n_a_as_b"], p["n_b_as_a"]))
	return "\n".join(L)


def fmt_subsets(res):
	L = ["| subset | runs | macro-F1 | accuracy |", "|---|---|---|---|"]
	for s, m in res.items():
		L.append("| %s | %d | %.3f | %.3f |" % (s, m["n"], m["macro_f1"], m["accuracy"]))
	return "\n".join(L)


def main():
	date = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()
	os.makedirs(REPORTS, exist_ok=True)
	t0 = time.time()
	full = pd.read_parquet(DATA)
	tr = full[(full.split == "train") & (full.trainable == 1)].reset_index(drop=True)
	mcu = full[(full.split == "mcu_test") & (full.trainable == 1)].reset_index(drop=True)
	classes = sorted(tr.label.unique())
	k = len(classes)
	cidx = {c: i for i, c in enumerate(classes)}
	# Xid-detail features (ml/xid_features.py) are evaluated separately by ml/train_xid.py
	f_all = [c for c in tr.columns if c.startswith("f_") and not c.startswith(("f_xid13_", "f_xid31_"))]
	fw = [c for c in tr.columns if c.startswith("fw_")]
	feats_A = f_all + fw
	feats_B = [c for c in f_all if c not in B_DROP]
	feats_B2 = [c for c in f_all if c not in B2_DROP]
	groups = tr.meta_site.to_numpy()
	y = tr.label.map(cidx).to_numpy()
	R = {"date": date, "rows_train": len(tr), "rows_mcu": len(mcu), "classes": classes, "features_A": feats_A, "features_B": feats_B, "features_B2": feats_B2, "params": PARAMS}

	# sites per class and per workload
	R["sites"] = tr.groupby("label").meta_site.nunique().to_dict()
	R["runs"] = {c: {"total": int((tr.label == c).sum()), "sdc": int(((tr.label == c) & (tr.f_out_sdc == 1)).sum()),
		"crash_timeout": int(((tr.label == c) & (tr.f_out_sdc == 0)).sum())} for c in classes}
	R["workloads_per_class"] = {c: sorted(tr.meta_workload[tr.label == c].unique()) for c in classes}

	sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
	splits = list(sgkf.split(tr, y, groups))
	R["fold_sites"] = {c: [int(tr.meta_site[te][tr.label[te] == c].nunique()) for _, te in splits] for c in classes}

	# ---- variant A, group CV
	pA, foldA, itA = cross_val(tr, feats_A, classes, splits, groups)
	predA = np.array(classes)[pA.argmax(1)]
	R["A"] = {"iters": itA, "subsets": by_subset(tr, predA, classes), "per_fold_f1": per_fold_f1(tr, predA, foldA, classes)}
	R["A"]["pairs_sdc"] = confused_pairs(R["A"]["subsets"]["sdc"])
	R["A"]["pairs_crash"] = confused_pairs(R["A"]["subsets"]["crash_timeout"])
	R["A"]["per_workload"] = {w: metrics(tr.label[tr.meta_workload == w].to_numpy(), predA[(tr.meta_workload == w).to_numpy()], classes)
		for w in sorted(tr.meta_workload.unique())}
	oof = pd.DataFrame({"run_id": tr.meta_run_id, "label": tr.label, "workload": tr.meta_workload, "site": tr.meta_site,
		"outcome": tr.meta_outcome_class_for_rates, "fold": foldA, "pred": predA})
	for i, c in enumerate(classes):
		oof["p_" + c] = pA[:, i]
	oof.to_parquet(os.path.join(REPORTS, "oof_A.parquet"), index=False)

	# memory descriptor: ur (LDCU of the descriptor, uniform group) and addr (addr_part = desc)
	desc = ((tr.meta_memory_descriptor == "1") | (tr.meta_addr_part == "desc")).to_numpy()
	R["A"]["descriptor"] = {"runs": int(desc.sum()),
		"table": pd.crosstab(tr.label[desc].to_numpy(), predA[desc]).to_dict(),
		"outcomes": tr.meta_outcome_class_for_rates[desc].value_counts().to_dict(),
		"by_label": tr.label[desc].value_counts().to_dict()}
	nd = ~desc
	R["A"]["without_descriptor"] = by_subset(tr[nd].reset_index(drop=True), predA[nd], classes)

	# ---- variant B, group CV
	pB, foldB, itB = cross_val(tr, feats_B, classes, splits, groups)
	predB = np.array(classes)[pB.argmax(1)]
	R["B"] = {"iters": itB, "subsets": by_subset(tr, predB, classes), "per_fold_f1": per_fold_f1(tr, predB, foldB, classes)}
	R["B"]["pairs_sdc"] = confused_pairs(R["B"]["subsets"]["sdc"])
	R["B"]["pairs_crash"] = confused_pairs(R["B"]["subsets"]["crash_timeout"])

	# ---- variant B2, group CV
	pB2, foldB2, itB2 = cross_val(tr, feats_B2, classes, splits, groups)
	predB2 = np.array(classes)[pB2.argmax(1)]
	R["B2"] = {"iters": itB2, "subsets": by_subset(tr, predB2, classes), "per_fold_f1": per_fold_f1(tr, predB2, foldB2, classes)}
	R["B2"]["pairs_sdc"] = confused_pairs(R["B2"]["subsets"]["sdc"])
	R["B2"]["pairs_crash"] = confused_pairs(R["B2"]["subsets"]["crash_timeout"])

	# ---- variants B and B2, leave-one-workload-out on classes in >= 2 workloads
	multi = [c for c in classes if len(R["workloads_per_class"][c]) >= 2]
	R["lowo_classes"], R["lowo_excluded"] = multi, [c for c in classes if c not in multi]
	sub = tr[tr.label.isin(multi)].reset_index(drop=True)
	ys = sub.label.map({c: i for i, c in enumerate(multi)}).to_numpy()
	for name, fs in (("B_lowo", feats_B), ("B2_lowo", feats_B2)):
		predL = np.empty(len(sub), dtype=object)
		R[name] = {"per_workload": {}}
		for w in sorted(sub.meta_workload.unique()):
			te = (sub.meta_workload == w).to_numpy()
			m = fit(sub[fs][~te], ys[~te], sub.meta_site.to_numpy()[~te], len(multi))
			pw = np.array(multi)[m.predict(sub[fs][te], num_iteration=m.best_iteration).argmax(1)]
			predL[te] = pw
			R[name]["per_workload"][w] = by_subset(sub[te].reset_index(drop=True), pw, multi)
		R[name]["pooled"] = by_subset(sub, predL.astype(str), multi)
		R[name]["pairs_sdc"] = confused_pairs(R[name]["pooled"]["sdc"])

	# ---- final model A on all training: SHAP and MCU test
	rounds = int(np.median(itA))
	mA = fit_rounds(tr[feats_A], y, k, rounds)
	R["final_rounds"] = rounds
	R["shap"] = shap_importance(mA, tr[feats_A], classes)
	pm = np.array(classes)[mA.predict(mcu[feats_A]).argmax(1)]
	R["mcu"] = {"subsets": by_subset(mcu, pm, classes),
		"per_class": {c: {"runs": int((mcu.label == c).sum()), "accuracy": float((pm[mcu.label == c] == c).mean()) if (mcu.label == c).any() else None}
			for c in classes},
		"per_model": {fm: float((pm[mcu.meta_fault_model == fm] == mcu.label[mcu.meta_fault_model == fm]).mean())
			for fm in sorted(mcu.meta_fault_model.unique())}}
	R["mcu"]["per_class_model"] = {c: {fm: [int(((mcu.label == c) & (mcu.meta_fault_model == fm)).sum()),
		float((pm[((mcu.label == c) & (mcu.meta_fault_model == fm)).to_numpy()] == c).mean()) if ((mcu.label == c) & (mcu.meta_fault_model == fm)).any() else None]
		for fm in sorted(mcu.meta_fault_model.unique())} for c in classes}
	R["seconds"] = time.time() - t0
	json.dump(R, open(os.path.join(REPORTS, "baseline-%s.json" % date), "w"), indent=1, default=str)
	print("done in %.0f s" % R["seconds"])


if __name__ == "__main__":
	main()
