#!/usr/bin/env python3
# Xid-detail features (ml/xid_features.py): A vs A+xid and B2 vs B2+xid, with the folds,
# seed and parameters of ml/train_gbdt.py.
#
#   ~/thesis/venv-ml/bin/python ml/train_xid.py [date]  -> ml/reports/xid-features-<date>.json

import datetime, json, os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_gbdt as T  # noqa: E402
import xid_features as XF  # noqa: E402

FOCUS = ["int", "ld_const", "ld_data", "addr", "move", "mem_global", "ur"]
DIST = ["f_xid13_oor", "f_xid13_misaligned", "f_xid13_invalid_space", "f_xid13_illegal_param", "f_xid13_other",
	"f_xid13_n_sm", "f_xid31_read", "f_xid31_write", "f_xid31_low23_in_buf", "f_xid31_bit_to_buf",
	"f_xid31_hamming_min", "f_xid31_above_2_47"]


def crash_rows(d):
	return (d.f_out_sdc == 0).to_numpy()


def main():
	date = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()
	full = pd.read_parquet(T.DATA)
	tr = full[(full.split == "train") & (full.trainable == 1)].reset_index(drop=True)
	classes = sorted(tr.label.unique())
	k = len(classes)
	y = tr.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
	groups = tr.meta_site.to_numpy()
	xid = list(XF.FEATURES)
	f_all = [c for c in tr.columns if c.startswith("f_") and c not in xid]
	fw = [c for c in tr.columns if c.startswith("fw_")]
	feats = {"A": f_all + fw, "B2": [c for c in f_all if c not in T.B2_DROP]}
	feats["A+xid"] = feats["A"] + xid
	feats["B2+xid"] = feats["B2"] + xid
	splits = list(StratifiedGroupKFold(n_splits=T.N_FOLDS, shuffle=True, random_state=T.SEED).split(tr, y, groups))
	R = {"date": date, "rows_train": len(tr), "classes": classes, "xid_features": xid, "variants": {}}
	cm = crash_rows(tr)
	preds = {}
	for v, fs in feats.items():
		p, fold, it = T.cross_val(tr, fs, classes, splits, groups)
		pred = np.array(classes)[p.argmax(1)]
		preds[v] = pred
		res = {"iters": it, "subsets": T.by_subset(tr, pred, classes)}
		res["pairs_crash"] = T.confused_pairs(res["subsets"]["crash_timeout"])
		res["crash_per_workload"] = {w: T.metrics(tr.label[cm & (tr.meta_workload == w).to_numpy()].to_numpy(),
			pred[cm & (tr.meta_workload == w).to_numpy()], classes) for w in sorted(tr.meta_workload.unique())}
		nptr = cm & (tr.meta_workload != "ptr_chase").to_numpy()
		res["crash_without_ptr_chase"] = T.metrics(tr.label[nptr].to_numpy(), pred[nptr], classes)
		if v.endswith("+xid"):
			tr["oof_" + v] = pred
		R["variants"][v] = res
		print(v, {s: round(m["macro_f1"], 3) for s, m in res["subsets"].items()}, flush=True)
	# SDC predictions must not change: the new features are NaN on every SDC
	sdc = ~cm
	R["sdc_changed"] = {"A": int((preds["A"][sdc] != preds["A+xid"][sdc]).sum()), "B2": int((preds["B2"][sdc] != preds["B2+xid"][sdc]).sum())}

	# SHAP on crashes, final A+xid model (rounds = median of its folds)
	rounds = int(np.median(R["variants"]["A+xid"]["iters"]))
	m = T.fit_rounds(tr[feats["A+xid"]], y, k, rounds)
	cr = tr[cm].reset_index(drop=True)
	R["shap_crash"] = T.shap_importance(m, cr[feats["A+xid"]], classes)

	# distribution of the new features per true class, crash rows
	dist = {}
	for c in classes:
		x = cr[cr.label == c]
		if not len(x):
			continue
		e = {"crash_runs": len(x), "with_xid13": int(x.f_xid13_oor.notna().sum()), "with_xid31_addr": int(x.f_xid31_read.notna().sum()),
			"with_addr_test": int(x.f_xid31_low23_in_buf.notna().sum())}
		for f in DIST:
			col = x[f].dropna()
			e[f] = float(col.mean()) if len(col) else None
		ks = x.f_xid31_k.dropna().astype(int)
		e["k_hist"] = {int(a): int(b) for a, b in ks.value_counts().sort_index().items()}
		e["k_rel_hist"] = {int(a): int(b) for a, b in x.f_xid31_k_rel.dropna().astype(int).value_counts().sort_index().items()}
		e["k_ptr_only"] = float(x.f_xid31_k_ptr_only.dropna().mean()) if x.f_xid31_k_ptr_only.notna().any() else None
		e["k_ambiguous"] = float(x.f_xid31_k_ambiguous.dropna().mean()) if x.f_xid31_k_ambiguous.notna().any() else None
		dist[c] = e
	R["distribution"] = dist
	R["coverage_by_app"] = {a: {"crash": int(len(g)), "addr_test": int(g.f_xid31_low23_in_buf.notna().sum()), "xid31_addr": int(g.f_xid31_read.notna().sum())}
		for a, g in cr.groupby("meta_app")}
	json.dump(R, open(os.path.join(T.REPORTS, "xid-features-%s.json" % date), "w"), indent=1, default=str)


if __name__ == "__main__":
	main()
