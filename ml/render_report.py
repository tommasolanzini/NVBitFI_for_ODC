#!/usr/bin/env python3
# Tables of the stage-1 baseline report, from ml/reports/baseline-<date>.json.
#
#   ~/thesis/venv-ml/bin/python ml/render_report.py <date>   -> ml/reports/baseline-<date>-tables.md
#
# The commented report (baseline-<date>.md) is written by hand from these tables.

import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def cm_table(m):
	labels, cmn, cm = m["labels"], np.array(m["confusion_norm"]), np.array(m["confusion"])
	used = [j for j in range(len(labels)) if cm[:, j].sum() > 0 or cm[j].sum() > 0]
	L = ["| true \\ pred | " + " | ".join(labels[j] for j in used) + " | runs |", "|" + "---|" * (len(used) + 2)]
	for i in used:
		if cm[i].sum() == 0:
			continue
		cells = []
		for j in used:
			v = cmn[i, j]
			s = "%.2f" % v if v >= 0.005 else ("<.01" if v > 0 else ".")
			cells.append("**%s**" % s if i == j else s)
		L.append("| %s | %s | %d |" % (labels[i], " | ".join(cells), cm[i].sum()))
	return "\n".join(L)


def subsets_table(res):
	L = ["| subset | runs | macro-F1 | accuracy |", "|---|---|---|---|"]
	for s, m in res.items():
		L.append("| %s | %d | %.3f | %.3f |" % (s, m["n"], m["macro_f1"], m["accuracy"]))
	return "\n".join(L)


def pairs_table(pairs):
	L = ["| pair a ↔ b | a→b | b→a | runs a→b | runs b→a |", "|---|---|---|---|---|"]
	for p in pairs:
		L.append("| %s ↔ %s | %.2f | %.2f | %d | %d |" % (p["a"], p["b"], p["a_as_b"], p["b_as_a"], p["n_a_as_b"], p["n_b_as_a"]))
	return "\n".join(L)


def f1_table(R, variant):
	L = ["| class | sites | sites per fold | F1 per fold | mean ± std |", "|---|---|---|---|---|"]
	for c in R["classes"]:
		p = R[variant]["per_fold_f1"][c]
		vals = " ".join("%.2f" % v for v in p["values"])
		ms = "%.2f ± %.2f" % (p["mean"], p["std"]) if p["mean"] is not None else "-"
		L.append("| %s | %d | %s | %s | %s |" % (c, R["sites"][c], " ".join(map(str, R["fold_sites"][c])), vals, ms))
	return "\n".join(L)


def class_f1_compare(R):
	L = ["| class | A sdc | A crash | B sdc | B crash | B2 sdc | B2 crash |", "|---|---|---|---|---|---|---|"]
	for c in R["classes"]:
		row = []
		for v in ("A", "B", "B2"):
			for s in ("sdc", "crash_timeout"):
				x = R[v]["subsets"][s]["f1_per_class"].get(c)
				row.append("%.2f" % x if x is not None else "-")
		L.append("| %s | %s |" % (c, " | ".join(row)))
	return "\n".join(L)


def main():
	date = sys.argv[1]
	R = json.load(open(os.path.join(HERE, "reports/baseline-%s.json" % date)))
	o = []
	o.append("## Data\n\nTraining (single bit): %d runs, %d classes. MCU test: %d runs. Final model: %d rounds.\n" % (R["rows_train"], len(R["classes"]), R["rows_mcu"], R["final_rounds"]))
	o.append("| class | runs | SDC | crash+timeout | sites | workloads |\n|---|---|---|---|---|---|")
	for c in R["classes"]:
		r = R["runs"][c]
		o.append("| %s | %d | %d | %d | %d | %s |" % (c, r["total"], r["sdc"], r["crash_timeout"], R["sites"][c], ", ".join(R["workloads_per_class"][c])))
	for v, name in (("A", "Variant A (all f_ + fw_), StratifiedGroupKFold by site"), ("B", "Variant B (no fw_, no workload-revealing f_), StratifiedGroupKFold by site"),
			("B2", "Variant B2 (B without f_frac_completed_correct), StratifiedGroupKFold by site")):
		o.append("\n## %s\n\nBest iterations per fold: %s\n" % (name, R[v]["iters"]))
		o.append(subsets_table(R[v]["subsets"]))
		for s in ("sdc", "crash_timeout", "all"):
			o.append("\n### %s %s: confusion, row-normalised\n\n%s" % (v, s, cm_table(R[v]["subsets"][s])))
		o.append("\n### %s: 10 most confused pairs, SDC\n\n%s" % (v, pairs_table(R[v]["pairs_sdc"])))
		o.append("\n### %s: 10 most confused pairs, crash+timeout\n\n%s" % (v, pairs_table(R[v]["pairs_crash"])))
		o.append("\n### %s: F1 per fold (all runs)\n\n%s" % (v, f1_table(R, v)))
	o.append("\n## F1 per class, A vs B\n\n" + class_f1_compare(R))
	o.append("\n## A per workload\n\n| workload | runs | macro-F1 | accuracy |\n|---|---|---|---|")
	for w, m in R["A"]["per_workload"].items():
		o.append("| %s | %d | %.3f | %.3f |" % (w, m["n"], m["macro_f1"], m["accuracy"]))
	o.append("\n## A on memory-descriptor runs\n\n%d runs, labels %s, outcomes %s\n\nlabel → predicted: %s" % (R["A"]["descriptor"]["runs"], R["A"]["descriptor"]["by_label"], R["A"]["descriptor"]["outcomes"], json.dumps(R["A"]["descriptor"]["table"])))
	o.append("\nA without descriptor runs:\n\n" + subsets_table(R["A"]["without_descriptor"]))
	o.append("\nB2 leave-one-workload-out, pooled:\n\n" + subsets_table(R["B2_lowo"]["pooled"]))
	o.append("\n## B leave-one-workload-out\n\nClasses: %s. Excluded (one workload): %s.\n" % (", ".join(R["lowo_classes"]), ", ".join(R["lowo_excluded"])))
	o.append("Pooled:\n\n" + subsets_table(R["B_lowo"]["pooled"]))
	o.append("\n| held-out workload | sdc runs | sdc macro-F1 | sdc acc | crash runs | crash macro-F1 | crash acc |\n|---|---|---|---|---|---|---|")
	for w, m in R["B_lowo"]["per_workload"].items():
		s, c = m["sdc"], m["crash_timeout"]
		o.append("| %s | %d | %.3f | %.3f | %d | %.3f | %.3f |" % (w, s["n"], s["macro_f1"], s["accuracy"], c["n"], c["macro_f1"], c["accuracy"]))
	o.append("\n### LOWO pooled SDC confusion\n\n" + cm_table(R["B_lowo"]["pooled"]["sdc"]))
	o.append("\n### LOWO pooled crash confusion\n\n" + cm_table(R["B_lowo"]["pooled"]["crash_timeout"]))
	o.append("\n### LOWO: 10 most confused pairs, SDC\n\n" + pairs_table(R["B_lowo"]["pairs_sdc"]))
	o.append("\n## SHAP (variant A, final model)\n\nGlobal, mean |SHAP| summed over classes:\n\n| feature | importance |\n|---|---|")
	for f, v in R["shap"]["global"][:20]:
		o.append("| %s | %.3f |" % (f, v))
	o.append("\nPer class (mean |SHAP| of that class's output):\n\n| class | top features |\n|---|---|")
	for c, l in R["shap"]["per_class"].items():
		o.append("| %s | %s |" % (c, ", ".join("%s %.2f" % (f, v) for f, v in l[:6])))
	o.append("\n## MCU test (variant A trained on all single-bit runs)\n\n" + subsets_table(R["mcu"]["subsets"]))
	models = sorted(next(iter(R["mcu"]["per_class_model"].values())).keys())
	o.append("\n| class | runs | accuracy | " + " | ".join(models) + " |\n|---|---|---|" + "---|" * len(models))
	for c, v in R["mcu"]["per_class"].items():
		if not v["runs"]:
			continue
		cells = []
		for fm in models:
			n, a = R["mcu"]["per_class_model"][c][fm]
			cells.append("%.2f (%d)" % (a, n) if n else "-")
		o.append("| %s | %d | %.2f | %s |" % (c, v["runs"], v["accuracy"], " | ".join(cells)))
	o.append("\nAccuracy per fault model: " + ", ".join("%s %.3f" % kv for kv in R["mcu"]["per_model"].items()))
	o.append("\n### MCU confusion, all\n\n" + cm_table(R["mcu"]["subsets"]["all"]))
	open(os.path.join(HERE, "reports/baseline-%s-tables.md" % date), "w").write("\n".join(o) + "\n")


if __name__ == "__main__":
	main()
