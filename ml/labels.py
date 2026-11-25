#!/usr/bin/env python3
# Stage-1 labels from ml/label_map.yaml.
#
# Meaning of a label: GPR destination = functional unit that produced the value, from the
# opcode (unit output, pipeline latches and the register-file cell just written are
# architecturally indistinguishable); PR/UR/UP = register class, whatever the opcode;
# addr = address path at the moment of use (AGU and latches towards the LSU), memory
# descriptor included; mem_global/mem_shared = memory structure.
#
#   python3 ml/labels.py [runs.csv]   prints label counts and the unmapped opcodes
#
# label(row) takes a dict with the columns of dataset/runs.csv and returns the class
# name, or None (unmapped opcode or no rule). select(row) says whether the row enters
# stage 1 and split(row) returns "train", "mcu_test" or None.

import collections, csv, os, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "logs/final-2026-11-05/dataset/runs.csv")

with open(os.path.join(HERE, "label_map.yaml")) as f:
	MAP = yaml.safe_load(f)

OPCODE = {}
for cls, lists in MAP["opcode_classes"].items():
	for op in lists.get("seen", []) + lists.get("other", []):
		assert op not in OPCODE, "opcode %s in two classes" % op
		OPCODE[op] = cls
assert set(MAP["opcode_classes"]) <= set(MAP["classes"])


def _matches(row, cond):
	for k, v in cond.items():
		vals = v if isinstance(v, list) else [v]
		if str(row.get(k, "")) not in [str(x) for x in vals]:
			return False
	return True


def base_opcode(op):
	return op.split(".")[0] if op else ""


def label(row):
	for rule in MAP["rules"]:
		if _matches(row, rule["match"]):
			if rule["label"] == "from_opcode":
				return OPCODE.get(base_opcode(row.get("opcode", "")))
			return rule["label"]
	return None


def select(row):
	s = MAP["selection"]
	if row.get("outcome_class_for_rates") not in s["visible_outcomes"]:
		return False
	if not _matches(row, s["require"]):
		return False
	return not any(_matches(row, e) for e in s["exclude"])


def split(row):
	for name, cond in MAP["splits"].items():
		if _matches(row, cond):
			return name
	return None


def trainable(cls):
	return cls is not None and MAP["classes"][cls]["train"]


def main():
	path = sys.argv[1] if len(sys.argv) > 1 else RUNS
	counts, unmapped = collections.Counter(), collections.Counter()
	for r in csv.DictReader(open(path)):
		if not select(r):
			continue
		c = label(r)
		if c is None:
			unmapped[(r["injector"], r["group"], r["reg_class"], r["opcode"])] += 1
		counts[(split(r), c, r["outcome_class_for_rates"])] += 1
	for sp in ("train", "mcu_test"):
		print("== %s" % sp)
		for cls in MAP["classes"]:
			n = {o: counts[(sp, cls, o)] for o in ("sdc", "crash", "timeout")}
			tot = sum(n.values())
			flag = "" if MAP["classes"][cls]["train"] else "  (not trained: %s)" % MAP["classes"][cls].get("status", "")
			print("  %-10s %6d   sdc %5d  crash %5d  timeout %3d%s" % (cls, tot, n["sdc"], n["crash"], n["timeout"], flag))
	print("unmapped:", dict(unmapped) if unmapped else "none")


if __name__ == "__main__":
	main()
