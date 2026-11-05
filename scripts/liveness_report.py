#!/usr/bin/env python3
# Reads liveness.jsonl (liveness/ tool). No GPU.
#
#   liveness_report.py occupancy <liveness.jsonl>
#   liveness_report.py reweight <liveness.jsonl> <results-*.jsonl> ...
#
# occupancy: share of bit x time that is live, per kernel and register class.
# reweight:  outcome rates per group, from sampling writes to sampling bit x time.
#            A campaign draws one dynamic instruction, then one of its destinations,
#            then one bit. So a value is drawn with a probability that does not depend
#            on how long it lives. Weight of a run:
#                mean live time of the site x destinations of the instruction x bits
# See "Register liveness" in README.md for what these numbers do not include.

import collections, json, sys

CLASSES = ["GPR", "PR", "UR", "UP"]
BITS = {"GPR": 32, "PR": 1, "UR": 32, "UP": 1}
PRED_REGS = 7
# Spec of the GB207 (CLAUDE.md), not measured: only for the share of the whole file
RF_BYTES_PER_SM = 256 * 1024
# Destination classes a group can draw, as in inject_error
GROUP_CLASSES = {"gp": ("GPR",), "gppr": ("GPR", "PR"), "addr": (), "addr_store": ()}
SDC = (16, 17, 18)
MASKED = (1, 2, 3)
DOMINATED_SHARE = 0.5


def load(path):
	with open(path) as f:
		return [json.loads(l) for l in f if l.strip()]


def class_name(site):
	return CLASSES[site["cls"]]


# Live bit x time and the time it is compared with, for one launch
def occupancy(rec):
	out = {}
	per_thread = {"GPR": True, "PR": True, "UR": False, "UP": False}
	sizes = {"GPR": [("used", rec["used_gpr"]), ("allocated", rec["num_registers"])], "PR": [("all", PRED_REGS)], "UR": [("used", rec["used_ur"])], "UP": [("all", PRED_REGS)]}
	for cls in CLASSES:
		sites = [s for s in rec["sites"] if class_name(s) == cls]
		live = sum(s["live"] for s in sites)
		cycles = sum(s["live_cycles"] for s in sites)
		time = rec["thread_instrs"] if per_thread[cls] else rec["warp_instrs"]
		time_c = rec["thread_cycles"] if per_thread[cls] else rec["warp_cycles"]
		entry = {"writes": sum(s["count"] for s in sites), "dead_writes": sum(s["dead"] for s in sites), "live_instrs": live, "live_cycles": cycles}
		for name, regs in sizes[cls]:
			entry["share_" + name] = live / (regs * time) if regs and time else 0.0
			entry["share_cycles_" + name] = cycles / (regs * time_c) if regs and time_c else 0.0
		out[cls] = entry
	return out


# Share of the register file of the SMs that holds a live GPR bit, in cycles. Needs the
# residents of each SM, from the first and last cycle of each warp.
def rf_share(rec, rf_bytes=RF_BYTES_PER_SM):
	span = sum(s["span"] for s in rec["sm"])
	warp_cycles = sum(s["warp_cycles"] for s in rec["sm"])
	live = sum(s["live_cycles"] for s in rec["sites"] if class_name(s) == "GPR")
	return {"mean_resident_warps_per_sm": warp_cycles / span if span else 0.0,
		"allocated_share": warp_cycles * 32 * rec["num_registers"] * 4 / (rf_bytes * span) if span else 0.0,
		"live_share": live * 32 / (rf_bytes * 8 * span) if span else 0.0}


# {(kernel, pc, class, reg): site}, launches of a kernel summed
def site_table(records):
	table = {}
	for rec in records:
		for s in rec["sites"]:
			key = (rec["kernel"], s["pc"], class_name(s), s["reg"])
			if key in table:
				for k in ("count", "live", "live_cycles", "reads", "dead"):
					table[key][k] += s[k]
			else:
				table[key] = dict(s)
	return table


def destinations(table, kernel, pc, classes):
	return [k for k in table if k[0] == kernel and k[1] == pc and (classes is None or k[2] in classes)]


# Weight of one injection, None if the liveness run has no such site
def weight(table, r):
	pc, cls, reg = int(r["pc"], 16), r["reg_type"], r["reg_no"]
	key = (r["kernel"], pc, cls, reg)
	if key not in table:
		return None
	s = table[key]
	classes = GROUP_CLASSES.get(r["group"])
	n = len(destinations(table, r["kernel"], pc, classes)) or 1
	return s["live"] / s["count"] * n * BITS[cls]


# Live time of a sampled instance: of its site, not of that dynamic value. The same
# inst_id does not hit the same thread in two runs, so a site is the finest match.
def sample_row(table, r):
	key = (r["kernel"], int(r["pc"], 16), r["reg_type"], r["reg_no"])
	s = table.get(key)
	row = {"group": r["group"], "bfm": r.get("bfm"), "icount": r.get("icount"), "kernel": r["kernel"], "pc": r["pc"], "reg": "%s%d" % (r["reg_type"], r["reg_no"]), "outcome": outcome_class(r["outcome"])}
	if s:
		row.update({"site_writes": s["count"], "site_dead_writes": s["dead"], "live_instrs_mean": s["live"] / s["count"], "live_cycles_mean": s["live_cycles"] / s["count"], "reads_mean": s["reads"] / s["count"], "weight": weight(table, r)})
	return row


def outcome_class(cat):
	return "sdc" if cat in SDC else "masked" if cat in MASKED else "crash" if cat == 20 else "timeout" if cat == 4 else "other"


# Rates per group. Dead: never read in the liveness run, yet the outcome changed. Either
# the site is not always dead, or a read is missing from the analysis, or a tool artifact.
def reweight(table, results):
	groups = collections.defaultdict(list)
	for r in results:
		if r.get("reg_type") in BITS and r.get("injected"):
			groups[r["group"]].append(r)
	out = {}
	for group, rows in sorted(groups.items()):
		raw, weighted = collections.Counter(), collections.Counter()
		total_w = sum_w2 = 0.0
		missing = 0
		dead_effects = []
		by_site = collections.Counter()
		for r in rows:
			c = outcome_class(r["outcome"])
			raw[c] += 1
			w = weight(table, r)
			if w is None:
				missing += 1
				continue
			weighted[c] += w
			total_w += w
			sum_w2 += w * w
			by_site[(r["kernel"][:24], r["pc"], "%s%d" % (r["reg_type"], r["reg_no"]))] += w
			s = table[(r["kernel"], int(r["pc"], 16), r["reg_type"], r["reg_no"])]
			if c != "masked" and s["live"] == 0:
				dead_effects.append({"kernel": r["kernel"][:24], "pc": r["pc"], "reg": "%s%d" % (r["reg_type"], r["reg_no"]), "outcome": c, "icount": r.get("icount")})
		out[group] = {"runs": len(rows), "missing_sites": missing, "raw": dict(raw),
			"weighted": {k: v / total_w for k, v in weighted.items()} if total_w else {},
			"effective_runs": total_w * total_w / sum_w2 if sum_w2 else 0.0, "dead_effects": dead_effects}
		# A rate that one site decides describes that site, not the group
		if by_site and total_w:
			site_id, w = by_site.most_common(1)[0]
			out[group]["top_site"] = {"site": list(site_id), "share": w / total_w, "dominated": w / total_w > DOMINATED_SHARE}
	return out


def main():
	cmd, path = sys.argv[1], sys.argv[2]
	records = load(path)
	if cmd == "occupancy":
		out = []
		for rec in records:
			occ = occupancy(rec)
			rf = rf_share(rec)
			out.append({"kernel": rec["kernel"], "launch": rec["launch"], "classes": occ, "rf": rf, "reads_no_write": rec["reads_no_write"], "lane_mismatch": rec["lane_mismatch"]})
			print(rec["kernel"][:40], "launch", rec["launch"], "regs used/alloc %d/%d" % (rec["used_gpr"], rec["num_registers"]), "reads_no_write", rec["reads_no_write"], "lane_mismatch", rec["lane_mismatch"])
			for cls in CLASSES:
				e = occ[cls]
				print("  %-3s writes %10d dead %10d" % (cls, e["writes"], e["dead_writes"]), " ".join("%s %.3f" % (k[6:], v) for k, v in e.items() if k.startswith("share")))
			print("  file: residents per SM %.1f, allocated %.3f, live %.4f (spec %d KiB per SM)" % (rf["mean_resident_warps_per_sm"], rf["allocated_share"], rf["live_share"], RF_BYTES_PER_SM // 1024))
		json.dump(out, open(path + ".occupancy.json", "w"), indent=1)
	elif cmd == "reweight":
		results = []
		for p in sys.argv[3:]:
			results += load(p)
		out = reweight(site_table(records), results)
		json.dump(out, open(path + ".reweight.json", "w"), indent=1)
		table = site_table(records)
		with open(path + ".samples.jsonl", "w") as f:
			for r in results:
				if r.get("reg_type") in BITS and r.get("injected"):
					f.write(json.dumps(sample_row(table, r)) + "\n")
		for g, e in out.items():
			print(g, "runs", e["runs"], "missing", e["missing_sites"], "raw", e["raw"], "weighted", {k: round(v, 3) for k, v in e["weighted"].items()}, "effective %.0f" % e["effective_runs"], "dead effects", len(e["dead_effects"]), ("DOMINATED by %s, %.0f%% of the weight" % (e["top_site"]["site"], 100 * e["top_site"]["share"])) if e.get("top_site", {}).get("dominated") else "")
			for d in e["dead_effects"][:8]:
				print("   ", d)
	else:
		sys.exit("unknown command")


if __name__ == "__main__":
	main()
