#!/usr/bin/env python3
# Offline test of scripts/liveness_report.py: arithmetic on a hand-made kernel. No GPU.
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import liveness_report as lr

passed = failed = 0


def check(name, got, want):
	global passed, failed
	ok = abs(got - want) < 1e-9 if isinstance(want, float) else got == want
	passed += ok
	failed += not ok
	if not ok:
		print("FAIL %s: got %r, expected %r" % (name, got, want))


def site(pc, cls, reg, count, live, dead=0, cycles=0, group="gp"):
	return {"pc": pc, "cls": cls, "reg": reg, "group": group, "count": count, "live": live, "live_cycles": cycles, "reads": count - dead, "dead": dead, "sass": ""}


# 2 threads of 10 instructions in a warp of 10 instructions
rec = {"kernel": "k", "launch": 0, "used_gpr": 4, "used_ur": 2, "num_registers": 8, "thread_instrs": 20, "thread_cycles": 200, "warp_instrs": 10, "warp_cycles": 100,
	"sm": [{"sm": 0, "span": 100, "warp_cycles": 100}],
	"sites": [site(0, 0, 1, 2, 0, dead=2), site(16, 0, 2, 2, 8, cycles=80), site(16, 0, 3, 2, 12, cycles=120), site(32, 1, 0, 2, 7), site(48, 2, 4, 1, 5), site(64, 3, 0, 1, 7)]}
occ = lr.occupancy(rec)
check("GPR share of used", occ["GPR"]["share_used"], 20 / (4 * 20))
check("GPR share of allocated", occ["GPR"]["share_allocated"], 20 / (8 * 20))
check("GPR share in cycles", occ["GPR"]["share_cycles_used"], 200 / (4 * 200))
check("dead writes", occ["GPR"]["dead_writes"], 2)
check("predicate share", occ["PR"]["share_all"], 7 / (7 * 20))
check("UR on warp time", occ["UR"]["share_used"], 5 / (2 * 10))
check("UP on warp time", occ["UP"]["share_all"], 7 / (7 * 10))
rf = lr.rf_share(rec, rf_bytes=1024)
check("residents", rf["mean_resident_warps_per_sm"], 1.0)
check("allocated share of the file", rf["allocated_share"], 100 * 32 * 8 * 4 / (1024 * 100))
check("live share of the file", rf["live_share"], 200 * 32 / (1024 * 8 * 100))

table = lr.site_table([rec, rec])
check("launches summed", table[("k", 16, "GPR", 2)]["live"], 16)


def run(pc, reg, cat, group="gp", cls="GPR"):
	return {"kernel": "k", "pc": hex(pc), "reg_type": cls, "reg_no": reg, "group": group, "outcome": cat, "injected": True, "icount": 1}


# Mean live 4 on R2, 2 destinations in gp
check("weight has mean live, destinations, bits", lr.weight(table, run(16, 2, 17)), 4.0 * 2 * 32)
check("predicate has 1 bit", lr.weight(table, run(32, 0, 17, group="pr", cls="PR")), 3.5 * 1 * 1)
check("gp ignores the predicate of the instruction", len(lr.destinations(table, "k", 16, ("GPR",))), 2)
check("unknown site", lr.weight(table, run(999, 2, 17)), None)

# One SDC on a long-lived site and one Masked on a short one: the SDC counts for more
long_short = {"sites": [site(0, 0, 0, 1, 90), site(16, 0, 0, 1, 10)], "kernel": "k"}
t2 = lr.site_table([long_short])
out = lr.reweight(t2, [run(0, 0, 17), run(16, 0, 1)])["gp"]
check("raw is 50/50", out["raw"], {"sdc": 1, "masked": 1})
check("one site decides", (out["top_site"]["dominated"], out["top_site"]["site"][1]), (True, "0x0"))
check("share of that site", out["top_site"]["share"], 0.9)
even = lr.reweight(lr.site_table([{"kernel": "k", "sites": [site(0, 0, 0, 1, 10), site(16, 0, 0, 1, 10), site(32, 0, 0, 1, 10)]}]), [run(0, 0, 17), run(16, 0, 1), run(32, 0, 1)])["gp"]
check("no site decides", even["top_site"]["dominated"], False)
check("weighted sdc", out["weighted"]["sdc"], 0.9)
check("effective runs", out["effective_runs"], 100.0 ** 2 * (32 ** 2) / ((90.0 ** 2 + 10.0 ** 2) * (32 ** 2)))

# A dead site with an effect is reported, a dead site that is Masked is not
t3 = lr.site_table([{"kernel": "k", "sites": [site(0, 0, 1, 2, 0, dead=2), site(16, 0, 2, 2, 8)]}])
out = lr.reweight(t3, [run(0, 1, 20), run(0, 1, 1), run(16, 2, 17)])["gp"]
check("dead effect found", len(out["dead_effects"]), 1)
check("dead effect is the crash", out["dead_effects"][0]["outcome"], "crash")
check("not injected is left out", lr.reweight(t3, [dict(run(16, 2, 17), injected=False)]), {})
row = lr.sample_row(t3, run(16, 2, 17))
check("sample has the mean live", row["live_instrs_mean"], 4.0)
check("sample of unknown site has no weight", "weight" in lr.sample_row(t3, run(999, 2, 17)), False)

print("%d/%d liveness cases passed" % (passed, passed + failed))
sys.exit(1 if failed else 0)
