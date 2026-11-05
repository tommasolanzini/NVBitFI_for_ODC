# Offline tests of scripts/mem_plan.py. Usage: python3 tests/test_mem_plan.py
import collections, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import mem_plan as mp

passed = failed = 0

def check(name, got, expected):
	global passed, failed
	if got == expected:
		passed += 1
	else:
		failed += 1
		print("FAIL %s: got %r, expected %r" % (name, got, expected))

# alloc A, alloc B, copy to A, kernel, free A, kernel, copy from B, end
EV = [
	{"e": 0, "kind": "alloc", "t0": 0, "t1": 10, "id": 0, "size": 100},
	{"e": 1, "kind": "alloc", "t0": 10, "t1": 20, "id": 1, "size": 300},
	{"e": 2, "kind": "h2d", "t0": 20, "t1": 30, "src": 0, "dst": 1, "size": 100},
	{"e": 3, "kind": "kernel", "t0": 40, "t1": 60, "k": 0, "name": "k0", "instrs": 1000},
	{"e": 4, "kind": "free", "t0": 70, "t1": 80, "id": 0},
	{"e": 5, "kind": "kernel", "t0": 100, "t1": 130, "k": 1, "name": "k1", "instrs": 500},
	{"e": 6, "kind": "d2h", "t0": 140, "t1": 150, "src": 2, "dst": 0, "size": 300},
	{"e": 7, "kind": "term", "t0": 160, "t1": 160},
]

b = mp.buffers_of(EV)
check("buffer sizes", (b[0]["size"], b[1]["size"]), (100, 300))
check("free event", (b[0]["free_event"], b[1]["free_event"]), (4, None))

s, tail = mp.slots_of(EV)
check("slots", [(x["mode"], x["event"], x["duration"]) for x in s], [
	("a", 1, 10), ("a", 2, 10), ("a", 3, 20), ("b", 3, 20), ("a", 4, 10), ("a", 5, 30), ("b", 5, 30), ("a", 6, 10)])
check("tail starts at the last event", tail, 20)
check("slots plus tail cover the run", sum(x["duration"] for x in s) + tail, 160)

gap = lambda e: {"mode": "a", "event": e}
ker = lambda e: {"mode": "b", "event": e}
check("not live in the gap before its alloc", mp.is_live(b[1], gap(1)), False)
check("live after its alloc", mp.is_live(b[1], gap(2)), True)
check("live in the gap before its free", mp.is_live(b[0], gap(4)), True)
check("dead after its free", mp.is_live(b[0], gap(5)), False)
check("live in a kernel before its free", mp.is_live(b[0], ker(3)), True)
check("dead in a kernel after its free", mp.is_live(b[0], ker(5)), False)

x = mp.exposure(EV)
# A: 100 B x 8 x (a: 10+10+20+10 before e1-e4) (b: 20); B: 300 B x 8 x (a: 10+20+10+30+10 before e2-e6) (b: 20+30)
check("exposure a", round(x["between_kernels"] * 1e9), 100 * 8 * 50 + 300 * 8 * 80)
check("exposure b", round(x["during_kernels"] * 1e9), 100 * 8 * 20 + 300 * 8 * 50)
check("exposure tail: only what is still allocated", round(x["after_last_event"] * 1e9), 300 * 8 * 20)

for mode in ("a", "b"):
	ts = mp.sample_targets(EV, mode, 2000, 1)
	check("same seed, same targets " + mode, ts, mp.sample_targets(EV, mode, 2000, 1))
	check("all in mode " + mode, {t["mode"] for t in ts}, {mode})
	check("word aligned, in the buffer " + mode, all(t["offset"] % 4 == 0 and t["offset"] + 4 <= t["size"] for t in ts), True)
	check("bit range " + mode, all(0 <= t["bit"] < 32 for t in ts), True)
	sl = {(q["mode"], q["event"]): q for q in s}
	check("buffer live at the chosen time " + mode, all(mp.is_live(b[t["buffer"]], sl[(mode, t["event"])]) for t in ts), True)

ts = mp.sample_targets(EV, "b", 20000, 2)
c = collections.Counter((t["buffer"], t["event"]) for t in ts)
# weights bytes x time: (A,k0) 100x20=2000, (B,k0) 300x20=6000, (B,k1) 300x30=9000
check("weights by bits x time", [round(c[k] / 20000 * 17, 1) for k in ((0, 3), (1, 3), (1, 5))], [2.0, 6.0, 9.0])
check("instruction in range", all(t["inst"] < (1000 if t["event"] == 3 else 500) for t in ts), True)
check("nothing for a dead buffer", c[(0, 5)], 0)

info = mp.target_to_info({"mode": "b", "buffer": 1, "size": 4, "offset": 0, "bit": 3, "event": 5, "kernel": 1, "kernel_name": "k1", "inst": 7})
check("info file", info, "mode b\nbuffer 1\nsize 4\noffset 0\nbit 3\nevent 5\nkernel 1\nkernel_name k1\ninst 7\n")

timing = [dict(e) for e in EV]
for e in timing: e.pop("instrs", None)
merged = mp.merge_profiles(timing, EV)
check("merge takes times and counts", merged[3]["instrs"], 1000)
try:
	mp.merge_profiles(timing[:-1], EV)
	check("different events are rejected", False, True)
except ValueError:
	check("different events are rejected", True, True)

# Corrected kernel durations
kt = mp.kernel_timing(["LAUNCH k0 0 900.0\nLAUNCH k0 1 50.0\nLAUNCH k0 2 4.0\nnoise\nLAUNCH k1 0 70.0", "LAUNCH k0 2 6.0\nLAUNCH k1 3 2.0"])
check("only the later launches count", (kt["k0"]["launches"], kt["k0"]["mean_us"], kt["k1"]["mean_us"]), (2, 5.0, 2.0))
check("short name of a kernel", mp.kernel_short_name("k_walk(Node* const*, float*)"), "k_walk")
fast = {"k0": {"mean_us": 0.005}, "k1": {"mean_us": 0.003}}
fixed = mp.apply_kernel_timing(EV, fast)
fs, ftail = mp.slots_of(fixed)
check("kernel slots use the corrected duration", [x["duration"] for x in fs if x["mode"] == "b"], [5, 3])
check("gaps do not change", [x["duration"] for x in fs if x["mode"] == "a"], [x["duration"] for x in s if x["mode"] == "a"])
check("events are not modified", "duration" in EV[3], False)
try:
	mp.apply_kernel_timing(EV, {"k0": {"mean_us": 1.0}})
	check("a kernel without a duration is rejected", False, True)
except ValueError:
	check("a kernel without a duration is rejected", True, True)
check("weights follow the corrected duration", {t["kernel"] for t in mp.sample_targets(mp.apply_kernel_timing(EV, {"k0": {"mean_us": 1e-9}, "k1": {"mean_us": 1.0}}), "b", 50, 1)}, {EV[5]["k"]})
check("floor is subtracted, not below zero", mp.subtract_floor({"a": {"mean_us": 5.0, "launches": 2}, "b": {"mean_us": 1.0}}, 2.0), {"a": {"mean_us": 3.0, "launches": 2}, "b": {"mean_us": 0.0}})
check("mode b shares follow bits x duration", mp.mode_b_shares(EV), {"k0": 8000 / 17000, "k1": 9000 / 17000})
shm_ev = [{"e": 0, "kind": "kernel", "k": 0, "name": "k0", "t0": 0, "t1": 1000, "instrs": 100, "shm": 1024, "ctas": [[0, 1, 101]]},
	{"e": 1, "kind": "free", "t0": 1100, "t1": 1100}]
shm_fixed = mp.apply_kernel_timing(shm_ev, {"k0": {"mean_us": 0.1}})
check("run time of the SRAM uses the corrected duration", round(mp.shared_exposure(shm_fixed)["run_seconds"] * 1e9), 200)
check("residence of a CTA uses the corrected duration", mp.cta_seconds_ns(shm_fixed[0], [0, 1, 51]), 50.0)

R = lambda **k: dict({"reads_new": 0, "reads_old": 0, "host_reads": 0, "first_access": 0}, **k)
check("no access", mp.masked_subclass(R()), "no_read")
check("written first", mp.masked_subclass(R(first_access=2)), "overwritten")
check("written first, then read", mp.masked_subclass(R(first_access=2, reads_new=4)), "overwritten")
check("only stale reads", mp.masked_subclass(R(first_access=1, reads_old=9)), "stale_only")
check("one new read is enough", mp.masked_subclass(R(first_access=1, reads_old=9, reads_new=1)), "read_masked")
check("host read", mp.masked_subclass(R(first_access=1, host_reads=1)), "read_masked")
check("kernels mask", mp.kernels_from_mask(0b101), [0, 2])

# ---- shared memory ----
check("subclass ambiguous", mp.masked_subclass({"reads_new": 0, "reads_old": 0, "host_reads": 0, "first_access": mp.ACCESS_AMBIGUOUS}), "ambiguous")
# Two kernels with shared memory on two SMs. Clock: instrs; t1-t0 = 1000 ns.
SH = [
	{"e": 0, "kind": "alloc", "t0": 0, "t1": 10, "id": 0, "size": 100},
	{"e": 1, "kind": "kernel", "t0": 1000, "t1": 2000, "k": 0, "name": "k0", "instrs": 100, "shm": 400, "stores": 10,
		"ctas": [[3, 1, 51], [3, 26, 101], [7, 51, 101]]},
	{"e": 2, "kind": "kernel", "t0": 3000, "t1": 4000, "k": 1, "name": "plain", "instrs": 100},
	{"e": 3, "kind": "term", "t0": 10000, "t1": 10000},
]
check("kernels with shm", [k["k"] for k in mp.kernels_with_shm(SH)], [0])
check("cta time", mp.cta_seconds_ns(SH[1], SH[1]["ctas"][0]), 500.0)
check("peak per sm", mp.peak_ctas_per_sm(SH[1]), {3: 2, 7: 1})
x = mp.shared_exposure(SH)
resident_ns = 500 + 750 + 500
check("windows", abs(x["windows"] - 400 * 8 * resident_ns * 1e-9) < 1e-12, True)
check("reserved", abs(x["reserved"] - 1024 * 8 * resident_ns * 1e-9) < 1e-12, True)
check("peak", x["peak_ctas_on_one_sm"], 2)
check("run", abs(x["run_seconds"] - 1e-5) < 1e-15, True)
check("exposure adds up", abs(x["windows"] + x["reserved"] + x["l1_data"] - x["total"]) < 1e-9, True)
check("l1 parts", abs(x["never_shared"] + x["free_shared"] - x["l1_data"]) < 1e-9, True)
check("never shared", abs(x["never_shared"] - (128 * 1024 - 102400) * 8 * 20 * 1e-5) < 1e-9, True)

RES = [{"injected": True, "outcome": 16}] * 3 + [{"injected": True, "outcome": 20}] + [{"injected": True, "outcome": 3}] * 4 + [{"injected": False, "outcome": 19}]
EXP = {"total": 100.0, "windows": 10.0, "reserved": 5.0, "l1_data": 85.0}
a = mp.shared_avf(RES, EXP)
check("avf runs", (a["runs"], a["not_injected"], a["sdc"], a["crash"]), (8, 1, 3, 1))
check("avf injected part", a["avf_injected_part"], 0.5)
check("avf whole sram known part", a["whole_sram"]["known_part"], 0.05)
check("avf shares", (a["whole_sram"]["windows_share"], a["whole_sram"]["reserved_share"], a["whole_sram"]["l1_data_share"]), (0.1, 0.05, 0.85))
check("avf l1 marked", a["whole_sram"]["l1_data_avf"], mp.L1_NOT_REACHABLE)

u = mp.sample_uniform(SH, 20000, 5)
check("u only kernel with shm", {t["kernel"] for t in u}, {0})
check("u words after the reserved bytes", all(1024 <= t["shm_offset"] < 1024 + 400 and t["shm_offset"] % 4 == 0 for t in u), True)
check("u bits", all(0 <= t["bit"] < 32 for t in u), True)
by_sm = collections.Counter(t["sm"] for t in u)
check("u sm follows the residence", abs(by_sm[3] / len(u) - 1250 / 1750) < 0.02, True)
check("u time inside a residence", all((t["sm"] == 7 and 50 <= t["inst"] < 100) or (t["sm"] == 3 and 0 <= t["inst"] < 100) for t in u), True)
info = mp.target_to_info(u[0])
check("u info has sm and offset", "sm %d\n" % u[0]["sm"] in info and "shm_offset %d\n" % u[0]["shm_offset"] in info, True)
check("u info has pick", "pick %d\n" % u[0]["pick"] in info, True)
check("u same seed same targets", mp.sample_uniform(SH, 50, 9), mp.sample_uniform(SH, 50, 9))

s_t = mp.sample_after_store(SH, 500, 5)
check("s only kernel with stores", {t["kernel"] for t in s_t}, {0})
check("s index", all(0 <= t["inst"] < 10 for t in s_t), True)
for fn, name in ((mp.sample_uniform, "u"), (mp.sample_after_store, "s")):
	try:
		fn(SH[:1] + SH[2:], 1, 1)
		check("no shared memory raises " + name, False, True)
	except ValueError:
		check("no shared memory raises " + name, True, True)

# Correction 4: mean timeline
r1 = [{"e": 0, "kind": "alloc", "t0": 100, "t1": 110}, {"e": 1, "kind": "kernel", "t0": 200, "t1": 300, "k": 0, "name": "k"}]
r2 = [{"e": 0, "kind": "alloc", "t0": 1000, "t1": 1030}, {"e": 1, "kind": "kernel", "t0": 1300, "t1": 1500, "k": 0, "name": "k"}]
mt = mp.mean_timeline([r1, r2])
check("mean: first event at 0", mt[0]["t0"], 0)
check("mean t1 first event", mt[0]["t1"], 20)
check("mean of the gap", mt[1]["t0"], 200)
check("mean of the duration", mt[1]["t1"] - mt[1]["t0"], 150)
check("mean keeps the fields", mt[1]["name"], "k")
try:
	mp.mean_timeline([r1, r2[:1]])
	check("different events raise", False, True)
except ValueError:
	check("different events raise", True, True)
check("one run keeps its gaps", mp.mean_timeline([r1])[1]["t0"], 100)

# Correction 2: rates per kernel under the two bounds. Buffer of 100 bytes, live in both
EVB = [{"e": 0, "kind": "alloc", "t0": 0, "t1": 1, "id": 0, "size": 100},
	{"e": 1, "kind": "kernel", "t0": 10, "t1": 20, "k": 0, "name": "ka(int)", "instrs": 10},
	{"e": 2, "kind": "kernel", "t0": 30, "t1": 40, "k": 1, "name": "kb(int)", "instrs": 10}]
TIM = {"ka": {"mean_us": 3.0}, "kb": {"mean_us": 9.0}}
RES = [{"injected": True, "kernel_name": "ka(int)", "outcome": 17}] * 2 + [{"injected": True, "kernel_name": "ka(int)", "outcome": 3}] * 2 + \
	[{"injected": True, "kernel_name": "kb(int)", "outcome": 3}] * 6 + [{"injected": False, "kernel_name": "kb(int)", "outcome": 19}]
rb = mp.mode_b_rates(RES, EVB, TIM, 2.0)
check("rates: runs", (rb["runs"], rb["not_injected"]), (10, 1))
check("rates: upper shares", rb["upper_as_measured"]["kernel_shares"], {"ka": 0.25, "kb": 0.75})
check("rates: upper sdc", rb["upper_as_measured"]["rates"]["sdc"]["rate"], 0.25 * 0.5)
lo = rb["lower_without_floor"]
check("rates: lower shares", lo["kernel_shares"], {"ka": 1 / 8, "kb": 7 / 8})
check("rates: lower sdc", lo["rates"]["sdc"]["rate"], 1 / 8 * 0.5)
check("rates: masked plus sdc", lo["rates"]["sdc"]["rate"] + lo["rates"]["masked"]["rate"], 1.0)
check("rates: nothing unsampled", lo["unsampled_share"], 0)
check("rates: proportional sampling loses nothing", abs(mp.mode_b_rates(RES[:1] + RES[4:5], EVB, {"ka": {"mean_us": 1.0}, "kb": {"mean_us": 1.0}}, 0.0)["upper_as_measured"]["effective_runs"] - 2) < 1e-9, True)
rb2 = mp.mode_b_rates(RES[:4], EVB, TIM, 2.0)
check("rates: kernel with no runs", rb2["upper_as_measured"]["unsampled_share"], 0.75)
check("rates: no guess on it", rb2["upper_as_measured"]["rates"]["sdc"]["rate"], 0.25 * 0.5)

# Multi-cell upsets
GT = [{"mode": "a", "buffer": 0, "size": 100, "offset": 0, "bit": 31, "event": 1}, {"mode": "a", "buffer": 0, "size": 100, "offset": 96, "bit": 0, "event": 1}]
ab = mp.apply_mcu(GT, EVB, mp.MCU_ADJ_BITS, 3, None, 1)
check("adj_bits window in the word", all(0 <= t["bit"] <= 29 for t in mp.apply_mcu(GT * 200, EVB, mp.MCU_ADJ_BITS, 3, None, 1)), True)
check("adj_bits reaches both ends", {min(t["bit"] for t in mp.apply_mcu(GT * 200, EVB, mp.MCU_ADJ_BITS, 3, None, 1)), max(t["bit"] for t in mp.apply_mcu(GT * 200, EVB, mp.MCU_ADJ_BITS, 3, None, 1))}, {0, 29})
check("adj_bits one word", (ab[0]["mcu_k"], ab[0]["cell_deltas"], ab[0]["cells_requested"]), (3, [0], 1))
aw = mp.apply_mcu(GT * 50, EVB, mp.MCU_ADJ_WORDS, 3, 4, 1)
check("adj_words first word stays", all(t["cell_deltas"][0] == 0 for t in aw), True)
check("adj_words keeps the bit", (aw[0]["bit"], aw[0]["mcu_k"], aw[0]["mcu_step"]), (31, 1, 4))
check("adj_words inside the buffer", all(0 <= t["offset"] + d and t["offset"] + d + 4 <= 100 for t in aw for d in t["cell_deltas"]), True)
check("adj_words asks for k", {t["cells_requested"] for t in aw}, {3})
check("adj_words clips at the start", {len(t["cell_deltas"]) for t in aw if t["offset"] == 0}, {1, 3})
check("adj_words clips at the end", {len(t["cell_deltas"]) for t in aw if t["offset"] == 96}, {1, 3})
check("adj_words row step", {abs(d) for t in mp.apply_mcu(GT * 5, EVB, mp.MCU_ADJ_WORDS, 2, 64, 1) for d in t["cell_deltas"]} <= {0, 64}, True)
for bad in ((mp.MCU_ADJ_WORDS, 2, 3), (mp.MCU_ADJ_WORDS, 9, 4), (mp.MCU_ADJ_BITS, 33, None), ("none", 2, 4)):
	try:
		mp.apply_mcu(GT, EVB, bad[0], bad[1], bad[2], 1)
		check("bad MCU parameters raise %r" % (bad,), False, True)
	except ValueError:
		check("bad MCU parameters raise %r" % (bad,), True, True)
SHK = [{"e": 1, "kind": "kernel", "t0": 0, "t1": 1, "k": 0, "name": "s", "instrs": 10, "shm": 16, "ctas": []}]
us = mp.apply_mcu([{"mode": "u", "kernel": 0, "shm_offset": 1024 + 12, "bit": 1}] * 40, SHK, mp.MCU_ADJ_WORDS, 2, 4, 1)
check("shared u stays in the window", all(1024 <= 1036 + d and 1036 + d + 4 <= 1040 for t in us for d in t["cell_deltas"]), True)
ss = mp.apply_mcu([{"mode": "s", "kernel": 0, "bit": 77}] * 40, SHK, mp.MCU_ADJ_WORDS, 2, 8, 1)
check("shared s keeps all: the tool clips", {len(t["cell_deltas"]) for t in ss}, {2})
check("shared s bit stays a seed", mp.apply_mcu([{"mode": "s", "kernel": 0, "bit": 7777}], SHK, mp.MCU_ADJ_BITS, 2, None, 1)[0]["bit"], 7777)
info2 = mp.target_to_info(dict(GT[0], mcu_k=3, cell_deltas=[0, -4]))
check("info has the size", "mcu_k 3\n" in info2, True)
check("info has the words", "cell_delta 0\ncell_delta -4\n" in info2, True)
check("info stays the same without MCU", "cell_delta" in mp.target_to_info(GT[0]), False)
cell = lambda **k: dict({"applied": 1, "reads_old": 0, "reads_new": 0, "host_reads": 0, "first_access": 0}, **k)
sc = lambda *cs: mp.masked_subclass({"cells": list(cs)})
check("cells: nothing", sc(cell(), cell()), "no_read")
check("cells: any read of the flip wins", sc(cell(first_access=2), cell(reads_new=1, first_access=1)), "read_masked")
check("cells: overwritten over no read", sc(cell(), cell(first_access=2)), "overwritten")
check("cells: stale over overwritten", sc(cell(first_access=2), cell(reads_old=1, first_access=1)), "stale_only")
check("cells: ambiguous", sc(cell(first_access=3), cell(reads_old=1, first_access=1)), "ambiguous")
check("cells: read before a write counts", sc(cell(reads_new=2, first_access=1)), "read_masked")
check("cells: word not flipped does not count", sc(cell(applied=0, reads_new=5), cell()), "no_read")
check("old records still work", mp.masked_subclass({"reads_new": 1, "host_reads": 0, "reads_old": 0, "first_access": 1}), "read_masked")

# Default MCU sizes: only for weighting afterwards
import mcu_sizes as ms
for name, t in ms.MCU_SIZE_TABLES.items():
	check("sizes sum to 1 " + name, abs(sum(t.values()) - 1) < 1e-9, True)
wr = ms.weighted_rate({1: 0.5, 2: 1.0})
check("weighted rate", abs(wr["rate"] - (0.76 * 0.5 + 0.17)) < 1e-9, True)
check("share with no rate is not assumed", abs(wr["share_without_rate"] - 0.07) < 1e-9, True)

# Records of the register injector
os.environ.setdefault("NVBITFI_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import params as pr, run_one_injection as roi
check("code to model", (pr.bfm_model(43), pr.bfm_k(43), pr.bfm_model(0), pr.bfm_k(0), pr.mcu_bfm(pr.MCU_ADJ_LANES, 4)), (4, 3, 0, 1, 64))
check("model names", (pr.em_str(53), pr.em_str(0)), ("MCU_ADJ_REGS_K3", "FLIP_SINGLE_BIT"))
check("cells from the log", roi.get_cells({"numCells": "2", "cell0": "7 2 5890 0x400 0x72fd 0x76fd", "cell1": "8 2 5890 0x400 0x1 0x401"})[1],
	{"reg_no": 8, "lane": 2, "tid": 5890, "mask": "0x400", "before": "0x1", "after": "0x401"})
base = {"addrPart": "ra", "memAccess": "store", "addrValid": "1", "addrBit": "4", "addrRestored": "1", "addrOverwritten": "0"}
check("store in valid memory is apart", (roi.addr_record(base)["addr_class"], roi.addr_record(base)["store_wrong_location"]), ("store_wrong_location", True))
check("store outside", roi.addr_record(dict(base, addrValid="0"))["addr_class"], "store_invalid")
check("atomic in valid memory", roi.addr_record(dict(base, memAccess="atomic"))["addr_class"], "atomic_wrong_location")
check("load in valid memory is not a wrong store", (roi.addr_record(dict(base, memAccess="load"))["addr_class"], roi.addr_record(dict(base, memAccess="load"))["store_wrong_location"]), ("load_valid", False))
check("descriptor moves no address", (roi.addr_record(dict(base, addrPart="desc", addrValid="na"))["addr_class"], roi.addr_record(dict(base, addrPart="desc", addrValid="na"))["store_wrong_location"]), ("desc_store", False))
check("not the address model", roi.addr_record({}), {})

print("%d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
