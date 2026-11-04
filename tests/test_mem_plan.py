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
check("exposure adds up", abs(x["windows"] + x["reserved"] + x["free"] + x["not_usable"] - x["total"]) < 1e-9, True)
check("not usable", abs(x["not_usable"] - (128 * 1024 - 102400) * 8 * 20 * 1e-5) < 1e-9, True)

u = mp.sample_uniform(SH, 20000, 5)
check("u only kernel with shm", {t["kernel"] for t in u}, {0})
check("u words after the reserved bytes", all(1024 <= t["shm_offset"] < 1024 + 400 and t["shm_offset"] % 4 == 0 for t in u), True)
check("u bits", all(0 <= t["bit"] < 32 for t in u), True)
by_sm = collections.Counter(t["sm"] for t in u)
check("u sm follows the residence", abs(by_sm[3] / len(u) - 1250 / 1750) < 0.02, True)
check("u time inside a residence", all((t["sm"] == 7 and 50 <= t["inst"] < 100) or (t["sm"] == 3 and 0 <= t["inst"] < 100) for t in u), True)
info = mp.target_to_info(u[0])
check("u info has sm and offset", "sm %d\n" % u[0]["sm"] in info and "shm_offset %d\n" % u[0]["shm_offset"] in info, True)
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

print("%d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
