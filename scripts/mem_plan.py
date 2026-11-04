# Planning and analysis for the memory-data injector, no GPU needed: timeline
# of a run, choice of the targets (buffer or SM, word, bit, time), Masked subclasses.
# Offline tests: tests/test_mem_plan.py

import json, random

GDDR_BITS = 8 * 1024**3 * 8 # 8 GiB of GDDR7 on this GPU (coverage inputs go in one table later)

MODE_BETWEEN = "a" # between kernels: host read-XOR-write
MODE_DURING = "b"  # during a kernel: atomicXor at the N-th dynamic instruction
MODE_AFTER_STORE = "s" # shared memory: right after the N-th dynamic store
MODE_UNIFORM = "u"     # shared memory: uniform over SM x offset x time

# Shared memory of this GPU (coverage inputs go in one table later)
NUM_SMS = 20
SHM_PHYSICAL = 128 * 1024 # SRAM per SM, L1 data + shared, from the chip specs: software cannot measure it
SHM_USABLE = 102400       # cudaDeviceProp::sharedMemPerMultiprocessor, tests/shm_probe
SHM_RESERVED = 1024       # cudaDevAttrReservedSharedMemoryPerBlock: first bytes of the window of every CTA

# First access after the flip, as in mem_injector/mem_injector.h
ACCESS_READ = 1
ACCESS_WRITE = 2
ACCESS_AMBIGUOUS = 3 # shared memory: a store around the flip, see mem_injector.h


def load_events(path):
	return [json.loads(line) for line in open(path)]


# timing gives clean times, count gives the dynamic instructions: same events
def merge_profiles(timing, count):
	if [e["kind"] for e in timing] != [e["kind"] for e in count]:
		raise ValueError("timing and count runs have different events")
	merged = []
	for t, c in zip(timing, count):
		m = dict(t)
		if t["kind"] == "kernel":
			if t["name"] != c["name"]:
				raise ValueError("kernel %s != %s" % (t["name"], c["name"]))
			m["instrs"] = c["instrs"]
			for k in ("shm", "stores", "ctas"):
				if k in c:
					m[k] = c[k]
		merged.append(m)
	return merged


def buffers_of(events):
	b = {}
	for e in events:
		if e["kind"] == "alloc":
			b[e["id"]] = {"id": e["id"], "size": e["size"], "alloc_event": e["e"], "free_event": None}
		elif e["kind"] == "free" and "id" in e:
			b[e["id"]]["free_event"] = e["e"]
	return b


# The run as time slots. A flip in a gap happens on entry of the next event; the
# time inside a copy, alloc or free belongs to the gap after it. A kernel is
# its own slot. The time after the last event is returned apart: nothing can
# read a flip there.
def slots_of(events):
	slots = []
	boundary = None
	tail = 0
	for e in events:
		if e["kind"] == "term":
			tail = max(e["t0"] - boundary, 0) if boundary is not None else 0
			break
		if boundary is not None:
			slots.append({"mode": MODE_BETWEEN, "event": e["e"], "duration": max(e["t0"] - boundary, 0)})
		if e["kind"] == "kernel":
			slots.append({"mode": MODE_DURING, "event": e["e"], "kernel": e["k"], "name": e["name"],
				"instrs": e.get("instrs", 0), "duration": max(e["t1"] - e["t0"], 0)})
			boundary = e["t1"]
		else:
			boundary = e["t0"]
	return slots, tail


def is_live(buf, slot):
	if buf["alloc_event"] >= slot["event"]:
		return False
	if buf["free_event"] is None:
		return True
	if slot["mode"] == MODE_BETWEEN:
		return buf["free_event"] >= slot["event"]
	return buf["free_event"] > slot["event"]


# bits x time, in bit-seconds: per mode, after the last event, and for the bytes
# never allocated. The last two are Masked by definition: no injection is spent
# on them, they enter the rates analytically.
def exposure(events):
	buffers = buffers_of(events)
	slots, tail = slots_of(events)
	out = {MODE_BETWEEN: 0.0, MODE_DURING: 0.0}
	for s in slots:
		for b in buffers.values():
			if is_live(b, s):
				out[s["mode"]] += b["size"] * 8 * s["duration"] * 1e-9
	tail_bs = sum(b["size"] * 8 for b in buffers.values() if b["free_event"] is None) * tail * 1e-9
	duration = sum(s["duration"] for s in slots) + tail
	total_gddr = GDDR_BITS * duration * 1e-9
	return {"between_kernels": out[MODE_BETWEEN], "during_kernels": out[MODE_DURING], "after_last_event": tail_bs,
		"never_allocated": max(total_gddr - out[MODE_BETWEEN] - out[MODE_DURING] - tail_bs, 0.0),
		"gddr_total": total_gddr, "run_seconds": duration * 1e-9}


# Targets weighted by bits x lifetime, inside one timing mode. The two modes are
# sampled apart so that they can be compared; exposure() gives their weights.
def sample_targets(events, mode, n, seed):
	rng = random.Random(seed)
	buffers = buffers_of(events)
	slots, _ = slots_of(events)
	pairs, weights = [], []
	for s in slots:
		if s["mode"] != mode or s["duration"] <= 0:
			continue
		if mode == MODE_DURING and s["instrs"] <= 0:
			continue
		for b in buffers.values():
			if is_live(b, s) and b["size"] >= 4:
				pairs.append((b, s))
				weights.append(b["size"] * s["duration"])
	if not pairs:
		raise ValueError("nothing to inject in mode " + mode)
	targets = []
	for b, s in rng.choices(pairs, weights=weights, k=n):
		t = {"mode": mode, "buffer": b["id"], "size": b["size"], "offset": rng.randrange(b["size"] // 4) * 4,
			"bit": rng.randrange(32), "event": s["event"]}
		if mode == MODE_DURING:
			t.update({"kernel": s["kernel"], "kernel_name": s["name"], "inst": rng.randrange(s["instrs"])})
		targets.append(t)
	return targets


def kernels_with_shm(events):
	return [e for e in events if e["kind"] == "kernel" and e.get("shm", 0) > 0 and e.get("instrs", 0) > 0]


# Time a CTA is resident, in ns: its share of the instruction clock of the kernel
def cta_seconds_ns(kernel, cta):
	return max(cta[2] - cta[1], 0) / kernel["instrs"] * max(kernel["t1"] - kernel["t0"], 0)


def peak_ctas_per_sm(kernel):
	steps = {}
	for sm, first, last in kernel["ctas"]:
		steps.setdefault(sm, []).extend([(first, 1), (last + 1, -1)])
	peak = {}
	for sm, s in steps.items():
		now = 0
		for _, d in sorted(s, key=lambda x: (x[0], x[1])):
			now += d
			peak[sm] = max(peak.get(sm, 0), now)
	return peak


# bits x time of the SRAM of all the SMs (one structure of SHM_PHYSICAL bytes per
# SM), in bit-seconds:
#   windows:  application part of the windows of resident CTAs, reachable and injected
#   reserved: reserved part of those windows, kept apart
#   l1_data:  all the rest, L1 data: not reachable by software, so it is neither
#             injected nor Masked. Its AVF has to come from the beam. Its two parts
#             are never_shared (never usable as shared memory) and free_shared
#             (usable as shared memory, no CTA there).
def shared_exposure(events):
	t0 = min((e["t0"] for e in events), default=0)
	t_end = max((e["t1"] for e in events), default=0)
	duration = max(t_end - t0, 0)
	windows = reserved = 0.0
	peak = 0
	for k in kernels_with_shm(events):
		for c in k["ctas"]:
			ns = cta_seconds_ns(k, c)
			windows += k["shm"] * 8 * ns * 1e-9
			reserved += SHM_RESERVED * 8 * ns * 1e-9
		p = peak_ctas_per_sm(k)
		if p:
			peak = max(peak, max(p.values()))
	total = SHM_PHYSICAL * 8 * NUM_SMS * duration * 1e-9
	never_shared = (SHM_PHYSICAL - SHM_USABLE) * 8 * NUM_SMS * duration * 1e-9
	free_shared = max(total - windows - reserved - never_shared, 0.0)
	return {"total": total, "windows": windows, "reserved": reserved, "l1_data": never_shared + free_shared,
		"never_shared": never_shared, "free_shared": free_shared, "run_seconds": duration * 1e-9,
		"peak_ctas_on_one_sm": peak}


L1_NOT_REACHABLE = "not reachable, AVF to be estimated with the beam"
OUTCOME_SDC = (16, 17, 18)
OUTCOME_CRASH = 20


# AVF of shared memory from the runs of mode u, in two versions: on the injected
# part only, and on the whole SRAM of the SM. The second has no single number: it
# is the known part plus the shares whose AVF is not measured here. Runs not
# injected are left out and counted.
def shared_avf(results, exposure):
	runs = [r for r in results if r.get("injected")]
	n = len(runs)
	sdc = sum(r["outcome"] in OUTCOME_SDC for r in runs)
	crash = sum(r["outcome"] == OUTCOME_CRASH for r in runs)
	avf = (sdc + crash) / n if n else 0.0
	total = exposure["total"]
	return {"runs": n, "not_injected": len(results) - n, "sdc": sdc, "crash": crash,
		"avf_injected_part": avf,
		"whole_sram": {
			"windows_share": exposure["windows"] / total,
			"known_part": avf * exposure["windows"] / total,
			"reserved_share": exposure["reserved"] / total,
			"reserved_avf": "not measured, kept apart",
			"l1_data_share": exposure["l1_data"] / total,
			"l1_data_avf": L1_NOT_REACHABLE,
		}}


# Mode u: every bit of every resident window at every time has the same weight.
# Sampling a CTA by bytes x residence, then a time inside its residence, is the
# same as sampling SM x address x time restricted to the resident bytes. Only the
# SM and the time are kept: at run time the tool picks among the CTAs resident
# then (pick), because scheduling can place them differently.
def sample_uniform(events, n, seed):
	rng = random.Random(seed)
	pairs, weights = [], []
	for k in kernels_with_shm(events):
		for c in k["ctas"]:
			w = k["shm"] * cta_seconds_ns(k, c)
			if w > 0:
				pairs.append((k, c))
				weights.append(w)
	if not pairs:
		raise ValueError("no resident shared memory to inject")
	targets = []
	for k, (sm, first, last) in rng.choices(pairs, weights=weights, k=n):
		targets.append({"mode": MODE_UNIFORM, "kernel": k["k"], "kernel_name": k["name"], "event": k["e"],
			"inst": rng.randrange(first - 1, last - 1), "sm": sm,
			"shm_offset": SHM_RESERVED + rng.randrange(k["shm"] // 4) * 4, "bit": rng.randrange(32),
			"pick": rng.randrange(1 << 16)})
	return targets


# Mode s: every dynamic store to shared memory has the same weight. Not a model
# of upsets, it always hits live data: not for AVF or coverage.
def sample_after_store(events, n, seed):
	rng = random.Random(seed)
	kernels = [k for k in kernels_with_shm(events) if k.get("stores", 0) > 0]
	if not kernels:
		raise ValueError("no stores to shared memory")
	targets = []
	for k in rng.choices(kernels, weights=[k["stores"] for k in kernels], k=n):
		targets.append({"mode": MODE_AFTER_STORE, "kernel": k["k"], "kernel_name": k["name"], "event": k["e"],
			"inst": rng.randrange(k["stores"]), "bit": rng.randrange(1 << 16)})
	return targets


# Contents of nvbitfi-injection-info.txt
def target_to_info(t):
	return "".join("%s %s\n" % (k, t[k]) for k in
		("mode", "buffer", "size", "offset", "bit", "event", "kernel", "kernel_name", "inst", "sm", "shm_offset", "pick") if k in t)


# Why a Masked run was masked, from the trace in the injection record. Only
# "read_masked" says that the workload tolerates the corruption.
def masked_subclass(rec):
	new_seen = rec["reads_new"] + rec["host_reads"]
	if rec["first_access"] == ACCESS_WRITE:
		return "overwritten"
	if rec["first_access"] == ACCESS_AMBIGUOUS:
		return "ambiguous"
	if new_seen > 0:
		return "read_masked"
	if rec["reads_old"] > 0:
		return "stale_only"
	return "no_read"


# Kernels whose bit is set in the readKernels mask
def kernels_from_mask(mask):
	return [k for k in range(64) if mask >> k & 1]
