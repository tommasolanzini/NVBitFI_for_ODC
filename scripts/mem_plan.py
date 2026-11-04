# Planning and analysis for the memory-data injector, no GPU needed: timeline
# of a run, choice of the targets (buffer, word, bit, time), Masked subclasses.
# Offline tests: tests/test_mem_plan.py

import json, random

GDDR_BITS = 8 * 1024**3 * 8 # 8 GiB of GDDR7 on this GPU (coverage inputs go in one table later)

MODE_BETWEEN = "a" # between kernels: host read-XOR-write
MODE_DURING = "b"  # during a kernel: atomicXor at the N-th dynamic instruction

# First access after the flip, as in mem_injector/mem_injector.h
ACCESS_READ = 1
ACCESS_WRITE = 2


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


# Contents of nvbitfi-injection-info.txt
def target_to_info(t):
	return "".join("%s %s\n" % (k, t[k]) for k in
		("mode", "buffer", "size", "offset", "bit", "event", "kernel", "kernel_name", "inst") if k in t)


# Why a Masked run was masked, from the trace in the injection record. Only
# "read_masked" says that the workload tolerates the corruption.
def masked_subclass(rec):
	new_seen = rec["reads_new"] + rec["host_reads"]
	if rec["first_access"] == ACCESS_WRITE:
		return "overwritten"
	if new_seen > 0:
		return "read_masked"
	if rec["reads_old"] > 0:
		return "stale_only"
	return "no_read"


# Kernels whose bit is set in the readKernels mask
def kernels_from_mask(mask):
	return [k for k in range(64) if mask >> k & 1]
