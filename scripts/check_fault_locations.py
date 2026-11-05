#!/usr/bin/env python3
# Does the position of the error match the injection record? No GPU needed: compares
# the outputs of each run with the golden ones, for the new fault models.
#
#   check_fault_locations.py <directory with reg-* and mem-* folders>
#
# Writes location-check.json in that directory and prints one line per folder.
# Expected positions come from the workload source, see expected_* below. A run in
# "exceptions" is a run to look at by hand, not necessarily a bug.

import glob, json, os, struct, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N_GEMM = 128
TILE = 16
N_PTR = 64 # 64 CTAs x 64 threads, element = list = tid
LIST_LEN = 64
NODE_BYTES = 16
N_SHM = 128
CNN_BATCH = 32 # logits: row = image, column = class
CNN_C1, CNN_C2, CNN_W1, CNN_P1, CNN_W2, CNN_FLAT = 4, 8, 30, 15, 13, 288

OUTPUTS = {"gemm": ["output_core.bin", "output_tc.bin"], "ptr_chase": ["output_ptr.bin"], "shm_stencil": ["output_shm.bin"], "cnn": ["output_logits.bin"]}
SFU_FUNCS = ["rsqrt", "rcp", "sin", "exp", "log"] # in launch order
OUTPUTS["sfu_math"] = ["output_%s.bin" % f for f in SFU_FUNCS]
GEMM_BUFFERS = {0: "A", 1: "B", 2: "C", 3: "C_tc", 4: "A_h", 5: "B_h"}
SDC = (16, 17, 18)


def words(path):
	data = open(path, "rb").read()
	return struct.unpack("<%dI" % (len(data) // 4), data[:len(data) // 4 * 4])


# {output file: set of flat indices that differ from the golden}, None if it is missing
def wrong_elements(app, run_dir):
	out = {}
	for name in OUTPUTS[app]:
		path = os.path.join(run_dir, name)
		if not os.path.isfile(path):
			return None
		a, g = words(path), words(os.path.join(ROOT, "test-apps", app, "golden_" + name))
		if len(a) != len(g):
			return None
		out[name] = {i for i, (x, y) in enumerate(zip(a, g)) if x != y}
	return out


def tile_of_cta(cta):
	n = N_GEMM // TILE
	col, row = cta % n, cta // n
	return {(row * TILE + r) * N_GEMM + col * TILE + c for r in range(TILE) for c in range(TILE)}


def gemm_row(r):
	return {r * N_GEMM + c for c in range(N_GEMM)}


def gemm_col(c):
	return {r * N_GEMM + c for r in range(N_GEMM)}


def cnn_row(img):
	return {img * CNN_BATCH + c for c in range(CNN_BATCH)}


def cnn_col(cls):
	return {r * CNN_BATCH + cls for r in range(CNN_BATCH)}


# Function of a sfu_math kernel: k_rsqrt(floatconst*,float*,int) -> rsqrt
def sfu_func(kernel):
	return kernel.split("(")[0][len("k_"):]


# This kernel and the later ones read the same input and each has its own output
def sfu_from(kernel):
	return SFU_FUNCS[SFU_FUNCS.index(sfu_func(kernel)):]


def warp_of_tid(tid):
	return {tid - tid % 32 + i for i in range(32)}


# Elements a corruption inside the threads of a kernel can change
def expected_threads(app, rec):
	kernel = rec.get("kernel", "")
	tids = {c["tid"] for c in rec.get("cells", [])} or {rec.get("tid")}
	wide = rec.get("reg_type") in ("UR", "UP")
	if app == "shm_stencil":
		return {"output_shm.bin": {rec["cta"] * N_SHM + c for c in range(N_SHM)}} # the window of a CTA is one row
	if app == "cnn":
		return {"output_logits.bin": cnn_row(rec["cta"] % CNN_BATCH)} # first index of the CTA = image, in every kernel
	if app == "sfu_math":
		s = set().union(*[warp_of_tid(t) for t in tids]) if wide else tids
		return {"output_%s.bin" % f: (s if f == sfu_func(kernel) else set()) for f in SFU_FUNCS}
	if app == "gemm":
		if kernel.startswith("gemm_tc"):
			return {"output_core.bin": set(), "output_tc.bin": tile_of_cta(rec["cta"])}
		s = set().union(*[warp_of_tid(t) for t in tids]) if wide else tids
		return {"output_core.bin": s, "output_tc.bin": set()}
	s = set().union(*[warp_of_tid(t) for t in tids]) if wide else tids
	return {"output_ptr.bin": s}


# An address model store that lands in valid memory: it also changes what reads the
# landing word. None = the effect cannot be told from the source (nodes of a list).
def landing_effect(app, rec):
	b, off = rec.get("addr_new_buffer", -1), rec.get("addr_new_offset", 0)
	if app == "shm_stencil":
		return {} # another word of the same window: same row as the CTA
	if app == "cnn":
		return mem_word_effect(app, b, off)
	if app == "sfu_math":
		# buffer 0 = in, read by this kernel and the later ones; 1 = out, cleared before each launch
		if b == 0: return {"output_%s.bin" % f: {off // 4} for f in sfu_from(rec.get("kernel", ""))}
		if b == 1: return {"output_%s.bin" % sfu_func(rec.get("kernel", "")): {off // 4}}
		return {}
	if app == "gemm":
		i = off // 4
		if b == 2: return {"output_core.bin": {i}}
		if b == 3: return {"output_tc.bin": {i}}
		if b == 0: return {"output_core.bin": gemm_row(i // N_GEMM)}
		if b == 1: return {"output_core.bin": gemm_col(i % N_GEMM)}
		if b == 4: return {"output_tc.bin": gemm_row((off // 2) // N_GEMM)}
		if b == 5: return {"output_tc.bin": gemm_col((off // 2) % N_GEMM)}
	else:
		if b in (0, 1, 4): return {"output_ptr.bin": {off // 4}}
		if b == 2: return {"output_ptr.bin": {(off // NODE_BYTES) // LIST_LEN}}
		if b == 3: return {"output_ptr.bin": {off // 8}}
	return {}


def mem_word_effect(app, buffer, off):
	if app == "gemm":
		i = off // 4
		if buffer == 0: return {"output_core.bin": gemm_row(i // N_GEMM)}
		if buffer == 1: return {"output_core.bin": gemm_col(i % N_GEMM)}
		if buffer == 2: return {"output_core.bin": {i}}
		if buffer == 3: return {"output_tc.bin": {i}}
		# half: a 32-bit word holds two elements
		e0, e1 = off // 2, off // 2 + 1
		if buffer == 4: return {"output_tc.bin": gemm_row(e0 // N_GEMM) | gemm_row(e1 // N_GEMM)}
		if buffer == 5: return {"output_tc.bin": gemm_col(e0 % N_GEMM) | gemm_col(e1 % N_GEMM)}
	if app == "cnn":
		i = off // 4
		if buffer in (1, 2): return {"output_logits.bin": set(range(CNN_BATCH ** 2))} # filters: all images
		if buffer == 3: return {"output_logits.bin": cnn_col(i // CNN_FLAT)}
		if buffer == 4: return {"output_logits.bin": cnn_col(i)}
		if buffer == 9: return {"output_logits.bin": {i}}
		per_image = {0: 32 * 32, 5: CNN_C1 * CNN_W1 ** 2, 6: CNN_C1 * CNN_P1 ** 2, 7: CNN_C2 * CNN_W2 ** 2, 8: CNN_FLAT}
		if buffer in per_image: return {"output_logits.bin": cnn_row(i // per_image[buffer])}
	if app == "ptr_chase":
		if buffer in (0, 1, 4): return {"output_ptr.bin": {off // 4}} # scale, adj, out: one element per list
		if buffer == 2: return {"output_ptr.bin": {(off // NODE_BYTES) // LIST_LEN}} # the walker of that list
		if buffer == 3: return {"output_ptr.bin": {off // 8}} # heads
	return {}


def merge(a, b):
	out = {k: set(v) for k, v in a.items()}
	for k, v in b.items():
		out.setdefault(k, set()).update(v)
	return out


def inside(wrong, allowed):
	return all(w <= allowed.get(k, set()) for k, w in wrong.items())


def run_dir_of_reg(folder, app, r):
	return "%s/logs/%s/%s-group%d-model%d-icount%d" % (folder, app, app, r["igid"], r["bfm"], r["icount"])


def check_reg(folder):
	res = {"runs": 0, "outcomes": {}, "sdc": 0, "ok": 0, "no_output": 0, "exceptions": [], "wrong_location": {"runs": 0, "original_changed": 0, "landing_seen": 0}, "cells_short": 0, "by_part": {}, "by_class": {}}
	for path in sorted(glob.glob(folder + "/logs/*/results-*.jsonl")):
		app = os.path.basename(os.path.dirname(path))
		for line in open(path):
			r = json.loads(line)
			if r.get("suspended_s", 0) > 0.5: # the machine slept during this run: not a result
				res["suspended"] = res.get("suspended", 0) + 1
				continue
			res["runs"] += 1
			cat = out_class(r["outcome"])
			res["outcomes"][cat] = res["outcomes"].get(cat, 0) + 1
			if r.get("requested_cells", 0) > len(r.get("cells", [])):
				res["cells_short"] += 1
			if "addr_part" in r:
				for key, val in (("by_part", r["addr_part"]), ("by_class", r["addr_class"])):
					d = res[key].setdefault(val, {})
					d[cat] = d.get(cat, 0) + 1
			if r["outcome"] not in SDC:
				continue
			res["sdc"] += 1
			wrong = wrong_elements(app, run_dir_of_reg(folder, app, r))
			if wrong is None:
				res["no_output"] += 1
				continue
			allowed = expected_threads(app, r)
			if r.get("store_wrong_location"):
				w = res["wrong_location"]
				w["runs"] += 1
				landing = landing_effect(app, r)
				if any(v & allowed.get(k, set()) for k, v in wrong.items()):
					w["original_changed"] += 1
				if any(v & landing.get(k, set()) for k, v in wrong.items()):
					w["landing_seen"] += 1
				allowed = merge(allowed, landing)
			if inside(wrong, allowed):
				res["ok"] += 1
			else:
				res["exceptions"].append({"icount": r["icount"], "kernel": r.get("kernel", "")[:12], "opcode": r.get("opcode"), "pc": r.get("pc"), "tid": r.get("tid"),
					"reg": "%s%s" % (r.get("reg_type"), r.get("reg_no")), "cells": [(c["reg_no"], c["lane"], c["tid"]) for c in r.get("cells", [])],
					"class": r.get("addr_class"), "landing": r.get("addr_new_buffer"), "wrong": {k: sorted(v)[:6] for k, v in wrong.items()}, "wrong_count": {k: len(v) for k, v in wrong.items()}})
	return res


def out_class(cat):
	if cat in SDC:
		return "sdc"
	if cat in (1, 2, 3):
		return "masked"
	if cat == 20:
		return "crash"
	if cat == 4:
		return "timeout"
	return "other:%d" % cat


def check_mem(folder, app):
	res = {"runs": 0, "outcomes": {}, "subclass": {}, "sdc": 0, "ok": 0, "no_output": 0, "exceptions": [], "not_injected": 0, "same_word_in_dummy": 0, "words_requested": 0, "words_flipped": 0, "new_reads": 0}
	for path in sorted(glob.glob(folder + "/results-mode*.jsonl")):
		mode = os.path.basename(path)[len("results-mode")]
		for line in open(path):
			r = json.loads(line)
			if r.get("suspended_s", 0) > 0.5: # the machine slept during this run: not a result
				res["suspended"] = res.get("suspended", 0) + 1
				continue
			res["runs"] += 1
			cat = out_class(r["outcome"])
			res["outcomes"][cat] = res["outcomes"].get(cat, 0) + 1
			if r.get("masked_subclass"):
				res["subclass"][r["masked_subclass"]] = res["subclass"].get(r["masked_subclass"], 0) + 1
			if not r.get("injected"):
				res["not_injected"] += 1
			cells = r.get("cells", [])
			res["words_requested"] += r.get("cells_requested", 1)
			res["words_flipped"] += sum(c["applied"] for c in cells)
			res["same_word_in_dummy"] += all(c["old_word"] == c["new_word"] for c in cells) and bool(cells)
			res["new_reads"] += sum(c["reads_new"] for c in cells)
			if r["outcome"] not in SDC:
				continue
			res["sdc"] += 1
			wrong = wrong_elements(app, "%s/runs-mode%s/icount%d" % (folder, mode, r["icount"]))
			if wrong is None:
				res["no_output"] += 1
				continue
			if app == "shm_stencil":
				cta = r["cta"]
				allowed = {"output_shm.bin": {cta * N_SHM + c for c in range(N_SHM)}}
			else:
				allowed = {}
				for c in cells:
					allowed = merge(allowed, mem_word_effect(app, r["buffer"], r["offset"] + c["delta"]))
			if inside(wrong, allowed):
				res["ok"] += 1
			else:
				res["exceptions"].append({"mode": mode, "icount": r["icount"], "buffer": r.get("buffer"), "offset": r.get("offset"), "deltas": [c["delta"] for c in cells], "mask": r.get("mask", ""), "cta": r.get("cta"),
					"wrong": {k: sorted(v)[:6] for k, v in wrong.items()}, "wrong_count": {k: len(v) for k, v in wrong.items()}})
	return res


def main():
	base = os.path.abspath(sys.argv[1])
	out = {}
	for folder in sorted(glob.glob(base + "/reg-*")):
		out[os.path.basename(folder)] = check_reg(folder)
	for folder in sorted(glob.glob(base + "/mem-*")):
		name = os.path.basename(folder)
		app = name.split("-")[1]
		out[name] = check_mem(folder, app)
	json.dump(out, open(base + "/location-check.json", "w"), indent=1)
	for name, r in out.items():
		print("%-44s runs=%3d %s sdc=%d ok=%d no_output=%d exceptions=%d%s" % (name, r["runs"], r["outcomes"], r["sdc"], r["ok"], r["no_output"], len(r["exceptions"]),
			(" subclass=%s not_injected=%d words=%d/%d new_reads=%d same=%d" % (r["subclass"], r["not_injected"], r["words_flipped"], r["words_requested"], r["new_reads"], r["same_word_in_dummy"])) if "subclass" in r else (" cells_short=%d" % r["cells_short"])))


if __name__ == "__main__":
	main()
