#!/usr/bin/env python3
# Dataset for the network: one row per run of the final campaigns, all injectors. No GPU.
#
#   export_dataset.py [output dir]     default logs/final-2026-11-05/dataset
#
# Writes runs.csv and schema.json (one entry per column; SCHEMA.md is written from it).
# Sources: logs/final-v1-2026-11-03-gemm (gemm and sfu_math, single bit) and the real runs
# (dummy0) of logs/final-2026-11-05, blocks 1-6 (block of a cell from run_final_block.py).
# Workload size and per-kernel GPU features (no tool) come from logs/scaling-2026-11-10:
# ncu/ncu.json (scripts/run_ncu.py) and scaling.json, else from the memory profiles.
# Gate runs (DUMMY, no flip) and pilots are not included. Rows are never dropped: runs not
# valid for the rates carry valid_for_rates = 0 and the reason.

import csv, glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_fault_locations as cfl  # noqa: E402
import liveness_report as lr  # noqa: E402
import run_final_block as rfb  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOGS = ROOT + "/logs"
FINAL = LOGS + "/final-2026-11-05"
V1 = LOGS + "/final-v1-2026-11-03-gemm"
LIVE = LOGS + "/liveness-2026-11-07"
SCALING = LOGS + "/scaling-2026-11-10"
LIVE_LARGE = SCALING + "/liveness"
NCU = SCALING + "/ncu/ncu.json"
RF_REGS_PER_SM = 65536
L2_BYTES = 32 * 1024 * 1024

# workload -> (base workload, size parameter, size)
SIZES = {"gemm": ("gemm", "N", 128), "gemm_n1024": ("gemm", "N", 1024), "gemm_n2048": ("gemm", "N", 2048),
	"cnn": ("cnn", "batch", 32), "cnn_large": ("cnn", "batch", 256), "ptr_chase": ("ptr_chase", "lists", 4096),
	"shm_stencil": ("shm_stencil", "N", 128), "sfu_math": ("sfu_math", "N", 128)}

MEM_MODELS = {"adj_bits": "adj_bits", "adj_words": "adj_words"}
REG_MODELS = {"FLIP_SINGLE_BIT": "single_bit", "MCU_ADJ_BITS": "adj_bits", "MCU_ADJ_REGS": "adj_regs", "MCU_ADJ_LANES": "adj_lanes"}
SUSPENDED_S = 0.5

# Location exceptions checked one by one (ANOMALIES-2026-11-07.md, SUMMARY-block5.md): proven
# moved stores, except these three, consistent but with the landing outside the known buffers
INFERRED = {("reg-sfu_math-igid12-bfm63", 24), ("reg-ptr_chase-igid12-bfm54", 97), ("reg-cnn-igid12-bfm42", 22)}
# Block 6 (SUMMARY-block6.md): both exceptions proven moved stores
PROVEN = {("reg-cnn_large-igid12-bfm0", 289), ("reg-gemm_n1024-igid12-bfm0", 15)}


def proven_population(block, igid, bfm):
	if block == 2 and igid == 12 and bfm >= 40:  # MCU on GPR
		return True
	if block == 3 and igid in (9, 12):  # single bit GPR, MCU uniform, uniform gemm re-run
		return True
	if block == 5 and igid == 9:
		return True
	return False


# ---- columns: name -> description, in order
COLUMNS = [
	("run_id", "campaign/cell/icount, unique"),
	("campaign", "final-v1 (2026-11-03, gemm and sfu_math, code 937a7f8) or final-block1..6 (logs/final-2026-11-05)"),
	("cell", "folder of the cell: injector, workload, group or mode, fault model"),
	("icount", "run number inside the cell"),
	("seed", "seed of the injection list (memory cells); empty for registers (op_seed/bit_seed instead)"),
	("app", "workload binary: gemm, cnn, ptr_chase, shm_stencil, sfu_math, and the large variants of block 6: gemm_n1024, gemm_n2048, cnn_large"),
	("workload", "base workload (same code, different size): gemm, cnn, ptr_chase, shm_stencil, sfu_math"),
	("size_param", "what size counts: N (gemm: matrix side; shm_stencil, sfu_math: elements per row/array), batch (cnn: images), lists (ptr_chase)"),
	("size", "value of size_param"),
	("working_set_bytes", "peak live global memory of the run, bytes (allocation profile, no tool)"),
	("working_set_l2_share", "working_set_bytes / 32 MB L2, capped at 1"),
	("kernel_gpu_us", "GPU duration of the kernel of the row, no tool (ncu gpu__time_duration, mean of 5 runs); empty for memory mode a (flip between kernels, no kernel) and the other kernel_* too"),
	("kernel_warps_per_sm", "active warps per SM of that kernel, no tool (ncu sm__warps_active.avg.per_cycle_active)"),
	("kernel_rf_share", "allocated share of the register file: active warps per SM x 32 x registers allocated per thread (ncu launch__registers_per_thread, hardware granularity, max(16, REGCOUNT)) / 65536"),
	("kernel_rf_live_share", "live share of the register file: live GPRs per thread (liveness tool: registers used x live fraction) x 32 x active warps per SM / 65536; independent of the allocation; same value as rf_allocation.md, coverage_inputs and SCALING.md"),
	("kernel_l1_hit_pct", "L1 hit rate of that kernel (ncu l1tex__t_sector_hit_rate)"),
	("kernel_l2_hit_pct", "L2 hit rate of that kernel (ncu lts__t_sector_hit_rate; can exceed 100, counter quirk)"),
	("kernel", "kernel of the target, short name; memory mode a: the kernel before which the word is flipped is not a target, see flip_kernel"),
	("kernel_index", "launch index of that kernel in the run (registers: kcount; memory: flip_kernel)"),
	("injector", "reg (NVBitFI register injector) or mem (mem_injector, data where it is stored)"),
	("group", "register injector group: fp32, ld, pr, tc, sfu, uniform, gp, addr, addr_store; mem: global or shared"),
	("igid", "numeric group id (common/arch.h); empty for mem"),
	("reg_class", "GPR, PR (predicate register), UR (uniform register), UP (uniform predicate); empty for mem"),
	("reg_no", "register number of the first cell"),
	("mem_mode", "a: between kernels (host); b: during a kernel (atomicXor in L2); u: shared, uniform over SM x address x time; s: shared, right after a store"),
	("for_avf", "1 if the row can enter an AVF or rate of its structure; 0 for mode s (always live data: dataset only)"),
	("fault_model", "single_bit, adj_bits (k adjacent bits of one word/register), adj_regs (same bit, k adjacent registers), adj_lanes (same register, k adjacent lanes), adj_words (one bit in k words, mcu_step bytes apart)"),
	("k", "cells of the fault model (1 for single bit)"),
	("mcu_step", "adj_words: bytes between the words (4, 8, 512)"),
	("cells_requested", "cells requested"),
	("cells_injected", "cells really written (registers: R1 and the end of used registers can cut them; memory: words applied)"),
	("cells", "JSON list of the cells with their before/after values"),
	("mask", "XOR mask of the first cell"),
	("opcode", "SASS opcode of the target instruction (registers; memory mode b/u/s: instruction of the flipping thread)"),
	("pc", "PC offset of that instruction in the kernel"),
	("inst_id", "dynamic instance drawn (registers) or dynamic instruction of the flip (memory dyn_instr)"),
	("group_instrs", "injectable instances of the group counted in that run; -1 if the run crashed"),
	("tid", "flat thread index of the target"),
	("cta", "CTA index"),
	("sm", "SM id"),
	("warp_slot", "warp slot on the SM"),
	("lane", "lane"),
	("before", "register value before (first cell)"),
	("after", "register value after (first cell)"),
	("buffer", "memory: buffer id in allocation order (-1 for shared)"),
	("buffer_size", "memory: buffer bytes"),
	("offset", "memory: byte offset of the word (shared: in the window)"),
	("bit", "memory: bit flipped in the first word"),
	("old_word", "memory: word before"),
	("new_word", "memory: word after"),
	("residents", "shared mode u: resident CTAs on the SM at the choice"),
	("addr_part", "groups addr/addr_store: ra (register of the address) or desc (memory descriptor UR)"),
	("addr_space", "global or shared"),
	("addr_class", "load_valid, load_invalid, store_wrong_location, store_invalid, ..."),
	("addr_valid", "1 if the flipped address is inside a known buffer"),
	("addr_bit", "bit of the 64-bit address that was flipped"),
	("store_wrong_location", "1 if a store went to another valid place"),
	("addr_new_buffer", "buffer of the flipped address (-1: none)"),
	("addr_new_offset", "offset in that buffer"),
	("memory_descriptor", "1 if the target is the memory descriptor (LDCU c[0x0][0x358] into a UR pair, or addr_part desc): Masked there means output unchanged, not proven harmless"),
	("outcome", "NVBitFI category number (scripts/params.py)"),
	("outcome_str", "category text"),
	("outcome_class", "sdc, crash, timeout, masked, other (as recorded)"),
	("outcome_class_for_rates", "outcome_class after the decided corrections (final-v1 R1 runs counted as Masked)"),
	("masked_subclass", "memory Masked: no_read, overwritten, stale_only, read_masked, ambiguous"),
	("reads_old", "memory: traced loads that returned the old value"),
	("reads_new", "memory: traced loads that returned the flipped value"),
	("writes_after", "memory: stores to the word after the flip"),
	("read_kernels", "memory: kernels that read the word after the flip, ';'"),
	("exit_code", "exit code of the workload"),
	("cuda_error", "CUDA error string, if any"),
	("xids", "Xid numbers in dmesg during the run, ';'"),
	("signature", "JSON output signature of the SDC (wrong elements, pattern, bit classes; scripts/fp_signature.py)"),
	("wrong_elements", "SDC: number of output elements different from the golden (recomputed from the run's output files)"),
	("wrong_files", "SDC: output files with wrong elements, ';'"),
	("location_status", "SDC: expected (inside the area predicted from the record), expected_with_landing (addr groups: original or landing of the moved store), outside_proven_moved_store, outside_inferred_moved_store, outside_not_rechecked, no_output (no per-run outputs: the final-v1 uniform group, reused from the pilot); empty if not SDC"),
	("liveness_weight", "registers: weight of the run on bit x time (liveness_report.weight, liveness of 2026-11-07); empty if the site is not in the table or not a register group"),
	("runtime_s", "wall time of the run"),
	("suspended_s", "seconds the machine was suspended during the run"),
	("valid_for_rates", "1 if the run enters the rates"),
	("flags", "';' list: not_injected, suspended, v1_up_invalid (R5 zeroed by the NVBit UP write), v1_r1_counted_masked, r1_not_written (R1 among the cells, not written by design), cells_short, mode_s_dataset_only, gemm_pr_loop_control, stale_dmesg (whole old dmesg buffer recorded: xids dropped, Pot DUE with exit 0 counted as SDC)"),
]


def short(k):
	return (k or "").split("(")[0]


def cat(o):
	if o in cfl.SDC:
		return "sdc"
	if o in (1, 2, 3):
		return "masked"
	if o == 20 or o == 5:
		return "crash"
	if o == 4:
		return "timeout"
	return "other"


def j(v):
	return json.dumps(v, separators=(",", ":")) if v not in (None, "") else ""


LIVE_TABLES = {}


def live_weight(app, r):
	if app not in LIVE_TABLES:
		p = "%s/%s/liveness.jsonl" % (LIVE, app)
		if not os.path.isfile(p):
			p = "%s/%s/liveness.jsonl" % (LIVE_LARGE, app)
		LIVE_TABLES[app] = lr.site_table(lr.load(p)) if os.path.isfile(p) else None
	t = LIVE_TABLES[app]
	if t is None or r.get("reg_type") not in lr.BITS:
		return ""
	try:
		w = lr.weight(t, r)
	except (KeyError, ValueError, TypeError):
		return ""
	return "" if w is None else "%.6g" % w


def location(app, r, run_dir, block, name, igid, bfm):
	if r["outcome"] not in cfl.SDC:
		return {}
	wrong = cfl.wrong_elements(app, run_dir)
	if wrong is None:
		return {"location_status": "no_output"}
	out = {"wrong_elements": sum(len(v) for v in wrong.values()), "wrong_files": ";".join(k for k, v in wrong.items() if v)}
	if r["injector"] == "mem":
		if app == "shm_stencil":
			allowed = {"output_shm.bin": {r["cta"] * cfl.N_SHM + c for c in range(cfl.N_SHM)}}
		else:
			allowed = {}
			for c in r.get("cells") or [{"delta": 0}]:
				allowed = cfl.merge(allowed, cfl.mem_word_effect(app, r["buffer"], r["offset"] + c["delta"]))
		landing = False
	else:
		allowed = cfl.expected_threads(app, r)
		landing = bool(r.get("store_wrong_location"))
		if landing:
			allowed = cfl.merge(allowed, cfl.landing_effect(app, r))
	if cfl.inside(wrong, allowed):
		out["location_status"] = "expected_with_landing" if landing else "expected"
	elif (name, r["icount"]) in PROVEN:
		out["location_status"] = "outside_proven_moved_store"
	elif (name, r["icount"]) in INFERRED:
		out["location_status"] = "outside_inferred_moved_store"
	elif r["injector"] == "reg" and proven_population(block, igid, bfm):
		out["location_status"] = "outside_proven_moved_store"
	else:
		out["location_status"] = "outside_not_rechecked"
	return out


# (kernel, pc) of the LDCU that load the memory descriptor, c[0x0][0x358], from the SASS
DESCRIPTOR_SITES = {}


def descriptor_sites(app):
	if app not in DESCRIPTOR_SITES:
		sass = subprocess.run(["cuobjdump", "-sass", "%s/test-apps/%s/%s" % (ROOT, app, app)], stdout=subprocess.PIPE, universal_newlines=True).stdout
		sites, kernel = set(), None
		for line in sass.splitlines():
			m = re.match(r"\s+Function : _Z(\d+)(\w+)", line)
			if m:
				kernel = m.group(2)[:int(m.group(1))]
			m = re.search(r"/\*([0-9a-f]{4})\*/\s+LDCU\S*\s+UR\d+, c\[0x0\]\[0x358\]", line)
			if m and kernel:
				sites.add((kernel, "0x%x" % int(m.group(1), 16)))
		DESCRIPTOR_SITES[app] = sites
	return DESCRIPTOR_SITES[app]


def is_descriptor(r):
	if r.get("addr_part") == "desc":
		return True
	return r.get("opcode") == "LDCU" and r.get("reg_type") == "UR" and (short(r.get("kernel")), r.get("pc")) in descriptor_sites(r["app"])


NCU_DATA = {}
WS = {}
LIVE_GPR = {}


# live GPRs per thread per kernel (mean over the dynamic thread instructions), from the
# same liveness runs as the liveness weights
def live_gpr_per_thread(app):
	if app not in LIVE_GPR:
		out = {}
		for d in (LIVE, LIVE_LARGE):
			p = "%s/%s/liveness.jsonl" % (d, app)
			if os.path.isfile(p):
				agg = {}
				for rec in lr.load(p):
					a = agg.setdefault(short(rec["kernel"]), [0, 0])
					a[0] += sum(s["live"] for s in rec["sites"] if lr.class_name(s) == "GPR")
					a[1] += rec["thread_instrs"]
				out = {k: v[0] / v[1] for k, v in agg.items() if v[1]}
				break
		LIVE_GPR[app] = out
	return LIVE_GPR[app]


def peak_live(events):
	live, peak, size = 0, 0, {}
	for e in events:
		if e["kind"] == "alloc":
			size[e["id"]] = e["size"]
			live += e["size"]
			peak = max(peak, live)
		elif e["kind"] == "free" and e.get("id") in size:
			live -= size.pop(e["id"])
	return peak


def working_set(app):
	if app not in WS:
		ws = None
		p = SCALING + "/scaling.json"
		if os.path.isfile(p):
			t = json.load(open(p))["totals"].get(app)
			ws = t and t["bytes_peak_live"]
		if ws is None:
			profs = glob.glob("%s/profiles/mem-%s/mem-profile.json" % (FINAL, app)) + glob.glob("%s/dummy0/mem-%s-*/mem-profile.json" % (FINAL, app))
			ws = peak_live(json.load(open(profs[0]))) if profs else None
		WS[app] = ws
	return WS[app]


def features(app, kernel):
	base, param, size = SIZES.get(app, (app, "", ""))
	ws = working_set(app)
	d = {"workload": base, "size_param": param, "size": size, "working_set_bytes": ws if ws is not None else "",
		"working_set_l2_share": "%.4g" % min(ws / L2_BYTES, 1.0) if ws else ""}
	if not NCU_DATA and os.path.isfile(NCU):
		NCU_DATA.update(json.load(open(NCU))["workloads"])
	k = NCU_DATA.get(app, {}).get("kernels", {}).get(kernel)
	if k:
		t = k["time_ns_runs"]
		t = sum(t) / len(t) if isinstance(t, list) else t
		w = k["sm__warps_active.avg.per_cycle_active"]
		d.update({"kernel_gpu_us": "%.4g" % (t / 1e3), "kernel_warps_per_sm": "%.4g" % w,
			"kernel_rf_share": "%.4g" % (w * 32 * k["launch__registers_per_thread"] / RF_REGS_PER_SM),
			"kernel_l1_hit_pct": "%.4g" % k["l1tex__t_sector_hit_rate.pct"], "kernel_l2_hit_pct": "%.4g" % k["lts__t_sector_hit_rate.pct"]})
		lg = live_gpr_per_thread(app).get(kernel)
		if lg is not None:
			d["kernel_rf_live_share"] = "%.4g" % (lg * 32 * w / RF_REGS_PER_SM)
	return d


# Old get_dmesg_delta returned the whole buffer when the last line was not found
# (block 6, gemm_n2048 icount 227-232): a delta spanning more than a minute.
def stale_dmesg(r):
	ts = [float(x) for x in re.findall(r"\[\s*(\d+\.\d+)\]", str(r.get("dmesg") or ""))]
	return len(ts) > 1 and ts[-1] - ts[0] > 60


def fix_stale(r, flags):
	if not stale_dmesg(r):
		return
	flags.append("stale_dmesg")
	r["xids"] = []
	if r["outcome"] == 13 and r.get("exit_code") == 0:  # Pot DUE: output different, but dmesg recorded
		r["outcome"], r["outcome_str"] = 17, "SDC: Output file is different (reclassified: stale dmesg)"


def row_reg(r, campaign, block, name, run_dir):
	app = r["app"]
	flags = []
	fix_stale(r, flags)
	oc = cat(r["outcome"])
	ocr = oc
	valid = bool(r.get("injected", True)) and oc != "other"
	if not r.get("injected", True):
		flags.append("not_injected")
	if r.get("suspended_s", 0) > SUSPENDED_S:
		flags.append("suspended")
		valid = False
	if campaign == "final-v1":
		if r.get("reg_type") == "UP" and r["igid"] == 9:
			flags.append("v1_up_invalid")
			valid = False
		if r.get("reg_type") == "GPR" and r.get("reg_no") == 1 and oc != "masked":
			flags.append("v1_r1_counted_masked")
			ocr = "masked"
	if r.get("sp_not_written"):
		flags.append("r1_not_written")
	cells = r.get("cells") or []
	if r.get("requested_cells", 0) > len(cells) and cells:
		flags.append("cells_short")
	if app == "gemm" and r["igid"] == 3:
		flags.append("gemm_pr_loop_control")
	r["injector"] = "reg"
	d = {"app": app, "kernel": short(r.get("kernel")), "kernel_index": r.get("kcount"), "injector": "reg", "group": r.get("group"), "igid": r["igid"],
		"reg_class": r.get("reg_type"), "reg_no": r.get("reg_no"), "for_avf": 1, "fault_model": REG_MODELS.get(r.get("model"), (r.get("model") or "").lower() or "single_bit"),
		"k": r.get("k", 1), "cells_requested": r.get("requested_cells", 1), "cells_injected": len(cells) if cells else (1 if r.get("injected", True) else 0),
		"cells": j(cells), "mask": r.get("mask"), "opcode": r.get("opcode"), "pc": r.get("pc"), "inst_id": r.get("inst_id"), "group_instrs": r.get("group_instrs"),
		"tid": r.get("tid"), "cta": r.get("cta"), "sm": r.get("sm"), "warp_slot": r.get("warp_slot"), "lane": r.get("lane"), "before": r.get("before"), "after": r.get("after"),
		"addr_part": r.get("addr_part"), "addr_space": r.get("addr_space"), "addr_class": r.get("addr_class"), "addr_valid": int(r["addr_valid"]) if r.get("addr_valid") is not None else "",
		"addr_bit": r.get("addr_bit"), "store_wrong_location": int(r["store_wrong_location"]) if r.get("store_wrong_location") is not None else "",
		"addr_new_buffer": r.get("addr_new_buffer"), "addr_new_offset": r.get("addr_new_offset"), "memory_descriptor": int(is_descriptor(r)),
		"liveness_weight": live_weight(app, r)}
	d.update(features(app, d["kernel"]))
	d.update(common(r, campaign, name, oc, ocr, valid, flags))
	if r.get("injected", True) and os.path.isdir(run_dir):
		d.update(location(app, r, run_dir, block, name, r["igid"], r.get("bfm", 0)))
	elif r["outcome"] in cfl.SDC:
		d["location_status"] = "no_output"
	return d


def row_mem(r, campaign, block, name, run_dir, model, k, step):
	app = r["app"]
	flags = []
	fix_stale(r, flags)
	oc = cat(r["outcome"])
	valid = bool(r.get("injected")) and oc != "other"
	if not r.get("injected"):
		flags.append("not_injected")
	if r.get("suspended_s", 0) > SUSPENDED_S:
		flags.append("suspended")
		valid = False
	if r.get("mode") == "s":
		flags.append("mode_s_dataset_only")
	cells = r.get("cells") or []
	r["injector"] = "mem"
	d = {"seed": r.get("seed"), "app": app, "kernel": short(r.get("kernel_name")), "kernel_index": r.get("flip_kernel"), "injector": "mem", "group": r.get("structure"),
		"mem_mode": r.get("mode"), "for_avf": 0 if r.get("mode") == "s" else 1, "fault_model": model, "k": k, "mcu_step": step or "",
		"cells_requested": r.get("cells_requested", 1), "cells_injected": sum(c.get("applied", 0) for c in cells) if cells else int(bool(r.get("injected"))),
		"cells": j(cells), "mask": r.get("mask") or (hex(int(r["old_word"], 16) ^ int(r["new_word"], 16)) if r.get("old_word") and r.get("new_word") else ""),
		"opcode": "", "pc": r.get("pc"), "inst_id": r.get("dyn_instr"), "tid": r.get("tid"), "cta": r.get("cta"), "sm": r.get("sm"), "warp_slot": r.get("warp_slot"), "lane": r.get("lane"),
		"buffer": r.get("buffer"), "buffer_size": r.get("buffer_size"), "offset": r.get("offset"), "bit": r.get("bit"), "old_word": r.get("old_word"), "new_word": r.get("new_word"),
		"residents": r.get("residents", ""), "masked_subclass": r.get("masked_subclass") or "", "reads_old": r.get("reads_old"), "reads_new": r.get("reads_new"),
		"writes_after": r.get("writes_after"), "read_kernels": ";".join(str(x) for x in r.get("read_kernels") or []), "memory_descriptor": 0}
	d.update(features(app, d["kernel"]))
	d.update(common(r, campaign, name, oc, oc, valid, flags))
	if r.get("injected") and os.path.isdir(run_dir):
		d.update(location(app, r, run_dir, block, name, None, None))
	elif r["outcome"] in cfl.SDC:
		d["location_status"] = "no_output"
	return d


def common(r, campaign, name, oc, ocr, valid, flags):
	return {"run_id": "%s/%s/%d" % (campaign, name, r["icount"]), "campaign": campaign, "cell": name, "icount": r["icount"],
		"outcome": r["outcome"], "outcome_str": r.get("outcome_str"), "outcome_class": oc, "outcome_class_for_rates": ocr,
		"exit_code": r.get("exit_code"), "cuda_error": r.get("cuda_error") or "", "xids": ";".join(str(x) for x in r.get("xids") or []),
		"signature": j(r.get("signature")), "runtime_s": r.get("runtime_s"), "suspended_s": r.get("suspended_s"),
		"valid_for_rates": int(valid), "flags": ";".join(flags)}


def cell_blocks():
	out = {}
	for b, parts in rfb.blocks().items():
		for p in parts:
			for c in p["cells"]:
				out[c.name] = (b, c)
	return out


def main():
	dest = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else FINAL + "/dataset")
	os.makedirs(dest, exist_ok=True)
	names = [c for c, _ in COLUMNS]
	n = 0
	counts = {}
	with open(dest + "/runs.csv", "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
		w.writeheader()

		def emit(d):
			nonlocal n
			w.writerow({k: ("" if d.get(k) is None else d.get(k)) for k in names})
			n += 1
			counts[d["campaign"]] = counts.get(d["campaign"], 0) + 1

		# final-v1: gemm and sfu_math, single bit
		for path in sorted(glob.glob(V1 + "/*/results-*.jsonl")):
			app = os.path.basename(os.path.dirname(path))
			igid = int(re.search(r"igid(\d+)", path).group(1))
			name = "reg-%s-igid%d-bfm0" % (app, igid)
			for line in open(path):
				r = json.loads(line)
				rd = "%s/%s/%s-group%d-model%d-icount%d" % (V1, app, app, igid, r.get("bfm", 0), r["icount"])
				emit(row_reg(r, "final-v1", 0, name, rd))
		# blocks 1-5
		blocks = cell_blocks()
		for d in sorted(glob.glob(FINAL + "/dummy0/reg-*") + glob.glob(FINAL + "/dummy0/mem-*")):
			name = os.path.basename(d)
			if name not in blocks:
				sys.exit("cell %s is not in run_final_block.py" % name)
			b, c = blocks[name]
			campaign = "final-block%d" % b
			if c.tool == "reg":
				for path in glob.glob(d + "/logs/*/results-*.jsonl"):
					for line in open(path):
						r = json.loads(line)
						emit(row_reg(r, campaign, b, name, cfl.run_dir_of_reg(d, r["app"], r)))
			else:
				for path in glob.glob(d + "/results-mode*.jsonl"):
					for line in open(path):
						r = json.loads(line)
						rd = "%s/runs-mode%s/icount%d" % (d, r["mode"], r["icount"])
						emit(row_mem(r, campaign, b, name, rd, c.mcu or "single_bit", c.k or 1, c.step))
	json.dump({"file": "runs.csv", "rows": n, "rows_per_campaign": counts, "generated_by": "scripts/export_dataset.py",
		"columns": [{"name": c, "description": t} for c, t in COLUMNS]}, open(dest + "/schema.json", "w"), indent=1)
	write_md(dest, n, counts)
	print("%d rows -> %s/runs.csv" % (n, dest), counts)


NOTES = [
	"One row per run. Gate runs (DUMMY builds, no flip) and pilots are not included.",
	"Rates: use the rows with valid_for_rates = 1 and outcome_class_for_rates; for an AVF also for_avf = 1. These reproduce the SUMMARY files.",
	"Register MCU rows: the target fields (reg_no, before, after, tid, lane) are those of the first cell; all cells are in `cells`.",
	"Memory rows: opcode is empty (the flip is not tied to an instruction of the workload); pc and inst_id are those of the thread that flipped (modes b, u, s).",
	"Shared memory (mode u/s) exists only for shm_stencil; uniform predicates (UP) only on gemm and cnn. Missing combinations are empty by construction, not missing data.",
	"memory_descriptor = 1 rows: Masked means output unchanged (tests/desc_probe/README.md).",
	"gemm PR rows (gemm_pr_loop_control) are loop-control predicates of gemm_core only, as are most PR rows of the other workloads.",
	"liveness_weight: relative weight of the run if the population is resampled on bit x time; normalise within a population. Empty for address groups (no destination register).",
	"Location: recomputed from the output files with scripts/check_fault_locations.py; the outside_* labels come from ANOMALIES-2026-11-07.md, SUMMARY-block5.md and SUMMARY-block6.md.",
	"Workload size (block 6): gemm_n1024/gemm_n2048/cnn_large are gemm/cnn at a larger size (`workload`, `size`). Rates of global memory mode b depend on the size (SUMMARY-block6.md): do not pool sizes for that rate. kernel_* features are measured without a tool (ncu) and are the same for every run of a kernel.",
]


def write_md(dest, n, counts):
	L = ["# Dataset of the final campaigns (generated)", "", "`runs.csv`: %d rows, from `scripts/export_dataset.py`. Per campaign: %s." % (n, ", ".join("%s %d" % kv for kv in sorted(counts.items()))), "",
		"## Use", ""] + ["- " + x for x in NOTES] + ["", "## Columns", "", "| Column | Meaning |", "|---|---|"]
	L += ["| `%s` | %s |" % (c, t.replace("|", "/")) for c, t in COLUMNS]
	open(dest + "/SCHEMA.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
	main()
