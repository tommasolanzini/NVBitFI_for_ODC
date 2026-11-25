#!/usr/bin/env python3
# Stage-1 dataset: one row per visible run (SDC, crash, timeout), features observable
# without the injection record. No GPU.
#
#   ~/thesis/venv-ml/bin/python ml/build_dataset.py [runs.csv] [out.parquet]
#
# Input: logs/final-2026-11-05/dataset/runs.csv (scripts/export_dataset.py) and the
# per-run output files. Labels and selection: ml/label_map.yaml through ml/labels.py.
#
# Column prefixes:
#   f_     observable features (outcome, CUDA error, Xid, output signature, geometry);
#          f_xid13_* / f_xid31_*: details of the Xid messages (ml/xid_features.py)
#   fw_    workload one-hot and size: the trainer runs with and without them
#   meta_  metadata for labels, splits and analysis; NEVER an input
#   label, split, trainable
# Forbidden as input (ground truth or not observable under beam): the injection record,
# location check, Masked subclasses, liveness weights, suspended_s, block, seed, hashes,
# runtime (only the timeout flag is kept). These stay out of f_/fw_ by construction.

import json, math, os, sys
import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import labels  # noqa: E402
import xid_features  # noqa: E402

RUNS = os.path.join(ROOT, "logs/final-2026-11-05/dataset/runs.csv")
OUT = os.path.join(HERE, "data/stage1_v0.parquet")
V1 = os.path.join(ROOT, "logs/final-v1-2026-11-03-gemm")
FINAL = os.path.join(ROOT, "logs/final-2026-11-05")
APPS = os.path.join(ROOT, "test-apps")

WORKLOADS = ["gemm", "cnn", "ptr_chase", "shm_stencil", "sfu_math"]
# Output files per base workload, in launch order. All are fp32, row-major.
PARTS = {"gemm": ["core", "tc"], "cnn": ["logits"], "ptr_chase": ["ptr"], "shm_stencil": ["shm"],
	"sfu_math": ["rsqrt", "rcp", "sin", "exp", "log"]}
# Apps that write one output file per kernel: crash features "kernels completed" exist.
PER_KERNEL_FILES = {"gemm", "sfu_math"}
# Value of an output element before its kernel writes it (from the sources): only sfu_math
# clears its buffer (cudaMemset 0 before each kernel). The others use cudaMalloc only, so
# the initial value is undefined -> NaN. ptr_chase even allocates `out` after freeing `scale`.
INIT_VALUE = {"sfu_math": 0}
# Matrix outputs, for the rank of the difference (gemm: C; cnn: batch x classes logits)
MATRIX = {"gemm", "cnn"}
# Binary that holds the code, for the static site (gemm_n1024/n2048 share gemm_large)
BINARY = {"gemm_n1024": "gemm_large", "gemm_n2048": "gemm_large"}

CUDA_ERRORS = {"an illegal memory access was encountered": "illegal_address",
	"an illegal instruction was encountered": "illegal_instruction",
	"misaligned address": "misaligned",
	"operation not supported on global/shared address space": "unsupported_space"}
# Execution exceptions, the same in FI and under beam; platform-neutral names
XIDS = {31: "mmu_fault", 13: "sm_exception", 43: "channel_stopped"}
PATTERNS = ["single", "row", "column", "block", "scattered"]
MIN_BLOCK_FILL = 0.5  # as scripts/fp_signature.py
REL_FLOOR = 1e-10
SITE_LINE = 128  # bytes, memory site granularity
RANK_SIDE = 256

META = ["run_id", "campaign", "cell", "icount", "app", "workload", "kernel", "kernel_index", "pc", "opcode",
	"injector", "group", "igid", "reg_class", "mem_mode", "buffer", "offset", "addr_part", "memory_descriptor",
	"fault_model", "k", "flags", "outcome_class_for_rates"]


def golden_dir(app):
	if app in ("gemm_n1024", "gemm_n2048"):
		return os.path.join(APPS, "gemm_large", "golden-" + app)
	return os.path.join(APPS, app)


def shape(app, n_elems):
	if app == "cnn_large":
		return n_elems // 32, 32
	s = math.isqrt(n_elems)
	assert s * s == n_elems, (app, n_elems)
	return s, s


_golden = {}


def golden(app, part):
	key = (app, part)
	if key not in _golden:
		g = np.fromfile(os.path.join(golden_dir(app), "golden_output_%s.bin" % part), dtype="<u4")
		_golden[key] = (g, np.unique(g))
	return _golden[key]


def run_dir(r):
	if r["campaign"] == "final-v1":
		return "%s/%s/%s-group%s-model0-icount%s" % (V1, r["app"], r["app"], r["igid"], r["icount"])
	d = "%s/dummy0/%s" % (FINAL, r["cell"])
	if r["injector"] == "mem":
		return "%s/runs-mode%s/icount%s" % (d, r["mem_mode"], r["icount"])
	bfm = r["cell"].rsplit("bfm", 1)[1]
	return "%s/logs/%s/%s-group%s-model%s-icount%s" % (d, r["app"], r["app"], r["igid"], bfm, r["icount"])


def read_part(rd, app, part):
	"""(indices of the wrong elements, their output words) of one part, or None if the run
	did not write it. Large apps keep only wrong-<part>.bin (flat index, output XOR golden)."""
	g, _ = golden(app, part)
	p = os.path.join(rd, "output_%s.bin" % part)
	if os.path.isfile(p):
		o = np.fromfile(p, dtype="<u4")
		if len(o) != len(g):
			return None
		idx = np.flatnonzero(o != g)
		return idx, o[idx]
	p = os.path.join(rd, "wrong-%s.bin" % part)
	if os.path.isfile(p):
		pairs = np.fromfile(p, dtype="<u4").reshape(-1, 2)
		idx = pairs[:, 0].astype(np.int64)
		order = np.argsort(idx)
		return idx[order], (g[idx] ^ pairs[:, 1])[order]
	return None


def in_sorted(v, u):
	"""v in u, u sorted and unique (np.isin would sort u again on every call)."""
	pos = np.minimum(np.searchsorted(u, v), len(u) - 1)
	return u[pos] == v


def pattern(rows, cols):
	if len(rows) == 1:
		return "single"
	if rows.min() == rows.max():
		return "row"
	if cols.min() == cols.max():
		return "column"
	area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
	return "block" if len(rows) / area >= MIN_BLOCK_FILL else "scattered"


def rel_errors(o, g):
	with np.errstate(invalid="ignore"):
		x, y = o.view("<f4").astype(np.float64), g.view("<f4").astype(np.float64)
	fin = np.isfinite(x) & np.isfinite(y)
	x, y = x[fin], y[fin]
	d = np.abs(x - y)
	rel = np.where(y != 0, d / np.where(y != 0, np.abs(y), 1), d)
	return rel


def diff_rank(vals, g_full, nr, nc, idx):
	"""Rank of the relative difference restricted to the rows and columns that hold wrong
	elements, at most RANK_SIDE of each (fixed seed): exact up to RANK_SIDE, else a lower bound."""
	rows, cols = np.unique(idx // nc), np.unique(idx % nc)
	rng = np.random.default_rng(0)
	if len(rows) > RANK_SIDE:
		rows = np.sort(rng.choice(rows, RANK_SIDE, replace=False))
	if len(cols) > RANK_SIDE:
		cols = np.sort(rng.choice(cols, RANK_SIDE, replace=False))
	y = g_full.view("<f4").reshape(nr, nc)[np.ix_(rows, cols)].astype(np.float64)
	x = y.copy()
	r, c = idx // nc, idx % nc
	keep = np.isin(r, rows) & np.isin(c, cols)
	ri, ci = np.searchsorted(rows, r[keep]), np.searchsorted(cols, c[keep])
	with np.errstate(all="ignore"):
		x[ri, ci] = vals[keep].view("<f4").astype(np.float64)
		rel = (x - y) / np.maximum(np.abs(y), 1e-30)
	rel = np.clip(np.nan_to_num(rel, nan=1e3, posinf=1e3, neginf=-1e3), -1e3, 1e3)
	return float(np.linalg.matrix_rank(rel))


def output_features(r, rd):
	"""Features of the outputs. SDC: signature and geometry. Every visible run: kernels completed."""
	app, wl = r["app"], r["workload"]
	parts = PARTS[wl]
	f = {}
	present, correct, wrong_parts = 0, 0, []
	for part in parts:
		res = read_part(rd, app, part)
		if res is None:
			continue
		present += 1
		idx, vals = res
		if len(idx) == 0:
			correct += 1
		else:
			wrong_parts.append((part, vals, golden(app, part)[0], idx))
	if wl in PER_KERNEL_FILES:
		f["f_partial_na"] = 0
		f["f_frac_kernels_completed"] = present / len(parts)
		f["f_frac_completed_correct"] = correct / present if present else np.nan
	else:
		f["f_partial_na"] = 1
		f["f_frac_kernels_completed"] = np.nan
		f["f_frac_completed_correct"] = np.nan
	if r["outcome_class_for_rates"] != "sdc":
		return f
	if not wrong_parts or present != len(parts):
		f["_sdc_without_outputs"] = 1
		return f

	total = sum(len(golden(app, p)[0]) for p in parts)
	o_all = np.concatenate([v for _, v, _, _ in wrong_parts])
	g_all = np.concatenate([g[idx] for _, _, g, idx in wrong_parts])
	n = len(o_all)
	xor = o_all ^ g_all
	msb = np.frexp(xor.astype(np.float64))[1] - 1  # 0..31
	ham = np.bitwise_count(xor)
	x = o_all.view("<f4")
	nonfin = ~np.isfinite(x)
	rel = rel_errors(o_all, g_all)

	f["f_log10_wrong"] = math.log10(n)
	f["f_frac_wrong"] = n / total
	f["f_frac_outputs_wrong"] = len(wrong_parts) / len(parts)
	f["f_frac_nonfinite"] = nonfin.mean()
	f["f_log10_max_rel"] = math.log10(max(rel.max(), REL_FLOOR)) if len(rel) else np.nan
	f["f_log10_med_rel"] = math.log10(max(np.median(rel), REL_FLOOR)) if len(rel) else np.nan
	f["f_bit_sign"] = (msb == 31).mean()
	f["f_bit_exponent"] = ((msb >= 23) & (msb < 31)).mean()
	f["f_bit_mantissa"] = (msb < 23).mean()
	f["f_msb_max"] = msb.max() / 31
	f["f_msb_median"] = np.median(msb) / 31
	f["f_hamming_mean"] = ham.mean() / 32
	f["f_hamming_max"] = ham.max() / 32
	f["f_frac_eq_init"] = (o_all == INIT_VALUE[wl]).mean() if wl in INIT_VALUE else np.nan
	f["f_frac_eq_other_golden"] = np.mean(np.concatenate([in_sorted(v, golden(app, p)[1]) for p, v, _, _ in wrong_parts]))

	# Geometry on the output with most wrong elements (rows x columns as written)
	part, vals, g, idx = max(wrong_parts, key=lambda t: len(t[3]))
	nr, nc = shape(app, len(g))
	rows, cols = idx // nc, idx % nc
	pat = pattern(rows, cols)
	for p in PATTERNS:
		f["f_pattern_" + p] = int(pat == p)
	r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
	bh, bw = r1 - r0 + 1, c1 - c0 + 1
	f["f_frac_rows"] = len(np.unique(rows)) / nr
	f["f_frac_cols"] = len(np.unique(cols)) / nc
	f["f_bbox_rows_frac"] = bh / nr
	f["f_bbox_cols_frac"] = bw / nc
	f["f_bbox_fill"] = len(idx) / (bh * bw)
	mask = np.zeros((bh, bw), dtype=bool)
	mask[rows - r0, cols - c0] = True
	_, ncl = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
	f["f_log10_clusters"] = math.log10(ncl)
	# Hardware-neutral granularities: an aligned 16x16 tile (MMA fragment), an aligned
	# 32-element row segment (one warp writing consecutive elements)
	f["f_bbox_in_tile16"] = int(r0 // 16 == r1 // 16 and c0 // 16 == c1 // 16)
	f["f_bbox_eq_tile16"] = int(bh == 16 and bw == 16 and r0 % 16 == 0 and c0 % 16 == 0 and len(idx) == 256)
	f["f_eq_warp32"] = int(bh == 1 and bw == 32 and c0 % 32 == 0 and len(idx) == 32)
	f["f_rank_log2"] = math.log2(1 + diff_rank(vals, g, nr, nc, idx)) if wl in MATRIX else np.nan
	f["f_cnn_frac_images"] = len(np.unique(rows)) / nr if wl == "cnn" else np.nan
	return f


def base_features(r):
	oc = r["outcome_class_for_rates"]
	f = {"f_out_sdc": int(oc == "sdc"), "f_out_crash": int(oc == "crash"), "f_out_timeout": int(oc == "timeout")}
	err = r["cuda_error"]
	for v in CUDA_ERRORS.values():
		f["f_cuda_" + v] = int(CUDA_ERRORS.get(err) == v)
	f["f_cuda_other"] = int(bool(err) and err not in CUDA_ERRORS)
	xids = [int(x) for x in r["xids"].split(";") if x]
	for num, name in XIDS.items():
		c = xids.count(num)
		f["f_xid_" + name] = int(c > 0)
		f["f_xid_%s_count" % name] = c
	f["f_xid_other"] = int(any(x not in XIDS for x in xids))
	for w in WORKLOADS:
		f["fw_wl_" + w] = int(r["workload"] == w)
	f["fw_log10_size"] = math.log10(float(r["size"]))
	return f


def site(r):
	b = BINARY.get(r["app"], r["app"])
	if r["injector"] == "mem":
		# memory has no code site: structure, buffer and the 128-byte line (L2 line) of the word
		return "%s|mem_%s|buf%s|line%d" % (b, r["group"], r["buffer"], int(r["offset"]) // SITE_LINE)
	return "%s|%s|%s" % (b, r["kernel"], r["pc"])


def main():
	src = sys.argv[1] if len(sys.argv) > 1 else RUNS
	dest = sys.argv[2] if len(sys.argv) > 2 else OUT
	runs = pd.read_csv(src, dtype=str, keep_default_na=False)
	rows, problems, unmapped = [], [], {}
	dmesg = xid_features.dmesg_index()
	for r in runs.to_dict("records"):
		if not labels.select(r):
			continue
		cls = labels.label(r)
		if cls is None:
			key = (r["injector"], r["group"], r["reg_class"], r["opcode"])
			unmapped[key] = unmapped.get(key, 0) + 1
			continue
		d = {"meta_" + k: r[k] for k in META}
		d["meta_site"] = site(r)
		d["label"], d["split"], d["trainable"] = cls, labels.split(r), int(labels.trainable(cls))
		d.update(base_features(r))
		key = (r["campaign"] == "final-v1", r["cell"], int(r["icount"]))
		if key not in dmesg:
			problems.append(("no_dmesg_record", r["run_id"]))
		# stale_dmesg runs: the delta held the whole old buffer, their Xids are dropped in runs.csv
		dm = "" if "stale_dmesg" in r["flags"].split(";") else dmesg.get(key, "")
		d.update(xid_features.features(r["app"], dm))
		rd = run_dir(r)
		if not os.path.isdir(rd):
			problems.append(("no_run_dir", r["run_id"]))
		d.update(output_features(r, rd))
		if d.pop("_sdc_without_outputs", 0):
			problems.append(("sdc_without_outputs", r["run_id"]))
		rows.append(d)
	df = pd.DataFrame(rows)
	for c in ("meta_icount", "meta_k", "meta_kernel_index", "meta_offset"):
		df[c] = pd.to_numeric(df[c], errors="coerce")
	os.makedirs(os.path.dirname(dest), exist_ok=True)
	df.to_parquet(dest, index=False)
	feats = [c for c in df.columns if c.startswith("f_")]
	wl = [c for c in df.columns if c.startswith("fw_")]
	json.dump({"source": os.path.relpath(src, ROOT), "rows": len(df), "features": feats, "workload_features": wl,
		"xid_detail_features": xid_features.FEATURES,
		"meta": [c for c in df.columns if c.startswith("meta_")], "problems": problems,
		"unmapped": {"|".join(k): v for k, v in unmapped.items()}},
		open(dest.replace(".parquet", ".json"), "w"), indent=1)
	print("%d rows, %d features + %d workload -> %s" % (len(df), len(feats), len(wl), dest))
	print("unmapped:", unmapped or "none")
	print("problems:", len(problems), problems[:10])


if __name__ == "__main__":
	main()
