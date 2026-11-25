#!/usr/bin/env python3
# Features from the details of the Xid messages in the dmesg delta of a run (reduced
# version, 2026-11-25). Sources observable under beam only: the dmesg text of the run and
# the golden allocation profile of the workload. Never the injection record.
#
# What the messages contain on this GPU (driver 580, RTX PRO 1000 Blackwell):
#   Xid 13: SM warp exception type; no data address (the ESR words
#           that look like one are the same constant in every run and every binary).
#   Xid 31: MMU fault with the faulting address at 4 KiB page granularity (bits 0-11
#           lost) and the access type (read/write; no atomic observed).
# Buffer bases change between processes in bits >= 23; the low 23 bits (offset inside an
# 8 MiB region) are stable. Only buffers copied host<->device have a known address in the
# golden profile (mem-profile.json has no base for alloc events).
# The buffers sit in a mapped block of 4 MiB granules; a fault on the first page after the
# block (low 23 bits = end of the known buffers rounded up to 4 MiB) is a sequential access
# running off the mapping, not a bit flip of a pointer: f_xid31_end_of_block, and no k.

import glob, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL = os.path.join(ROOT, "logs/final-2026-11-05")
V1 = os.path.join(ROOT, "logs/final-v1-2026-11-03-gemm")

LOW = 23
LOW_MASK = (1 << LOW) - 1
PAGE = 12
K_RANGE = range(PAGE, LOW)  # bits observable and stable: 12..22
PAGE_BITS = LOW_MASK & ~((1 << PAGE) - 1)
CHUNK = 22  # mapping granule: 4 MiB

XID13_TYPES = {"Out Of Range Address": "oor", "Misaligned Address": "misaligned",
	"Invalid Address Space": "invalid_space", "Illegal Instruction Parameter": "illegal_param"}
XID13_CATS = ["oor", "misaligned", "invalid_space", "illegal_param", "other"]

RE_XID = re.compile(r"Xid \([^)]*\)[:-] (\d+),")
RE_WARP = re.compile(r"SM Warp Exception on \([^)]*\)[:-] ([A-Za-z ]+)")
RE_FAULT = re.compile(r"faulted @ 0x([0-9a-f_]+)\. Fault is of type \S+ ACCESS_TYPE_VIRT_(\w+)")

# Element size of a known buffer, from the sources: (base workload, buffer bytes) -> bytes.
# gemm: A, B, C, C_tc fp32 and A_h, B_h half; ptr_chase: scale fp32, heads pointers,
# nodes 16-byte structs; everything else fp32.
ELEM = {("gemm", 32768): 2, ("gemm", 2097152): 2, ("ptr_chase", 32768): 8, ("ptr_chase", 4194304): 16}

FEATURES = (["f_xid13_" + c for c in XID13_CATS] + ["f_xid31_read", "f_xid31_write",
	"f_xid31_end_of_block", "f_xid31_low23_in_buf", "f_xid31_bit_to_buf", "f_xid31_k",
	"f_xid31_hamming_min", "f_xid31_above_2_47"])


def _profile(app):
	"""([(low 23 bits of base, size, element size)] of the buffers with a known address, pages
	they cover, low 23 bits of the first page after the mapped block), or None (no profile, or
	the known buffers do not fit in one 8 MiB region)."""
	fs = glob.glob(os.path.join(FINAL, "profiles/mem-%s/mem-profile.json" % app))
	if not fs:
		return None
	known = {}
	for e in json.load(open(fs[0])):
		if e["kind"] in ("h2d", "d2h"):
			known[e.get("dst") or e.get("src")] = e["size"]
	base = {"gemm_n1024": "gemm", "gemm_n2048": "gemm", "cnn_large": "cnn"}.get(app, app)
	bufs = [(a & LOW_MASK, s, ELEM.get((base, s), 4)) for a, s in sorted(known.items())]
	if sum(s for _, s, _ in bufs) > (1 << LOW) or any(o + s > (1 << LOW) for o, s, _ in bufs):
		return None
	pages = np.unique(np.concatenate([np.arange(o >> PAGE, (o + s - 1 >> PAGE) + 1) << PAGE for o, s, _ in bufs]))
	end = max(o + s for o, s, _ in bufs)
	eob = (((end - 1) >> CHUNK) + 1 << CHUNK) & LOW_MASK
	return bufs, pages, eob


_PROFILES = {}


def profile(app):
	if app not in _PROFILES:
		_PROFILES[app] = _profile(app)
	return _PROFILES[app]


def _buffer_of(page, bufs):
	for o, s, e in bufs:
		if o < page + (1 << PAGE) and page < o + s:
			return o, s, e
	return None


def address_features(app, addr):
	f = {"f_xid31_above_2_47": int(addr >= 1 << 47)}
	prof = profile(app)
	if prof is None:
		return f
	bufs, pages, eob = prof
	page = addr & PAGE_BITS
	inside = _buffer_of(page, bufs) is not None
	f["f_xid31_low23_in_buf"] = int(inside)
	f["f_xid31_hamming_min"] = int(np.bitwise_count((pages ^ page) & PAGE_BITS).min())
	# addr >= 8 MiB: a null pointer (address 0) is not the end of the block
	f["f_xid31_end_of_block"] = int(page == eob and addr >= 1 << LOW)
	if inside or f["f_xid31_end_of_block"]:
		return f  # (b'), k: not applicable (stable bits in a buffer, or not a flip)
	for k in K_RANGE:
		if _buffer_of(page ^ (1 << k), bufs) is not None:
			# page granularity: k >= 12, so k_rel / misaligned / pointer-only flags are constant
			f.update({"f_xid31_bit_to_buf": 1, "f_xid31_k": k})
			return f
	f["f_xid31_bit_to_buf"] = 0
	return f


def features(app, dmesg):
	"""f_xid13_* and f_xid31_* of one run; NaN where not applicable (no such Xid)."""
	f = {c: np.nan for c in FEATURES}
	lines = [l for l in (dmesg or "").split("; ") if l]
	xid13 = [l for l in lines if any(x == "13" for x in RE_XID.findall(l))]
	if xid13:
		types = {XID13_TYPES.get(m.group(1).strip(), "other") for l in xid13 for m in RE_WARP.finditer(l)}
		if not types:
			types = {"other"}
		for c in XID13_CATS:
			f["f_xid13_" + c] = int(c in types)
	faults = [m for l in lines for m in RE_FAULT.finditer(l)]
	if faults:
		acc = {m.group(2) for m in faults}
		f["f_xid31_read"] = int("READ" in acc)
		f["f_xid31_write"] = int("WRITE" in acc)
		f.update(address_features(app, int(faults[0].group(1).replace("_", ""), 16)))
	return f


def dmesg_index():
	"""{(campaign is final-v1, cell, icount): dmesg delta} from the result JSONL files."""
	idx = {}
	for path in glob.glob(V1 + "/*/results-*.jsonl"):
		for line in open(path):
			r = json.loads(line)
			idx[(True, "reg-%s-igid%d-bfm0" % (r["app"], r["igid"]), r["icount"])] = r.get("dmesg") or ""
	for path in glob.glob(FINAL + "/dummy0/reg-*/logs/*/results-*.jsonl") + glob.glob(FINAL + "/dummy0/mem-*/results-mode*.jsonl"):
		cell = os.path.relpath(path, FINAL + "/dummy0").split("/")[0]
		for line in open(path):
			r = json.loads(line)
			idx[(False, cell, r["icount"])] = r.get("dmesg") or ""
	return idx
