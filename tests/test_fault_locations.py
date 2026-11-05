#!/usr/bin/env python3
# Offline test of scripts/check_fault_locations.py on sfu_math, with made-up outputs. No GPU.
import json, os, struct, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import check_fault_locations as cl

passed = failed = 0
N = 128
RSQRT = "k_rsqrt(floatconst*,float*,int)"
SIN = "k_sin(floatconst*,float*,int)"
LOG = "k_log(floatconst*,float*,int)"


def check(name, got, want):
	global passed, failed
	ok = got == want
	passed += ok
	failed += not ok
	if not ok:
		print("FAIL %s: got %r, expected %r" % (name, got, want))


def write_words(path, words):
	with open(path, "wb") as f:
		f.write(struct.pack("<%dI" % len(words), *words))


def rec(icount, kernel=RSQRT, tid=100, reg_type="GPR", **kw):
	r = {"icount": icount, "igid": 12, "bfm": 0, "outcome": 16, "kernel": kernel, "tid": tid, "cta": tid // N, "reg_type": reg_type, "reg_no": 2, "cells": [{"tid": tid, "lane": tid % 32, "reg_no": 2}], "opcode": "MOV", "pc": "0x0"}
	r.update(kw)
	return r


def scenario(tmp, runs):
	folder = tmp + "/reg-sfu_math-igid12-bfm0"
	logs = folder + "/logs/sfu_math"
	os.makedirs(logs)
	golden = [7] * (N * N)
	with open(logs + "/results-x.jsonl", "w") as f:
		for r, changes in runs:
			f.write(json.dumps(r) + "\n")
			if changes is None:
				continue
			d = "%s/sfu_math-group12-model0-icount%d" % (logs, r["icount"])
			os.makedirs(d)
			for func in cl.SFU_FUNCS:
				words = list(golden)
				for i in changes.get(func, []):
					words[i] = 8
				write_words("%s/output_%s.bin" % (d, func), words)
	return cl.check_reg(folder)


# Positions, no files
got = cl.expected_threads("sfu_math", rec(1))
check("only the output of the kernel", {k: sorted(v) for k, v in got.items() if v}, {"output_rsqrt.bin": [100]})
check("all outputs are listed", sorted(got), sorted("output_%s.bin" % f for f in cl.SFU_FUNCS))
got = cl.expected_threads("sfu_math", rec(1, kernel=SIN, reg_type="UR"))
check("uniform register: the warp", got["output_sin.bin"], set(range(96, 128)))
check("uniform register: not another output", got["output_rsqrt.bin"], set())
check("uniform predicate: the warp", cl.expected_threads("sfu_math", rec(1, reg_type="UP"))["output_rsqrt.bin"], set(range(96, 128)))
two = rec(1, cells=[{"tid": 100, "lane": 4, "reg_no": 2}, {"tid": 101, "lane": 5, "reg_no": 2}])
check("lanes: every cell", cl.expected_threads("sfu_math", two)["output_rsqrt.bin"], {100, 101})
check("landing in the input: this kernel and the later ones", sorted(cl.landing_effect("sfu_math", rec(1, kernel=SIN, addr_new_buffer=0, addr_new_offset=40))), ["output_exp.bin", "output_log.bin", "output_sin.bin"])
check("landing in the input: the element", cl.landing_effect("sfu_math", rec(1, kernel=LOG, addr_new_buffer=0, addr_new_offset=40)), {"output_log.bin": {10}})
check("landing in the output: this kernel", cl.landing_effect("sfu_math", rec(1, kernel=SIN, addr_new_buffer=1, addr_new_offset=40)), {"output_sin.bin": {10}})
check("landing elsewhere", cl.landing_effect("sfu_math", rec(1, addr_new_buffer=7, addr_new_offset=40)), {})
check("other workloads unchanged", cl.expected_threads("ptr_chase", {"reg_type": "GPR", "tid": 5, "cells": []}), {"output_ptr.bin": {5}})

# Whole check on files
with tempfile.TemporaryDirectory() as tmp:
	cl.ROOT = tmp
	g = tmp + "/test-apps/sfu_math"
	os.makedirs(g)
	for func in cl.SFU_FUNCS:
		write_words("%s/golden_output_%s.bin" % (g, func), [7] * (N * N))
	res = scenario(tmp, [
		(rec(1), {"rsqrt": [100]}),                        # right element
		(rec(2), {"rsqrt": [101]}),                        # another thread
		(rec(3), {"rcp": [100]}),                          # right thread, another kernel
		(rec(4, reg_type="UR"), {"rsqrt": list(range(96, 128))}),      # the whole warp
		(rec(5, reg_type="UR"), {"rsqrt": [128]}),                     # outside the warp
		(rec(6, kernel=SIN, store_wrong_location=True, addr_new_buffer=0, addr_new_offset=40), {"sin": [100, 10], "log": [10]}),
		(rec(7), None),                                    # no output
		(rec(8, outcome=3), None),                         # Masked: not looked at
	])
check("runs", res["runs"], 8)
check("SDCs", res["sdc"], 7)
check("in position", res["ok"], 3)
check("no output", res["no_output"], 1)
check("exceptions", sorted(e["icount"] for e in res["exceptions"]), [2, 3, 5])
check("landing seen", (res["wrong_location"]["runs"], res["wrong_location"]["original_changed"], res["wrong_location"]["landing_seen"]), (1, 1, 1))

print("%d/%d location cases passed" % (passed, passed + failed))
sys.exit(1 if failed else 0)
