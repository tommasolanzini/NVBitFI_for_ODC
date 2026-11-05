#!/usr/bin/env python3
# Offline test of scripts/run_final_block.py: cells of the plan and the gate. No GPU.
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import run_final_block as fb

passed = failed = 0


def check(name, got, want):
	global passed, failed
	ok = got == want
	passed += ok
	failed += not ok
	if not ok:
		print("FAIL %s: got %r, expected %r" % (name, got, want))


check("split 600", fb.split_k(600), {2: 400, 3: 100, 4: 100})
check("split 300", fb.split_k(300), {2: 200, 3: 50, 4: 50})

b = fb.blocks()
real = lambda parts: sum(c.n for p in parts for c in p["cells"])
check("block 1 runs", real(b[1]), 10000)
check("block 2 runs", real(b[2]), 2000 + 4000 + 9000)
check("block 3 runs", real(b[3]), 3000 + 600 + 1000 + 3000 + 2400 + 4800)
check("block 4 runs", real(b[4]), 4800)
names = [c.name for bl in b.values() for p in bl for c in p["cells"]]
check("no cell twice", len(names), len(set(names)))
gates = [c.name for bl in b.values() for p in bl for c in p["gate"]]
check("a gate never shares its folder with a cell of another size", all(g in names for g in gates), True)
check("every memory MCU word cell has a step", all(c.step for bl in b.values() for p in bl for c in p["cells"] if c.mcu == "adj_words"), True)
check("no lanes on uniform registers", any(c.igid == fb.G_UNIFORM and c.bfm // 10 == fb.ADJ_LANES for bl in b.values() for p in bl for c in p["cells"]), False)
check("no adjacent registers on predicate registers", any(c.igid == fb.G_PR and c.bfm // 10 == fb.ADJ_REGS for bl in b.values() for p in bl for c in p["cells"]), False)
check("store stratum on 3 workloads", sorted(c.app for p in b[3] for c in p["cells"] if c.igid == fb.G_ADDR_STORE), ["cnn", "gemm", "ptr_chase"])
check("single bit GPR on 3 workloads", sorted(c.app for p in b[3] for c in p["cells"] if c.igid == fb.G_GP and c.bfm == 0), ["ptr_chase", "sfu_math", "shm_stencil"])
check("gemm uniform again, single bit", [(c.n, c.bfm) for bl in b.values() for p in bl for c in p["cells"] if c.app == "gemm" and c.igid == fb.G_UNIFORM and c.bfm == 0], [(1000, 0)])
check("gemm uniform in block 3", [c.n for p in b[3] for c in p["gate"] if c.app == "gemm" and c.igid == fb.G_UNIFORM and c.bfm == 0], [100])
c = b[1][0]["cells"][0]
check("seed differs between gate and cell", c.seed("dummy1") != c.seed("dummy0"), True)
check("seed is stable", c.seed("dummy0"), fb.Cell("mem", "gemm", 1, mode="a").seed("dummy0"))
check("name of a memory MCU cell", fb.Cell("mem", "gemm", 1, mode="a", mcu="adj_words", k=2, step=8).name, "mem-gemm-modea-adj_words-k2-step8")


def rec(**kw):
	r = {"icount": 1, "outcome": 3, "outcome_str": "Masked", "injected": True, "kernel": "k", "group_instrs": 10, "cells": [{"before": "0x1", "after": "0x1"}]}
	r.update(kw)
	return r


prof = {"k": {"gp": 10}}
check("gate passes", fb.gate_problems_reg([rec()], 1, prof, "gp"), [])
check("SDC stops", len(fb.gate_problems_reg([rec(outcome=17, outcome_str="SDC")], 1, prof, "gp")), 1)
check("crash stops", len(fb.gate_problems_reg([rec(outcome=20, outcome_str="DUE")], 1, prof, "gp")), 1)
check("timeout stops", len(fb.gate_problems_reg([rec(outcome=4, outcome_str="DUE")], 1, prof, "gp")), 1)
check("uncategorized stops", len(fb.gate_problems_reg([rec(outcome=19, outcome_str="?")], 1, prof, "gp")), 1)
check("missing runs stop", len(fb.gate_problems_reg([rec()], 2, prof, "gp")), 1)
check("not injected stops", len(fb.gate_problems_reg([rec(injected=False)], 1, prof, "gp")), 1)
check("changed register stops", len(fb.gate_problems_reg([rec(cells=[{"before": "0x1", "after": "0x3"}])], 1, prof, "gp")), 1)
check("wrong count stops", len(fb.gate_problems_reg([rec(group_instrs=9)], 1, prof, "gp")), 1)
m = {"icount": 1, "outcome": 3, "injected": True, "old_word": "0x1", "new_word": "0x1", "cells": [{"old_word": "0x1", "new_word": "0x1"}]}
check("memory gate passes", fb.gate_problems_mem([m], 1), [])
check("memory SDC stops", len(fb.gate_problems_mem([dict(m, outcome=17, outcome_str="SDC")], 1)), 1)
check("memory changed word stops", len(fb.gate_problems_mem([dict(m, new_word="0x3")], 1)), 1)
check("memory changed cell stops", len(fb.gate_problems_mem([dict(m, cells=[{"old_word": "0x1", "new_word": "0x3"}])], 1)), 1)

import tempfile
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
	f.write("NVBit-igprofile; index: 0; kernel_name: k(float*); ctas: 1; instrs: 5; FADD: 3, addr: 7, addr_store: 2, gp: 4,\n")
pr = fb.parse_profile(f.name)
check("profile by name", (pr["k(float*)"]["addr"], pr["k(float*)"]["addr_store"], pr["k(float*)"]["gp"]), (7, 2, 4))
os.unlink(f.name)

import json, types
with tempfile.TemporaryDirectory() as d:
	os.makedirs(d + "/config")
	with open(d + "/tool.so", "w") as f:
		f.write("abc")
	drv = fb.Driver(types.SimpleNamespace(root=d))
	drv.record_hash(d + "/tool.so", kind="tool", dummy=2)
	drv.record_hash(d + "/missing.so", kind="tool", dummy=0)
	hashes = [json.loads(l) for l in open(d + "/config/hashes.jsonl")]
check("hash of a file", hashes[0]["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
check("hash keeps the DUMMY value", hashes[0]["dummy"], 2)
check("hash of a missing file", hashes[1]["sha256"], None)

print("%d/%d final block cases passed" % (passed, passed + failed))
sys.exit(1 if failed else 0)
