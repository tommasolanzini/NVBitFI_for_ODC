#!/usr/bin/env python3
"""Flip each of the 64 bits of the memory descriptor, and all of them at once, and
compare the kernel time and the checksum with the unmodified run. Runs are in
random order, so that a drift of the GPU does not look like an effect of one bit.
Usage: sweep.py <repetitions> <output.json>"""
import json, os, random, re, statistics, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SETS = [64, 1024, 262144] # KiB: L1, L2, DRAM on the time plateaus of this GPU
ACCESSES = "1000000"

def run(ws, lo, hi, match=None):
	env = dict(os.environ, LD_PRELOAD=HERE + "/desc_tool.so", NOBANNER="1", ACK_CTX_INIT_LIMITATION="1",
		DESC_MASK_LO="%x" % lo, DESC_MASK_HI="%x" % hi)
	if match: env["DESC_MATCH"] = match
	r = subprocess.run([HERE + "/desc_probe", str(ws), ACCESSES], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
	t = re.search(r"^(\d+\.\d+)$", r.stderr, re.M)
	return {"ws": ws, "lo": lo, "hi": hi, "rc": r.returncode, "ms": float(t.group(1)) if t else None,
		"out": r.stdout.strip(), "ldcu": (re.search(r"ldcu.*", r.stderr) or [""])[0]}

def main():
	reps, out = int(sys.argv[1]), sys.argv[2]
	masks = [(0, 0)] * 8 + [(1 << b, 0) for b in range(32)] + [(0, 1 << b) for b in range(32)] + [(0xffffffff, 0xffffffff)]
	jobs = [(ws, lo, hi) for ws in SETS for lo, hi in masks for _ in range(reps)]
	random.seed(1)
	random.shuffle(jobs)
	res = [run(*j) for j in jobs]
	# Positive control: the same mechanism on the loop count (c[0x0][0x38c], accesses)
	control = [run(64, 1 << 21, 0, "c[0x0][0x38c]") for _ in range(3)] + [run(64, 0, 0, "c[0x0][0x38c]") for _ in range(3)]
	json.dump({"res": res, "control": control}, open(out, "w"))

if __name__ == "__main__":
	main()
