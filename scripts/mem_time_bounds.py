#!/usr/bin/env python3
# Upper and lower bound of the time weights, no GPU needed. Upper: durations as
# measured. Lower: without the host time of an empty kernel. See "The first launch of
# a kernel" in mem_injector/README.md.
#
#   mem_time_bounds.py <profile dir, global> <profile dir, shared> <results of mode u> <floor us> <out.json> [ncu.json]
#
# A profile dir holds mem-profile.json and kernel-timing.json, and is named mem-<workload>...
# With ncu.json (scripts/run_ncu.py) a third case, gpu_ncu: durations measured on the GPU
# side, without launch cost, which replace the two bounds.

import json, os, re, sys

import mem_plan as mp


def load(base):
	return json.load(open(base + "/mem-profile.json")), json.load(open(base + "/kernel-timing.json"))


# Workload of a profile dir, from its name: mem-<workload> or mem-<workload>-mode...
def workload_of(d):
	return re.match(r"mem-([a-z_]+?)(?:-mode.*)?$", os.path.basename(os.path.normpath(d))).group(1)


def main():
	global_dir, shared_dir, results_path, floor_us, out_path = sys.argv[1:6]
	ncu_json = sys.argv[6] if len(sys.argv) > 6 else None
	floor_us = float(floor_us)
	global_events, global_timing = load(global_dir)
	shared_events, shared_timing = load(shared_dir)
	results = [json.loads(line) for line in open(results_path)]
	out = {"floor_us": floor_us}
	cases = [("upper_as_measured", global_timing, shared_timing, 0.0), ("lower_without_floor", global_timing, shared_timing, floor_us)]
	if ncu_json:
		cases.append(("gpu_ncu", mp.kernel_timing_from_ncu(ncu_json, workload_of(global_dir)), mp.kernel_timing_from_ncu(ncu_json, workload_of(shared_dir)), 0.0))
	for name, gt, st, floor in cases:
		g = mp.apply_kernel_timing(global_events, mp.subtract_floor(gt, floor))
		s = mp.apply_kernel_timing(shared_events, mp.subtract_floor(st, floor))
		x = mp.shared_exposure(s)
		out[name] = {"mode_b_shares": mp.mode_b_shares(g), "shared_exposure": x, "shared_avf": mp.shared_avf(results, x)}
	json.dump(out, open(out_path, "w"), indent=1)
	print(json.dumps(out, indent=1))


main()
