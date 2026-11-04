#!/usr/bin/env python3
# Upper and lower bound of the time weights, no GPU needed. Upper: durations as
# measured. Lower: without the host time of an empty kernel. See "The first launch of
# a kernel" in mem_injector/README.md.
#
#   mem_time_bounds.py <profile dir, global> <profile dir, shared> <results of mode u> <floor us> <out.json>
#
# A profile dir holds mem-profile.json and kernel-timing.json.

import json, sys

import mem_plan as mp


def load(base):
	return json.load(open(base + "/mem-profile.json")), json.load(open(base + "/kernel-timing.json"))


def main():
	global_dir, shared_dir, results_path, floor_us, out_path = sys.argv[1:6]
	floor_us = float(floor_us)
	global_events, global_timing = load(global_dir)
	shared_events, shared_timing = load(shared_dir)
	results = [json.loads(line) for line in open(results_path)]
	out = {"floor_us": floor_us}
	for name, floor in (("upper_as_measured", 0.0), ("lower_without_floor", floor_us)):
		g = mp.apply_kernel_timing(global_events, mp.subtract_floor(global_timing, floor))
		s = mp.apply_kernel_timing(shared_events, mp.subtract_floor(shared_timing, floor))
		x = mp.shared_exposure(s)
		out[name] = {"mode_b_shares": mp.mode_b_shares(g), "shared_exposure": x, "shared_avf": mp.shared_avf(results, x)}
	json.dump(out, open(out_path, "w"), indent=1)
	print(json.dumps(out, indent=1))


main()
