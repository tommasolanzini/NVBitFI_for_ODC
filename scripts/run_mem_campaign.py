#!/usr/bin/env python3
# Campaign driver of the memory-data injector (mem_injector/). Results and
# their configuration go in one directory, as for the other campaigns.
#
#   run_mem_campaign.py profile <app> <dir>
#   run_mem_campaign.py list <app> <dir> --mode a|b -n N --seed S
#   run_mem_campaign.py run <app> <dir> --mode a|b
#   run_mem_campaign.py one <app> <dir> <file with one target as JSON>
#
# Reuses the classification of run_one_injection.py, so that the outcomes mean
# the same as in the register campaigns. Build first:
#   make -C mem_injector ARCH=sm_120 DUMMY=0

import argparse, datetime, json, os, shutil, signal, subprocess, sys, time

import params as p
import common_functions as cf
import run_one_injection as roi
import mem_plan as mp

TOOL = p.NVBITFI_HOME + "/mem_injector/mem_injector.so"


def prepare_env(app):
	cf.set_env(app, False)
	os.environ["PRELOAD_FLAG"] = "LD_PRELOAD=" + TOOL


def run_workload(app, run_dir):
	cmd = p.script_dir[app] + "/" + p.run_script + " " + p.apps[app][4]
	return subprocess.Popen(cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid, cwd=run_dir)


def phase_run(app, base, phase):
	run_dir = base + "/profile-" + phase
	shutil.rmtree(run_dir, True)
	os.makedirs(run_dir)
	os.environ["MEMINJ_PHASE"] = phase
	os.environ["MEMINJ_EVENTS"] = run_dir + "/events.jsonl"
	rc = run_workload(app, run_dir).wait()
	del os.environ["MEMINJ_PHASE"], os.environ["MEMINJ_EVENTS"]
	if rc != 0:
		sys.exit("%s run failed with exit code %d" % (phase, rc))
	return mp.load_events(run_dir + "/events.jsonl")


def cmd_profile(a, base):
	prepare_env(a.app)
	events = mp.merge_profiles(phase_run(a.app, base, "timing"), phase_run(a.app, base, "count"))
	json.dump(events, open(base + "/mem-profile.json", "w"), indent=1)
	x = mp.exposure(events)
	json.dump(x, open(base + "/exposure.json", "w"), indent=1)
	print(json.dumps(x, indent=1))
	for b in mp.buffers_of(events).values():
		print("buffer", b)


def list_path(base, mode):
	return base + "/injection-list-mode" + mode + ".jsonl"


def cmd_list(a, base):
	events = json.load(open(base + "/mem-profile.json"))
	with open(list_path(base, a.mode), "w") as f:
		for i, t in enumerate(mp.sample_targets(events, a.mode, a.n, a.seed), 1):
			t["icount"] = i
			t["seed"] = a.seed
			f.write(json.dumps(t) + "\n")


def to_int(v, default=-1):
	try:
		return int(v, 0)
	except (TypeError, ValueError):
		return default


def record_from_log(log):
	return {
		"structure": "global", "timing_mode": log.get("mode", ""),
		"buffer": to_int(log.get("bufferId")), "buffer_size": to_int(log.get("bufferSize")),
		"offset": to_int(log.get("offset")), "bit": to_int(log.get("bit")),
		"old_word": log.get("oldWord", ""), "new_word": log.get("newWord", ""),
		"target_event": to_int(log.get("targetEvent")), "flip_kernel": to_int(log.get("flipKernel")),
		"dyn_instr": to_int(log.get("dynInstr")), "pc": log.get("pcOffset", ""),
		"tid": to_int(log.get("tid")), "cta": to_int(log.get("ctaID")), "sm": to_int(log.get("smID")),
		"warp_slot": to_int(log.get("warpID")), "lane": to_int(log.get("laneID")),
		"reads_old": to_int(log.get("readsOld"), 0), "reads_new": to_int(log.get("readsNew"), 0),
		"loads": to_int(log.get("loads")), "writes_after": to_int(log.get("writesAfter"), 0),
		"sms_old": to_int(log.get("smOld"), 0), "sms_new": to_int(log.get("smNew"), 0),
		"sms_loads": to_int(log.get("smLoads"), 0), "first_access": to_int(log.get("firstAccess"), 0),
		"host_reads": to_int(log.get("hostReads"), 0), "host_writes": to_int(log.get("hostWrites"), 0),
		"read_kernels": mp.kernels_from_mask(to_int(log.get("readKernels"), 0)),
		"injected": bool(log.get("errorInjected", False)) and "newWord" in log,
	}


def run_target(a, base, t, out_name):
	icount = str(t.get("icount", 0))
	run_dir = "%s/runs-mode%s/icount%s" % (base, t["mode"], icount)
	shutil.rmtree(run_dir, True)
	os.makedirs(run_dir)
	open(run_dir + "/" + p.injection_seeds, "w").write(mp.target_to_info(t))

	# the classification of run_one_injection.py works in the current directory
	roi.stdout_fname = run_dir + "/" + p.stdout_file
	roi.stderr_fname = run_dir + "/" + p.stderr_file
	roi.new_directory = run_dir
	start = datetime.datetime.now()
	dmesg_before = roi.cmdline("dmesg")
	cwd = os.getcwd()
	os.chdir(run_dir)
	pr = run_workload(a.app, run_dir)
	timeout_flag, retcode = roi.is_timeout(a.app, pr)
	dmesg_delta = roi.get_dmesg_delta(dmesg_before, roi.cmdline("dmesg")).replace("\n", "; ").replace(":", "-")
	log = roi.read_inj_log()
	rec = dict(t)
	rec.update(record_from_log(log))
	if timeout_flag:
		cat = p.TIMEOUT
	else:
		cat = roi.classify_injection(a.app, "mem", t.get("kernel_name", ""), t.get("kernel", -1), t.get("inst", -1), 0, t["bit"], retcode, dmesg_delta)
	rec.update({"app": a.app, "outcome": cat, "outcome_str": p.CAT_STR[cat - 1],
		"exit_code": retcode, "xids": roi.get_xids(dmesg_delta), "dmesg": dmesg_delta,
		"cuda_error": roi.get_cuda_error(open(p.inj_run_log).read() if os.path.isfile(p.inj_run_log) else "") or roi.get_cuda_error(open(p.stdout_file).read() if os.path.isfile(p.stdout_file) else ""),
		"signature": roi.get_signature(a.app),
		"runtime_s": (datetime.datetime.now() - start).total_seconds()})
	rec["masked_subclass"] = mp.masked_subclass(rec) if cat in (p.MASKED_OTHER,) else None
	os.chdir(cwd)
	with open(base + "/" + out_name, "a") as f:
		f.write(json.dumps(rec) + "\n")
	print("icount=%s mode=%s buffer=%d offset=%d bit=%d outcome=%s reads old/new/all=%d/%d/%d host=%d first=%d subclass=%s" % (icount, t["mode"], t["buffer"], t["offset"], t["bit"],
		p.CAT_STR[cat - 1], rec["reads_old"], rec["reads_new"], rec["loads"], rec["host_reads"], rec["first_access"], rec["masked_subclass"]))
	time.sleep(0.2)


def cmd_run(a, base):
	prepare_env(a.app)
	out = "results-mode%s.jsonl" % a.mode
	done = set()
	if os.path.isfile(base + "/" + out):
		done = {json.loads(line)["icount"] for line in open(base + "/" + out)}
	for line in open(list_path(base, a.mode)):
		t = json.loads(line)
		if t["icount"] not in done:
			run_target(a, base, t, out)


def cmd_one(a, base):
	prepare_env(a.app)
	run_target(a, base, json.load(open(a.target)), "results-manual.jsonl")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("command", choices=["profile", "list", "run", "one"])
	ap.add_argument("app")
	ap.add_argument("dir")
	ap.add_argument("target", nargs="?")
	ap.add_argument("--mode", choices=["a", "b"])
	ap.add_argument("-n", type=int, default=100)
	ap.add_argument("--seed", type=int, default=1)
	a = ap.parse_args()
	base = os.path.abspath(a.dir)
	os.makedirs(base, exist_ok=True)
	{"profile": cmd_profile, "list": cmd_list, "run": cmd_run, "one": cmd_one}[a.command](a, base)


if __name__ == "__main__":
	main()
