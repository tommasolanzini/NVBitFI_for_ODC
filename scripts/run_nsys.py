#!/usr/bin/env python3
# Timeline of a run without a tool, with Nsight Systems (nsys): kernels, copies and the
# CUDA API calls (cudaMalloc, cudaFree...), for the intervals between kernels.
#
#   run_nsys.py <output dir> [--nsys PATH] [--runs R] [--only w1,w2]
#
# Needs nsys 2025.3 or later (CUDA 13.0): nsys 2025.1.3 of CUDA 12.9 collects nothing on
# this GPU (driver CUDA 13.0 not supported, CUPTI errors). No root needed (CUDA trace only,
# no CPU sampling).
#
# Per workload, in <output dir>/<workload>/: run-<i>.nsys-rep, run-<i>_cuda_gpu_trace.csv,
# run-<i>_cuda_api_trace.csv; and <output dir>/nsys.json:
#   workloads -> runs -> {"gpu": [{name, start_ns, end_ns, kind}], "api": [{name, start_ns, end_ns}]}
# Times are relative to the first CUDA API call of the run.

import argparse, csv, glob, json, os, shutil, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ncu import WORKLOADS  # noqa: E402


def find_nsys():
	found = sorted(glob.glob("/opt/nvidia/nsight-systems/2025.[3-9]*/bin/nsys") + glob.glob("/opt/nvidia/nsight-systems/202[6-9]*/bin/nsys"))
	return found[-1] if found else shutil.which("nsys")


def read_csv(path):
	if not os.path.isfile(path):
		return []
	with open(path) as f:
		return list(csv.DictReader(f))


def num(v):
	try:
		return int(float(v))
	except (TypeError, ValueError):
		return None


def parse(prefix):
	gpu, api = [], []
	for r in read_csv(prefix + "_cuda_gpu_trace.csv"):
		s, d = num(r.get("Start (ns)")), num(r.get("Duration (ns)"))
		if s is None or d is None:
			continue
		name = r.get("Name", "")
		kind = "memcpy" if "memcpy" in name.lower() else "memset" if "memset" in name.lower() else "kernel"
		gpu.append({"name": name, "kind": kind, "start_ns": s, "end_ns": s + d, "bytes_mb": r.get("Bytes (MB)")})
	for r in read_csv(prefix + "_cuda_api_trace.csv"):
		s, d = num(r.get("Start (ns)")), num(r.get("Duration (ns)"))
		if s is None or d is None:
			continue
		api.append({"name": r.get("Name", ""), "start_ns": s, "end_ns": s + d})
	t0 = min([e["start_ns"] for e in api] + [e["start_ns"] for e in gpu]) if (api or gpu) else 0
	for e in gpu + api:
		e["start_ns"] -= t0
		e["end_ns"] -= t0
	return {"gpu": sorted(gpu, key=lambda e: e["start_ns"]), "api": sorted(api, key=lambda e: e["start_ns"])}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("out")
	ap.add_argument("--nsys", default=find_nsys())
	ap.add_argument("--runs", type=int, default=5)
	ap.add_argument("--only")
	a = ap.parse_args()
	if not a.nsys:
		sys.exit("nsys not found: pass --nsys")
	out = os.path.abspath(a.out)
	os.makedirs(out, exist_ok=True)
	ver = subprocess.run([a.nsys, "--version"], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
	names = a.only.split(",") if a.only else ["gemm", "cnn", "ptr_chase", "shm_stencil", "sfu_math"]
	env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
	env["TMPDIR"] = out + "/tmp" # /tmp/nvidia can be left owned by root by a sudo ncu run
	os.makedirs(env["TMPDIR"], exist_ok=True)
	result = {"nsys": a.nsys, "nsys_version": ver, "runs": a.runs, "workloads": {}}
	for w in names:
		d = os.path.join(out, w)
		os.makedirs(d, exist_ok=True)
		log = open(d + "/nsys.log", "a")
		runs = []
		for i in range(a.runs):
			rep = "%s/run-%d" % (d, i)
			cmd = [a.nsys, "profile", "-t", "cuda", "--sample=none", "--cpuctxsw=none", "-f", "true", "-o", rep, WORKLOADS[w]]
			log.write("\n=== %s\n" % " ".join(cmd))
			log.flush()
			if subprocess.call(cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT):
				sys.exit("%s: nsys profile failed, see %s/nsys.log" % (w, d))
			cmd = [a.nsys, "stats", "--force-overwrite", "true", "--format", "csv", "-r", "cuda_gpu_trace,cuda_api_trace", "-o", rep, rep + ".nsys-rep"]
			log.write("\n=== %s\n" % " ".join(cmd))
			log.flush()
			subprocess.call(cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
			runs.append(parse(rep))
		result["workloads"][w] = {"binary": os.path.relpath(WORKLOADS[w], os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs": runs}
		k = [e for e in runs[-1]["gpu"] if e["kind"] == "kernel"]
		print(w, "kernels", len(k), "gpu events", len(runs[-1]["gpu"]), "api calls", len(runs[-1]["api"]), flush=True)
	json.dump(result, open(out + "/nsys.json", "w"), indent=1)


if __name__ == "__main__":
	main()
