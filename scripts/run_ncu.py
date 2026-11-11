#!/usr/bin/env python3
# Kernel durations, occupancy and caches on the GPU side, with Nsight Compute (ncu), no NVBit tool.
#
#   sudo python3 run_ncu.py <output dir> [--ncu PATH] [--runs R] [--only w1,w2]
#
# Needs ncu 2025.3 or later (CUDA 13.0): ncu 2025.2.1 of CUDA 12.9 does not list the GB207
# chip and fails with "LibraryNotLoaded". Needs access to the performance counters
# (RmProfilingAdminOnly=1 here), so root: the files are given back to the user of sudo.
#
# Per workload, in <output dir>/<workload>/:
#   time-<i>.ncu-rep, time.csv  gpu__time_duration only (one pass, no replay), caches and
#                               clocks left as in a normal run (--cache-control none,
#                               --clock-control none), R runs
#   metrics.ncu-rep, metrics.csv occupancy, L1, L2, DRAM. --replay-mode application: the whole
#                               application runs once per pass, so every pass sees the caches
#                               as a normal run does (kernel replay would flush or warm them)
# and <output dir>/ncu.json with everything per workload and kernel.

import argparse, csv, glob, io, json, os, shutil, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
APPS = ROOT + "/test-apps"

# name: binary. All write their outputs in the current directory.
WORKLOADS = {
	"gemm": APPS + "/gemm/gemm",
	"cnn": APPS + "/cnn/cnn",
	"ptr_chase": APPS + "/ptr_chase/ptr_chase",
	"shm_stencil": APPS + "/shm_stencil/shm_stencil",
	"sfu_math": APPS + "/sfu_math/sfu_math",
	"cnn_large": APPS + "/cnn_large/cnn_large",
	"gemm_n256": APPS + "/gemm_large/gemm_n256",
	"gemm_n512": APPS + "/gemm_large/gemm_n512",
	"gemm_n1024": APPS + "/gemm_large/gemm_n1024",
	"gemm_n2048": APPS + "/gemm_large/gemm_n2048",
	"gemm_large": APPS + "/gemm_large/gemm_large",
}

METRICS = [
	"gpu__time_duration.sum", "sm__cycles_elapsed.avg", "sm__cycles_active.avg",
	"sm__warps_active.avg.per_cycle_active", "sm__warps_active.avg.pct_of_peak_sustained_active",
	"l1tex__t_bytes.sum", "l1tex__t_sectors.sum", "l1tex__t_sector_hit_rate.pct",
	"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
	"l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
	"l1tex__m_xbar2l1tex_read_bytes.sum", "l1tex__m_l1tex2xbar_write_bytes.sum",
	"lts__t_bytes.sum", "lts__t_sectors.sum", "lts__t_sector_hit_rate.pct", "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum",
	"dram__bytes_op_read.sum", "dram__bytes_op_write.sum",
]
SECTIONS = ["LaunchStats", "Occupancy"]


def find_ncu():
	found = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
	return found[-1] if found else shutil.which("ncu")


def run(cmd, cwd, log):
	with open(log, "a") as f:
		f.write("\n=== %s\n" % " ".join(cmd))
		f.flush()
		return subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)


def raw_csv(ncu, rep):
	out = subprocess.run([ncu, "--import", rep, "--csv", "--page", "raw"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True).stdout
	rows = list(csv.reader(io.StringIO(out)))
	if len(rows) < 2:
		return [], out
	head = rows[0]
	return [dict(zip(head, r)) for r in rows[2:]], out  # row 1: units


def num(v):
	try:
		return float(v.replace(",", ""))
	except (AttributeError, ValueError):
		return v


def give_back(path):
	uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
	if uid is None:
		return
	for d, dirs, files in os.walk(path):
		for x in [d] + [os.path.join(d, f) for f in files]:
			os.chown(x, int(uid), int(gid))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("out")
	ap.add_argument("--ncu", default=find_ncu())
	ap.add_argument("--runs", type=int, default=5)
	ap.add_argument("--only")
	a = ap.parse_args()
	if not a.ncu:
		sys.exit("ncu not found: pass --ncu")
	out = os.path.abspath(a.out)
	os.makedirs(out, exist_ok=True)
	ver = subprocess.run([a.ncu, "--version"], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip().splitlines()[-1]
	names = a.only.split(",") if a.only else list(WORKLOADS)
	env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
	result = {"ncu": a.ncu, "ncu_version": ver, "runs": a.runs, "metrics": METRICS, "sections": SECTIONS, "workloads": {}}
	try:
		for w in names:
			d = os.path.join(out, w)
			os.makedirs(d, exist_ok=True)
			log = d + "/ncu.log"
			binary = WORKLOADS[w]
			times = {}
			for i in range(a.runs):
				rep = "%s/time-%d" % (d, i)
				rc = run([a.ncu, "-f", "-o", rep, "--metrics", "gpu__time_duration.sum", "--cache-control", "none", "--clock-control", "none", binary], d, log)
				if rc != 0:
					sys.exit("%s: ncu failed (exit %d), see %s" % (w, rc, log))
				rows, text = raw_csv(a.ncu, rep + ".ncu-rep")
				open("%s/time-%d.csv" % (d, i), "w").write(text)
				for r in rows:
					times.setdefault(r["Kernel Name"].split("(")[0], []).append(num(r["gpu__time_duration.sum"]))
			rep = d + "/metrics"
			cmd = [a.ncu, "-f", "-o", rep, "--replay-mode", "application", "--cache-control", "none", "--clock-control", "none", "--metrics", ",".join(METRICS)]
			for s in SECTIONS:
				cmd += ["--section", s]
			rc = run(cmd + [binary], d, log)
			if rc != 0:
				sys.exit("%s: ncu failed (exit %d), see %s" % (w, rc, log))
			rows, text = raw_csv(a.ncu, rep + ".ncu-rep")
			open(d + "/metrics.csv", "w").write(text)
			kernels = {}
			for r in rows:
				k = r["Kernel Name"].split("(")[0]
				kernels[k] = {m: num(v) for m, v in r.items() if "__" in m}
				kernels[k]["time_ns_runs"] = times.get(k, [])
			result["workloads"][w] = {"binary": os.path.relpath(binary, ROOT), "kernels": kernels}
			print(w, ", ".join("%s %.2f us" % (k, sum(v["time_ns_runs"]) / len(v["time_ns_runs"]) / 1000) for k, v in kernels.items() if v["time_ns_runs"]), flush=True)
		json.dump(result, open(out + "/ncu.json", "w"), indent=1)
	finally:
		give_back(out)


if __name__ == "__main__":
	main()
