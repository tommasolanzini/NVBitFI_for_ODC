#!/usr/bin/env python3
# Driver of the final campaign, plan in logs/mem-plan-2026-11-04/PLAN.md.
#
#   run_final_block.py <1-4> [--root DIR] [--dry-run] [--smoke N] [--part P] [--gate-dummy 0|1|2]
#
# A block is made of parts. For each part: build with DUMMY=2, run its gate, stop the
# whole block if a gate run is not Masked; build with DUMMY=0, run its cells. The lists
# stay on disk and the campaign scripts skip the runs already in the JSONL, so launching
# the block again continues it, also after a stop.
#
#   --dry-run       prints cells, runs and time. No GPU, nothing written
#   --smoke N       every cell has N runs: to try the chain
#   --part P        only that part
#   --gate-dummy D  gate built with DUMMY=D instead of 2 (the same value written back,
#                   so the write path runs too). 0 = real flips, to see the gate stop
#
# Exit code: 0 done, 2 stopped by a gate or a check. The reason is in <root>/STOPPED.txt.

import argparse, glob, hashlib, json, os, re, shutil, subprocess, sys, time, zlib

GATE_DUMMY = 2

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_RUN_ROOT = ROOT + "/logs/final-2026-11-05"
ARCH = "sm_120"

G_FP32, G_LD, G_PR, G_UNIFORM, G_GP, G_ADDR, G_ADDR_STORE = 1, 2, 3, 9, 12, 13, 14
ADJ_BITS, ADJ_REGS, ADJ_LANES = 4, 5, 6
ALL_APPS = ["gemm", "ptr_chase", "cnn", "shm_stencil", "sfu_math"]
PR_APPS = ["gemm", "ptr_chase", "shm_stencil", "sfu_math"]

# Seconds per run, from PLAN.md: only for the estimate of --dry-run
S_REG = {"gemm": .55, "ptr_chase": .55, "shm_stencil": .55, "sfu_math": .55, "cnn": .57}
S_MEM = {("gemm", "a"): 1.24, ("gemm", "b"): 1.24, ("ptr_chase", "a"): 1.33, ("ptr_chase", "b"): 1.42, ("cnn", "a"): 2.22, ("cnn", "b"): 2.22, ("shm_stencil", "u"): .72, ("shm_stencil", "s"): .72}

MASKED = (1, 2, 3)
SUSPENDED_S = 0.5


# Runs per k: mcu_sizes.py default without size 1, at least 1/6 of the cell at k=4
def split_k(n):
	import mcu_sizes
	table = mcu_sizes.MCU_SIZE_TABLES[mcu_sizes.MCU_SIZE_DEFAULT]
	mcu = sum(table[k] for k in (2, 3, 4))
	k4 = max(round(n * table[4] / mcu), n // 6)
	k3 = round(n * table[3] / mcu)
	return {2: n - k3 - k4, 3: k3, 4: k4}


class Cell:
	def __init__(self, tool, app, n, igid=None, bfm=None, mode=None, mcu=None, k=None, step=None):
		self.tool, self.app, self.n, self.igid, self.bfm, self.mode, self.mcu, self.k, self.step = tool, app, n, igid, bfm, mode, mcu, k, step

	@property
	def name(self):
		if self.tool == "reg":
			return "reg-%s-igid%d-bfm%d" % (self.app, self.igid, self.bfm)
		s = "mem-%s-mode%s" % (self.app, self.mode)
		if self.mcu:
			s += "-%s-k%d" % (self.mcu, self.k)
		if self.step:
			s += "-step%d" % self.step
		return s

	def seed(self, stage):
		return zlib.crc32((stage + self.name).encode()) & 0x7fffffff

	def seconds(self):
		return self.n * (S_REG[self.app] if self.tool == "reg" else S_MEM[(self.app, self.mode)])

	def scaled(self, n):
		c = Cell(self.tool, self.app, self.n, self.igid, self.bfm, self.mode, self.mcu, self.k, self.step)
		c.n = n
		return c


def reg_cells(apps, igid, n):
	return [Cell("reg", a, n, igid=igid, bfm=0) for a in apps]


def reg_mcu(apps, igid, models, n):
	return [Cell("reg", a, c, igid=igid, bfm=m * 10 + k) for a in apps for m in models for k, c in split_k(n).items()]


def reg_mcu_gate(apps, igid, models):
	return [Cell("reg", a, 20, igid=igid, bfm=m * 10 + 3) for a in apps for m in models]


def mem_cells(cells, n):
	return [Cell("mem", a, n, mode=m) for a, m in cells]


def mem_models(k, c):
	return [("adj_bits", k, None, c)] + [("adj_words", k, s, c) for s in (4, 8, 512)]


def mem_mcu(cells, n):
	return [Cell("mem", a, c, mode=m, mcu=mc, k=k, step=st) for a, m in cells for k, nk in split_k(n).items() for mc, k, st, c in mem_models(k, nk)]


def mem_mcu_gate(cells):
	return [Cell("mem", a, c, mode=m, mcu=mc, k=k, step=st) for a, m in cells for mc, k, st, c in mem_models(3, 20)]


GEMM = [("gemm", "a"), ("gemm", "b")]
PTR = [("ptr_chase", "a"), ("ptr_chase", "b")]
SHM = [("shm_stencil", "u"), ("shm_stencil", "s")]
CNN = [("cnn", "a"), ("cnn", "b")]
CNN_GROUPS = [G_FP32, G_LD, G_UNIFORM, G_GP]


def part(name, gate, cells):
	return {"name": name, "gate": gate, "cells": cells}


def blocks():
	return {
		1: [part("memory single bit, gemm ptr_chase shm_stencil", mem_cells(GEMM + PTR + SHM, 100), mem_cells(GEMM + PTR + SHM, 1000)),
			part("registers on cnn", [c for g in CNN_GROUPS for c in reg_cells(["cnn"], g, 100)], [c for g in CNN_GROUPS for c in reg_cells(["cnn"], g, 1000)])],
		2: [part("memory single bit, cnn", mem_cells(CNN, 100), mem_cells(CNN, 1000)),
			part("addresses", reg_cells(ALL_APPS, G_ADDR, 20), reg_cells(["gemm", "ptr_chase", "cnn"], G_ADDR, 1000) + reg_cells(["shm_stencil", "sfu_math"], G_ADDR, 500)),
			part("MCU on GPR", reg_mcu_gate(ALL_APPS, G_GP, [ADJ_BITS, ADJ_REGS, ADJ_LANES]), reg_mcu(ALL_APPS, G_GP, [ADJ_BITS, ADJ_REGS, ADJ_LANES], 600))],
		3: [part("single bit on GPR, ptr_chase shm_stencil sfu_math", reg_cells(["ptr_chase", "shm_stencil", "sfu_math"], G_GP, 100), reg_cells(["ptr_chase", "shm_stencil", "sfu_math"], G_GP, 1000)),
			part("store stratum", reg_cells(["gemm", "ptr_chase", "cnn"], G_ADDR_STORE, 20), reg_cells(["gemm", "ptr_chase", "cnn"], G_ADDR_STORE, 200)),
			part("uniform single bit, gemm", reg_cells(["gemm"], G_UNIFORM, 100), reg_cells(["gemm"], G_UNIFORM, 1000)),
			part("MCU on uniform and predicate registers", reg_mcu_gate(ALL_APPS, G_UNIFORM, [ADJ_BITS, ADJ_REGS]) + reg_mcu_gate(PR_APPS, G_PR, [ADJ_BITS, ADJ_LANES]),
				reg_mcu(ALL_APPS, G_UNIFORM, [ADJ_BITS, ADJ_REGS], 300) + reg_mcu(PR_APPS, G_PR, [ADJ_BITS, ADJ_LANES], 300)),
			part("MCU in memory, gemm ptr_chase", mem_mcu_gate(GEMM + PTR), mem_mcu(GEMM + PTR, 300))],
		4: [part("MCU in memory, cnn shm_stencil", mem_mcu_gate(CNN + SHM), mem_mcu(CNN + SHM, 300))],
	}


class Stop(Exception):
	pass


def sha256_file(path):
	h = hashlib.sha256()
	try:
		with open(path, "rb") as f:
			for chunk in iter(lambda: f.read(1 << 20), b""):
				h.update(chunk)
	except OSError:
		return None
	return h.hexdigest()


def load_jsonl(path):
	with open(path) as f:
		return [json.loads(l) for l in f if l.strip()]


# {kernel: {group: injectable instances}} from a profile of the tool
def parse_profile(path):
	out = {}
	with open(path) as f:
		for line in f:
			if "kernel_name" not in line:
				continue
			k = re.search(r"kernel_name: ([^;]*);", line).group(1)
			out[k] = {g: int(n) for g, n in re.findall(r"\b([a-z_0-9]+): (\d+),", line)}
	return out


# Why a gate run does not pass, register injector. Empty list: it passes.
def gate_problems_reg(records, n, profile, group):
	bad = []
	if len(records) != n:
		bad.append("%d runs of %d" % (len(records), n))
	for r in records:
		tag = "icount %s" % r.get("icount")
		if r["outcome"] not in MASKED:
			bad.append("%s: %s" % (tag, r.get("outcome_str")))
		if not r.get("injected"):
			bad.append(tag + ": not injected")
		for c in r.get("cells", []):
			if c["before"] != c["after"]:
				bad.append(tag + ": a register changed in DUMMY")
		want = profile.get(r.get("kernel"), {}).get(group)
		if want is not None and r.get("group_instrs") not in (want, None) and r["outcome"] in MASKED:
			bad.append("%s: %s instances, profile says %s" % (tag, r.get("group_instrs"), want))
	return bad


# Same, memory injector. New reads are only reported: they are not an outcome.
def gate_problems_mem(records, n):
	bad = []
	if len(records) != n:
		bad.append("%d runs of %d" % (len(records), n))
	for r in records:
		tag = "icount %s" % r.get("icount")
		if r["outcome"] not in MASKED:
			bad.append("%s: %s" % (tag, r.get("outcome_str")))
		if r.get("injected") and r.get("old_word") != r.get("new_word"):
			bad.append(tag + ": the word changed in DUMMY")
		for c in r.get("cells", []):
			if c["old_word"] != c["new_word"]:
				bad.append(tag + ": a word changed in DUMMY")
	return bad


class Driver:
	def __init__(self, a):
		self.a = a
		self.root = os.path.abspath(a.root)
		self.env = dict(os.environ, NVBITFI_HOME=ROOT)
		self.dummy_built = {}

	def log(self, **kw):
		kw["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
		line = json.dumps(kw)
		print(line, flush=True)
		with open(self.root + "/status.jsonl", "a") as f:
			f.write(line + "\n")

	# config/hashes.jsonl: a tool after every build, because DUMMY changes it during a
	# block, and the workloads at the start of a block. None: the file is missing.
	def record_hash(self, path, **kw):
		kw.update(time=time.strftime("%Y-%m-%dT%H:%M:%S"), path=os.path.relpath(path, ROOT), sha256=sha256_file(path))
		with open(self.root + "/config/hashes.jsonl", "a") as f:
			f.write(json.dumps(kw) + "\n")

	def sh(self, cmd, env, logfile):
		with open(logfile, "a") as f:
			f.write("\n=== %s\n" % cmd)
			f.flush()
			return subprocess.call(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)

	def gpu_state(self):
		try:
			return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader"], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
		except OSError:
			return "?"

	# The machine must be as agreed. Before every cell, since a block lasts hours.
	def preflight(self, first=False):
		problems = []
		if open("/proc/sys/kernel/dmesg_restrict").read().strip() != "0":
			problems.append("kernel.dmesg_restrict is not 0")
		online = [open(f).read().strip() for f in glob.glob("/sys/class/power_supply/*/online")]
		if online and "1" not in online:
			problems.append("not on mains power")
		for f in glob.glob("/proc/acpi/button/lid/*/state"):
			if "closed" in open(f).read():
				problems.append("lid closed")
		if first:
			apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
			if apps:
				problems.append("another process uses the GPU: " + apps)
		if problems:
			raise Stop("; ".join(problems))

	def build(self, tool, dummy):
		if tool in self.dummy_built and self.dummy_built[tool] == dummy:
			return
		log = "%s/build.log" % self.root
		extra = " DUMMY=%d" % dummy if tool != "profiler" else ""
		if self.sh("make -C %s/%s ARCH=%s%s" % (ROOT, tool, ARCH, extra), self.env, log):
			raise Stop("build of %s failed, see %s" % (tool, log))
		self.dummy_built[tool] = dummy
		self.record_hash("%s/%s/%s.so" % (ROOT, tool, tool), kind="tool", dummy=dummy)

	def save_config(self):
		d = self.root + "/config"
		os.makedirs(d, exist_ok=True)
		stamp = time.strftime("%Y%m%d-%H%M%S")
		head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
		diff = subprocess.run(["git", "-C", ROOT, "diff", "HEAD", "--", ".", ":(exclude)test-apps/simple_add/stderr.txt"], stdout=subprocess.PIPE, universal_newlines=True).stdout
		untracked = subprocess.run(["git", "-C", ROOT, "ls-files", "--others", "--exclude-standard"], stdout=subprocess.PIPE, universal_newlines=True).stdout
		with open("%s/git-%s.txt" % (d, stamp), "w") as f:
			f.write("head %s\nuntracked:\n%s" % (head, untracked))
		with open("%s/uncommitted-%s.diff" % (d, stamp), "w") as f:
			f.write(diff)
		for u in untracked.split():
			if u.startswith("scripts/"):
				shutil.copy(os.path.join(ROOT, u), d)
		shutil.copy(ROOT + "/logs/mem-plan-2026-11-04/PLAN.md", "%s/PLAN-%s.md" % (d, stamp))
		for app in ALL_APPS:
			self.record_hash("%s/test-apps/%s/%s" % (ROOT, app, app), kind="workload")

	# A folder like NVBITFI_HOME with its own params.py: same scripts, one group and model
	def make_home(self, home, app, igid, bfm, n):
		os.makedirs(home, exist_ok=True)
		for d in ("injector", "profiler", "test-apps", "common"):
			if not os.path.lexists(home + "/" + d):
				os.symlink(ROOT + "/" + d, home + "/" + d)
		shutil.rmtree(home + "/scripts", True)
		shutil.copytree(ROOT + "/scripts", home + "/scripts", ignore=shutil.ignore_patterns("__pycache__"))
		m = {igid: [bfm]} if igid is not None else {}
		with open(home + "/scripts/params.py", "a") as f:
			f.write("\n# ---- this cell\napps = {k: apps[k] for k in %r}\nparse_apps = apps\ninst_value_igid_bfm_map = %r\nparse_inst_value_igid_bfm_map = inst_value_igid_bfm_map\napp_igids = {k: list(inst_value_igid_bfm_map) for k in apps}\nNUM_INJECTIONS = %d\nTHRESHOLD_JOBS = %d\nset_paths()\n" % ([app], m, n, n))

	def reg_profile(self, app):
		path = "%s/profiles/reg-%s.txt" % (self.root, app)
		if not os.path.isfile(path):
			self.build("profiler", None)
			home = "%s/profiles/reg-%s" % (self.root, app)
			self.make_home(home, app, None, None, 1)
			if self.sh("cd %s/scripts && python3 run_profiler.py" % home, dict(self.env, NVBITFI_HOME=home), home + ".log"):
				raise Stop("profile of %s failed" % app)
			shutil.copy("%s/logs/%s/nvbitfi-igprofile.txt" % (home, app), path)
		return path

	def mem_profile(self, app):
		base = "%s/profiles/mem-%s" % (self.root, app)
		if not os.path.isfile(base + "/mem-profile.json"):
			self.build("mem_injector", 1)
			if self.sh("cd %s/scripts && python3 run_mem_campaign.py profile %s %s" % (ROOT, app, base), self.env, base + ".log"):
				raise Stop("memory profile of %s failed" % app)
		return base

	def run_reg(self, c, stage, base):
		profile = self.reg_profile(c.app)
		self.build("injector", self.tool_dummy(stage))
		home = "%s/%s" % (base, c.name)
		self.make_home(home, c.app, c.igid, c.bfm, c.n)
		logs = "%s/logs/%s" % (home, c.app)
		os.makedirs(logs + "/injection-list", exist_ok=True)
		shutil.copy(profile, logs + "/nvbitfi-igprofile.txt")
		env = dict(self.env, NVBITFI_HOME=home, NVBITFI_SEED=str(c.seed(stage)))
		out = home + "/driver.log"
		if not glob.glob(logs + "/injection-list/*.txt") and self.sh("cd %s/scripts && python3 generate_injection_list.py" % home, env, out):
			raise Stop("list of %s failed" % c.name)
		if self.sh("cd %s/scripts && python3 run_injections.py standalone" % home, env, out):
			raise Stop("%s failed, see %s" % (c.name, out))
		files = glob.glob(logs + "/results-*.jsonl")
		return load_jsonl(files[0]) if files else []

	def run_mem(self, c, stage, base):
		profile = self.mem_profile(c.app)
		self.build("mem_injector", self.tool_dummy(stage))
		cell = "%s/%s" % (base, c.name)
		os.makedirs(cell, exist_ok=True)
		for f in ("mem-profile.json", "kernel-timing.json", "exposure.json", "exposure-shared.json"):
			if os.path.isfile(profile + "/" + f):
				shutil.copy(profile + "/" + f, cell)
		out = cell + "/driver.log"
		cmd = "cd %s/scripts && " % ROOT
		if not os.path.isfile("%s/injection-list-mode%s.jsonl" % (cell, c.mode)):
			extra = " --mcu %s --k %d" % (c.mcu, c.k) if c.mcu else ""
			extra += " --step %d" % c.step if c.step else ""
			cmd += "python3 run_mem_campaign.py list %s %s --mode %s -n %d --seed %d%s && " % (c.app, cell, c.mode, c.n, c.seed(stage), extra)
		cmd += "python3 run_mem_campaign.py run %s %s --mode %s" % (c.app, cell, c.mode)
		if self.sh(cmd, self.env, out):
			raise Stop("%s failed, see %s" % (c.name, out))
		path = "%s/results-mode%s.jsonl" % (cell, c.mode)
		return load_jsonl(path) if os.path.isfile(path) else []

	def tool_dummy(self, stage):
		return (self.a.gate_dummy if self.a.gate_dummy is not None else GATE_DUMMY) if stage == "dummy1" else 0

	def run_cell(self, c, stage):
		self.preflight()
		base = "%s/%s" % (self.root, stage)
		os.makedirs(base, exist_ok=True)
		start = time.monotonic()
		records = (self.run_reg if c.tool == "reg" else self.run_mem)(c, stage, base)
		seconds = time.monotonic() - start
		suspended = sum(r.get("suspended_s", 0) > SUSPENDED_S for r in records)
		self.log(stage=stage, cell=c.name, runs=len(records), of=c.n, seconds=round(seconds, 1), suspended=suspended, gpu=self.gpu_state())
		return records

	def check_gate(self, c, records):
		if c.tool == "reg":
			bad = gate_problems_reg(records, c.n, parse_profile("%s/profiles/reg-%s.txt" % (self.root, c.app)), records[0]["group"] if records else "")
		else:
			bad = gate_problems_mem(records, c.n)
		if bad:
			raise Stop("gate of %s is not 100%% Masked: %d problems, first ones: %s" % (c.name, len(bad), "; ".join(bad[:5])))

	def run_part(self, p):
		self.log(part=p["name"], event="start")
		for c in p["gate"]:
			self.check_gate(c, self.run_cell(c, "dummy1"))
		self.log(part=p["name"], event="gate passed", cells=len(p["gate"]))
		for c in p["cells"]:
			records = self.run_cell(c, "dummy0")
			if len(records) != c.n:
				raise Stop("%s has %d runs of %d" % (c.name, len(records), c.n))
		self.log(part=p["name"], event="done")

	def run_block(self, parts):
		os.makedirs(self.root + "/profiles", exist_ok=True)
		self.save_config()
		stopped = self.root + "/STOPPED.txt"
		if os.path.exists(stopped):
			os.rename(stopped, "%s.%s" % (stopped, time.strftime("%Y%m%d-%H%M%S")))
		try:
			self.preflight(first=True)
			for p in parts:
				self.run_part(p)
		except (Stop, OSError, ValueError, KeyError, IndexError) as e: # also a bug of this script: never a silent end
			with open(stopped, "w") as f:
				f.write("%s: %s\n" % (type(e).__name__, e))
			self.log(event="STOPPED", reason=str(e))
			return 2
		finally:
			# Never leave a DUMMY tool behind
			for tool, dummy in list(self.dummy_built.items()):
				if dummy: # None is the profiler, built without DUMMY
					self.build(tool, 0)
		self.log(event="block done")
		return 0


def scale(parts, n):
	return [{"name": p["name"], "gate": [c.scaled(n) for c in p["gate"]], "cells": [c.scaled(n) for c in p["cells"]]} for p in parts]


def dry_run(block, parts):
	total_runs = total_s = 0
	for p in parts:
		gate_s, s = sum(c.seconds() for c in p["gate"]), sum(c.seconds() for c in p["cells"])
		runs = sum(c.n for c in p["cells"])
		total_runs += runs + sum(c.n for c in p["gate"])
		total_s += gate_s + s
		print("part: %s\n  gate: %d cells, %d runs, %.1f min\n  cells: %d, %d runs, %.1f min" % (p["name"], len(p["gate"]), sum(c.n for c in p["gate"]), gate_s / 60, len(p["cells"]), runs, s / 60))
		for c in p["cells"]:
			print("    %-48s %5d runs  seed %d" % (c.name, c.n, c.seed("dummy0")))
	print("block %d: %d runs, %.1f h" % (block, total_runs, total_s / 3600))


def main():
	sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
	ap = argparse.ArgumentParser()
	ap.add_argument("block", type=int, choices=[1, 2, 3, 4])
	ap.add_argument("--root", default=DEFAULT_RUN_ROOT)
	ap.add_argument("--dry-run", action="store_true")
	ap.add_argument("--smoke", type=int)
	ap.add_argument("--part", type=int)
	ap.add_argument("--gate-dummy", type=int, choices=[0, 1, 2])
	a = ap.parse_args()
	parts = blocks()[a.block]
	if a.part is not None:
		parts = [parts[a.part]]
	if a.smoke:
		parts = scale(parts, a.smoke)
	if a.dry_run:
		dry_run(a.block, parts)
		return 0
	return Driver(a).run_block(parts)


if __name__ == "__main__":
	sys.exit(main())
