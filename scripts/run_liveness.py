#!/usr/bin/env python3
# Runs a workload under the liveness tool and checks that its output stays the golden
# one: the tool must not disturb it.
#
#   run_liveness.py <app> <output dir>
#
# Writes <output dir>/liveness.jsonl and run.txt. See "Register liveness" in README.md.

import filecmp, glob, os, subprocess, sys, time

from run_tmp import RunTmp

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
	app, out = sys.argv[1], os.path.abspath(sys.argv[2])
	os.makedirs(out, exist_ok=True)
	app_dir = os.path.join(ROOT, "test-apps", app)
	start = time.monotonic()
	with RunTmp(): # NVBit temporary files removed also after a crash; env built inside to get the run's TMPDIR
		env = dict(os.environ, NOBANNER="1", ACK_CTX_INIT_LIMITATION="1", BIN_DIR=app_dir, BIN_PATH=app_dir, APP_DIR=app_dir,
			PRELOAD_FLAG="LD_PRELOAD=" + os.path.join(ROOT, "liveness", "liveness.so"), LIVE_OUT=os.path.join(out, "liveness.jsonl"))
		with open(out + "/stdout.txt", "w") as so, open(out + "/stderr.txt", "w") as se:
			rc = subprocess.call([os.path.join(app_dir, "run.sh")], cwd=out, env=env, stdout=so, stderr=se)
	seconds = time.monotonic() - start
	same = {}
	for golden in glob.glob(app_dir + "/golden_output*"):
		name = os.path.basename(golden)[len("golden_"):]
		path = os.path.join(out, name)
		same[name] = os.path.isfile(path) and filecmp.cmp(golden, path, shallow=False)
	with open(out + "/run.txt", "w") as f:
		f.write("exit %d, %.1f s, same as golden: %s\n" % (rc, seconds, same))
	print(app, "exit", rc, "%.1f s" % seconds, same)
	sys.exit(0 if rc == 0 and all(same.values()) and same else 1)


if __name__ == "__main__":
	main()
