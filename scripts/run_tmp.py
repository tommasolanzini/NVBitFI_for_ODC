import glob, os, shutil, tempfile

# Temporary files of one run, removed when it ends (crash and timeout included).
# NVBit 1.8 ignores TMPDIR: it always makes /tmp/nvbit_tool_tmpdir.XXXXXX (about
# 4 MB) and never removes it, so 20000 runs filled the disk on 2026-11-06. Runs are
# sequential, so the NVBit entries that appear during the run are this run's. The
# run also gets its own TMPDIR, for whatever honours it.
NVBIT_TMP_GLOBS = ["/tmp/nvbit_tool_tmpdir.*", "/tmp/nvbit_code_*", "/tmp/nvbit_sass_*"]

def nvbit_tmp_entries():
	return {e for g in NVBIT_TMP_GLOBS for e in glob.glob(g) if os.lstat(e).st_uid == os.getuid()}

def remove_tmp_entry(e):
	if os.path.isdir(e) and not os.path.islink(e):
		shutil.rmtree(e, True)
	else:
		try:
			os.unlink(e)
		except OSError:
			pass

class RunTmp:
	def __enter__(self):
		self.before = nvbit_tmp_entries()
		self.old = os.environ.get("TMPDIR")
		self.dir = tempfile.mkdtemp(prefix="nvbitfi-run-")
		os.environ["TMPDIR"] = self.dir
		return self.dir

	def __exit__(self, *exc):
		if self.old is None:
			os.environ.pop("TMPDIR", None)
		else:
			os.environ["TMPDIR"] = self.old
		remove_tmp_entry(self.dir)
		for e in nvbit_tmp_entries() - self.before:
			remove_tmp_entry(e)
		return False

# Bytes free on the file systems the runs write to
def free_bytes(paths=("/tmp",)):
	return min(shutil.disk_usage(d).free for d in paths)
