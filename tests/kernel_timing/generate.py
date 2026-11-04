#!/usr/bin/env python3
# Copies a workload source and wraps every kernel launch in REPEAT, see repeat.h.
import re, sys

LAUNCH = re.compile(r"^(\s*)(\w+)<<<(.*)>>>\((.*)\);$", re.M)

src = open(sys.argv[1]).read()
out, n = LAUNCH.subn(r"\1REPEAT(\2, \2<<<\3>>>(\4));", src)
if n == 0:
	sys.exit("no kernel launch found in " + sys.argv[1])
open(sys.argv[2], "w").write('#include "repeat.h"\n' + out)
