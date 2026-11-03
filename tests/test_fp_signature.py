#!/usr/bin/env python3
"""Offline test of scripts/fp_signature.py (no GPU needed)."""
import math, os, struct, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from fp_signature import classify_pattern, compare

N = 8
ONE = 0x3f800000
golden = [ONE] * (N * N)

def pack(bits):
	return struct.pack("<%dI" % len(bits), *bits)

def flip(*changes):
	out = list(golden)
	for idx, val in changes:
		out[idx] = val
	return compare(pack(out), pack(golden))

passed = failed = 0
def check(name, got, expected):
	global passed, failed
	if got == expected:
		passed += 1
	else:
		failed += 1
		print("FAIL %s: got %r, expected %r" % (name, got, expected))

check("none", classify_pattern([], N), "none")
check("single", classify_pattern([9], N), "single")
check("row", classify_pattern([8, 10, 15], N), "row")
check("column", classify_pattern([1, 9, 57], N), "column")
check("block", classify_pattern([0, 1, 8, 9], N), "block")
check("sparse block", classify_pattern([0, 1, 8], N), "block")
check("scattered", classify_pattern([0, 63], N), "scattered")

r = compare(pack(golden), pack(golden))
check("equal", (r["status"], r["wrong_elements"], r["pattern"], r["max_rel_error"]), ("ok", 0, "none", None))

r = flip((5, ONE ^ (1 << 31))) # 1.0 -> -1.0
check("sign", r["bit_class"], {"sign": 1, "exponent": 0, "mantissa": 0})
check("sign error", r["max_rel_error"], 2.0)
r = flip((5, ONE ^ (1 << 23))) # 1.0 -> 0.5
check("exponent", r["bit_class"], {"sign": 0, "exponent": 1, "mantissa": 0})
check("exponent error", r["max_rel_error"], 0.5)
r = flip((5, ONE ^ 1))
check("mantissa", r["bit_class"], {"sign": 0, "exponent": 0, "mantissa": 1})
check("mantissa error", math.isclose(r["max_rel_error"], 2 ** -23), True)

r = flip((5, 0x7fc00000), (6, 0x7f800000)) # NaN, +inf
check("nonfinite", (r["nonfinite_elements"], r["max_rel_error"], r["wrong_elements"]), (2, None, 2))

zero = [0] * (N * N)
wrong_zero = list(zero); wrong_zero[0] = ONE
check("golden 0", compare(pack(wrong_zero), pack(zero))["max_rel_error"], 1.0)

check("missing", compare(None, pack(golden)), {"status": "missing"})
check("bad size", compare(pack(golden[:-1]), pack(golden)), {"status": "bad_size"})

print("%d/%d signature cases passed" % (passed, passed + failed))
sys.exit(1 if failed else 0)
