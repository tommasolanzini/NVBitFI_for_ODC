"""Output signature of a fault injection run: compare fp32 result matrices with
their golden copies. Used by the signature.py script of each workload.

Prints one JSON object with the features of each result matrix:
  status             ok, missing if the run did not write the file (crash),
                     bad_size if it has the wrong size
  wrong_elements     elements that differ bit by bit from the golden matrix
  pattern            none, single, row, column, block or scattered
  max_rel_error      largest |x - golden| / |golden| over finite wrong elements
                     (absolute error when golden is 0), null if there are none
  nonfinite_elements wrong elements that became NaN or infinite
  bit_class          wrong elements by their most significant differing fp32
                     bit: sign (31), exponent (30-23), mantissa (22-0)

Uses only the standard library: it runs once per injection.
"""
import json, math, os, struct
from array import array

MIN_BLOCK_FILL = 0.5 # wrong elements / bounding box area to call it a block

def classify_pattern(wrong, n):
	"""wrong: flat row-major indices of the wrong elements of an n x n matrix."""
	if not wrong:
		return "none"
	if len(wrong) == 1:
		return "single"
	rows = [i // n for i in wrong]
	cols = [i % n for i in wrong]
	if min(rows) == max(rows):
		return "row"
	if min(cols) == max(cols):
		return "column"
	area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
	return "block" if len(wrong) / area >= MIN_BLOCK_FILL else "scattered"

def bit_class(xor):
	if xor >> 31:
		return "sign"
	return "exponent" if xor >> 23 else "mantissa"

def compare(out_bytes, golden_bytes):
	"""Features of one matrix; out_bytes is None if the output does not exist."""
	if out_bytes is None:
		return {"status": "missing"}
	if len(out_bytes) != len(golden_bytes):
		return {"status": "bad_size"}
	out_bits = array("I", out_bytes)
	golden_bits = array("I", golden_bytes)
	n = math.isqrt(len(golden_bits))

	wrong = []
	classes = {"sign": 0, "exponent": 0, "mantissa": 0}
	max_rel = None
	nonfinite = 0
	for i, (o, g) in enumerate(zip(out_bits, golden_bits)):
		if o == g:
			continue
		wrong.append(i)
		classes[bit_class(o ^ g)] += 1
		x, y = struct.unpack("<2f", struct.pack("<2I", o, g))
		if not math.isfinite(x):
			nonfinite += 1
			continue
		rel = abs(x - y) / abs(y) if y != 0 else abs(x - y)
		max_rel = rel if max_rel is None else max(max_rel, rel)
	return {"status": "ok", "wrong_elements": len(wrong), "pattern": classify_pattern(wrong, n),
		"max_rel_error": max_rel, "nonfinite_elements": nonfinite, "bit_class": classes}

def read_or_none(fname):
	if not os.path.isfile(fname):
		return None
	with open(fname, "rb") as f:
		return f.read()

def print_signature(app_dir, names):
	"""Compare output_<name>.bin (current directory) with app_dir/golden_output_<name>.bin."""
	sig = {}
	for name in names:
		with open(os.path.join(app_dir, "golden_output_%s.bin" % name), "rb") as f:
			sig[name] = compare(read_or_none("output_%s.bin" % name), f.read())
	print(json.dumps(sig, allow_nan=False))
