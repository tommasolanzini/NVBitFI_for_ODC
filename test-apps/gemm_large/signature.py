#!/usr/bin/env python3
"""Output signature of a gemm_large run, see scripts/fp_signature.py. The outputs are
large (4 MB per matrix at N=1024, 16 MB at N=2048): only the wrong elements are kept,
in wrong-core.bin and wrong-tc.bin, and output_*.bin are removed."""
import os, sys
APP_DIR = os.path.dirname(os.path.realpath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(APP_DIR, "..", "..", "scripts"))
from fp_signature import print_signature

print_signature(os.path.join(APP_DIR, "golden-" + open("gemm_bin.txt").read().strip()), ["core", "tc"], compact=True)
