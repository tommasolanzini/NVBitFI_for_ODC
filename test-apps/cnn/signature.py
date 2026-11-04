#!/usr/bin/env python3
"""Output signature of a CNN run, see scripts/fp_signature.py."""
import os, sys
APP_DIR = os.path.dirname(os.path.realpath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(APP_DIR, "..", "..", "scripts"))
from fp_signature import print_signature

print_signature(APP_DIR, ["logits"])
