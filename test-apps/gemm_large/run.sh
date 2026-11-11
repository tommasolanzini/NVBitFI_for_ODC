#!/bin/bash
# The binary: first argument (params.py passes it), else GEMM_BIN, else gemm_large (N=4096).
# Its name goes in gemm_bin.txt, read by sdc_check.sh and signature.py for the golden.
B=${1:-${GEMM_BIN:-gemm_large}}
echo ${B} > gemm_bin.txt
eval ${PRELOAD_FLAG} ${BIN_DIR}/${B} > stdout.txt 2> stderr.txt
