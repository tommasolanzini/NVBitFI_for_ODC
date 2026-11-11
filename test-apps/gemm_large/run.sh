#!/bin/bash
# GEMM_BIN picks the size: gemm_large (N=4096, default) or gemm_n<N>
eval ${PRELOAD_FLAG} ${BIN_DIR}/${GEMM_BIN:-gemm_large} > stdout.txt 2> stderr.txt
