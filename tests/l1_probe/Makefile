# Experiment on L1 copies and atomics executed in L2 (see README in this directory).
ARCH ?= sm_120

l1_probe: l1_probe.cu
	nvcc -O2 -std=c++17 -arch=$(ARCH) $< -o $@

clean:
	rm -f l1_probe

.PHONY: clean
