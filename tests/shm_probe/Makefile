# Experiments on shared memory (see README in this directory).
ARCH ?= sm_120

all: shm_probe
	$(MAKE) -C tool ARCH=$(ARCH)

shm_probe: shm_probe.cu
	nvcc -O2 -std=c++17 -arch=$(ARCH) $< -o $@

clean:
	rm -f shm_probe
	$(MAKE) -C tool clean

.PHONY: all clean
