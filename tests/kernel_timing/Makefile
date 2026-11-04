# Builds repeat_<app>: a generated copy of a workload where every kernel launch
# runs LAUNCHES times and prints its host-side duration. The workloads themselves
# keep launching each kernel once. Do not use these binaries for injection.
ARCH ?= 120
APPS = ptr_chase shm_stencil gemm
TARGETS = $(APPS:%=repeat_%) empty_kernel

all: $(TARGETS)

define repeat_rule
repeat_$(1): ../../test-apps/$(1)/$(1).cu repeat.h generate.py
	python3 generate.py $$< $$@.gen.cu
	nvcc -O2 -std=c++17 -I. -arch=sm_$$(strip $$(ARCH)) $$@.gen.cu -o $$@
endef
$(foreach a,$(APPS),$(eval $(call repeat_rule,$(a))))

# Host time of an empty kernel: the floor of the durations, see README.md
empty_kernel: empty_kernel.cu
	nvcc -O2 -arch=sm_$(strip $(ARCH)) $< -o $@

clean:
	rm -f $(TARGETS) *.gen.cu *.bin

.PHONY: all clean
