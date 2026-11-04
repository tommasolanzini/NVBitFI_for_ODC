# Memory-data injector: corrupts one bit of global memory where it is stored.
# Build settings are shared with the other tools in ../common/build.mk.
# Usage: make [NVBIT_PATH=...] [ARCH=sm_120] [DUMMY=1]
NVBIT_TOOL = mem_injector.so
TOOL_FLAGS = -DDUMMY=$(DUMMY) # the host part applies the mask of timing mode a
include ../common/build.mk
