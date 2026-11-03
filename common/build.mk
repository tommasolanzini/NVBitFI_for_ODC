# Shared build settings for the NVBitFI tools (injector, profiler, pf_injector).
# Requires NVBit >= 1.7 (tested with NVBit 1.8) and CUDA >= 12.0.
#
# Overridable from the command line or the environment, e.g.
#   make NVBIT_PATH=/opt/nvbit_release/core ARCH=sm_120
#
# NVBIT_PATH : path to the "core" folder of the NVBit release.
#              Default assumes this repo lives in nvbit_release/tools/<repo>.
# ARCH       : GPU architecture(s) the tool device code is built for.
#              "all" (default) works on every GPU NVBit supports;
#              sm_120 = Blackwell RTX/RTX PRO (e.g. RTX PRO 1000 Blackwell).
# DUMMY      : 1 = go through the whole injection path but do not corrupt
#              anything (sanity check: a DUMMY campaign must give 100% masked).

NVBIT_PATH ?= ../../../core
ARCH       ?= all
DUMMY      ?= 0

NVCC = nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS = ptxas

ifeq ($(DEBUG),1)
NVCC_OPT = -O0 -g
else
NVCC_OPT = -O3
endif

ifneq ($(MAKECMDGOALS),clean)  # no toolchain checks needed to clean
NVCC_VER_REQ = 12.0
NVCC_VER = $(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK = $(shell echo "$(NVCC_VER) >= $(NVCC_VER_REQ)" | bc)
ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc >= $(NVCC_VER_REQ) is required by NVBit >= 1.7)
endif

# Same rule as the NVBit 1.8 sample tools: ptxas < 12.3 needs an explicit
# register cap for the injected device functions, newer ptxas handles it.
PTXAS_VER_ADD_FLAG = 12.3
PTXAS_VER = $(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK = $(shell echo "$(PTXAS_VER) >= $(PTXAS_VER_ADD_FLAG)" | bc)
ifeq ($(PTXAS_VER_CHECK),0)
MAXRREGCOUNT_FLAG = -maxrregcount=24
else
MAXRREGCOUNT_FLAG =
endif
endif

INCLUDES = -I$(NVBIT_PATH) -I../common
LIBS = -L$(NVBIT_PATH) -lnvbit
NVCC_PATH = -L$(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES = $(wildcard *.cu)
OBJECTS = $(SOURCES:.cu=.o)
HEADERS = ../common/*.h $(wildcard *.h)

# Stamp file with the build settings: its timestamp changes only when its
# content changes, so objects are rebuilt when a setting changes (make cannot
# see command-line variables otherwise) and left alone when none did.
FLAGS_STAMP = .build_flags
BUILD_FLAGS = ARCH=$(ARCH) DUMMY=$(DUMMY) DEBUG=$(DEBUG) TOOL_FLAGS=$(TOOL_FLAGS) NVBIT_PATH=$(NVBIT_PATH) NVCC_VER=$(NVCC_VER) MAXRREGCOUNT_FLAG=$(MAXRREGCOUNT_FLAG)

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

$(FLAGS_STAMP): FORCE
	@echo '$(BUILD_FLAGS)' | cmp -s - $@ || echo '$(BUILD_FLAGS)' > $@

%.o: %.cu $(HEADERS) $(FLAGS_STAMP)
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) $(TOOL_FLAGS) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) $(NVCC_OPT) -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu $(HEADERS) $(FLAGS_STAMP)
	$(NVCC) $(INCLUDES) $(TOOL_FLAGS) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -DDUMMY=$(DUMMY) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.so *.o $(FLAGS_STAMP)

FORCE:

.PHONY: all clean FORCE
