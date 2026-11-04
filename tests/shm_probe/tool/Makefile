# Small NVBit tool for the shared memory probe (see ../README.md).
NVBIT_PATH ?= ../../../../../core
ARCH ?= sm_120
INCLUDES = -I$(NVBIT_PATH) -I../../../common
NVCC = nvcc -ccbin=$(CXX) -D_FORCE_INLINES
NVCC_LIB = -L$(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

probe_tool.so: probe_tool.o inject_funcs.o
	$(NVCC) -arch=$(ARCH) -O3 $^ -L$(NVBIT_PATH) -lnvbit $(NVCC_LIB) -lcuda -lcudart_static -shared -o $@

probe_tool.o: probe_tool.cu probe.h
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no -arch=$(ARCH) -O3 -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu probe.h
	$(NVCC) $(INCLUDES) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.so *.o

.PHONY: clean
