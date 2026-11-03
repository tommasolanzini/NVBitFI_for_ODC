# Experiment on the memory descriptor (see README in this directory).
NVBIT_PATH ?= ../../../../core
ARCH ?= sm_120
NVCC = nvcc -ccbin=$(CXX) -D_FORCE_INLINES -I$(NVBIT_PATH)
CUDA_LIB = -L$(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

all: desc_probe desc_tool.so

desc_probe: desc_probe.cu
	nvcc -O2 -std=c++17 -arch=$(ARCH) $< -o $@

desc_tool.o: desc_tool.cu
	$(NVCC) -dc -c -std=c++17 -Xptxas -cloning=no -arch=$(ARCH) -O3 -Xcompiler -fPIC $< -o $@

desc_funcs.o: desc_funcs.cu
	$(NVCC) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -fPIC -c $< -o $@

desc_tool.so: desc_funcs.o desc_tool.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 desc_funcs.o desc_tool.o -L$(NVBIT_PATH) -lnvbit $(CUDA_LIB) -lcuda -lcudart_static -shared -o $@

clean:
	rm -f desc_probe *.o *.so

.PHONY: all clean
