# Large variant of gemm, occupancy measurements only (see gemm_large.cu).
# gemm_large: N=4096. gemm_n<N>: the same source at other sizes, for the liveness tool,
# which cannot run N=4096 (thread slots, 24 bytes x 47 per thread, beyond the 8 GB of the GPU).
ARCH ?= 120 # sm_120 = Blackwell RTX / RTX PRO
SIZES = 128 256 512 1024 2048
TARGETS = gemm_large $(SIZES:%=gemm_n%)
NVCC = nvcc -O2 -std=c++17 -arch=sm_$(strip $(ARCH))

all: $(TARGETS)

gemm_large: gemm_large.cu
	$(NVCC) -DN=4096 $< -o $@

gemm_n%: gemm_large.cu
	$(NVCC) -DN=$* $< -o $@

# Reference outputs of runs without a tool, one per binary: golden-<binary>/
golden: $(TARGETS)
	for t in $(TARGETS); do mkdir -p golden-$$t && (cd golden-$$t && ../$$t > golden_stdout.txt 2> golden_stderr.txt && mv output_hash.txt golden_output_hash.txt) || exit 1; done

clean:
	rm -rf $(TARGETS) golden-* *.txt

.PHONY: all golden clean
