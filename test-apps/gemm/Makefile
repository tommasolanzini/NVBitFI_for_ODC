TARGET = gemm
ARCH ?= 120 # sm_120 = Blackwell RTX / RTX PRO

all: $(TARGET)

$(TARGET): gemm.cu
	nvcc -O2 -std=c++17 -arch=sm_$(strip $(ARCH)) gemm.cu -o $(TARGET)

# Reference outputs of a run without injection
golden: $(TARGET)
	./$(TARGET) > golden_stdout.txt 2> golden_stderr.txt
	mv output_core.bin golden_output_core.bin
	mv output_tc.bin golden_output_tc.bin

clean:
	rm -f $(TARGET) *.o *.bin stdout.txt stderr.txt golden_stdout.txt golden_stderr.txt

.PHONY: all golden clean
