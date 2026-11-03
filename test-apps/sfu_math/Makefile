TARGET = sfu_math
ARCH ?= 120 # sm_120 = Blackwell RTX / RTX PRO
FUNCS = rsqrt rcp sin exp log

all: $(TARGET)

$(TARGET): sfu_math.cu
	nvcc -O2 -std=c++17 -arch=sm_$(strip $(ARCH)) sfu_math.cu -o $(TARGET)

# Reference outputs of a run without injection
golden: $(TARGET)
	./$(TARGET) > golden_stdout.txt 2> golden_stderr.txt
	for f in $(FUNCS); do mv output_$$f.bin golden_output_$$f.bin; done

clean:
	rm -f $(TARGET) *.o *.bin stdout.txt stderr.txt golden_stdout.txt golden_stderr.txt

.PHONY: all golden clean
