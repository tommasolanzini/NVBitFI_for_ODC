TARGET = ptr_chase
ARCH ?= 120 # sm_120 = Blackwell RTX / RTX PRO

all: $(TARGET)

$(TARGET): ptr_chase.cu
	nvcc -O2 -std=c++17 -arch=sm_$(strip $(ARCH)) ptr_chase.cu -o $(TARGET)

# Reference outputs of a run without injection
golden: $(TARGET)
	./$(TARGET) > golden_stdout.txt 2> golden_stderr.txt
	mv output_ptr.bin golden_output_ptr.bin

clean:
	rm -f $(TARGET) *.o *.bin stdout.txt stderr.txt golden_stdout.txt golden_stderr.txt

.PHONY: all golden clean
