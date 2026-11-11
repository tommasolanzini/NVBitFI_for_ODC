# Large variant of cnn (batch 256), occupancy measurements only, see cnn_large.cu
TARGET = cnn_large
ARCH ?= 120 # sm_120 = Blackwell RTX / RTX PRO

all: $(TARGET)

$(TARGET): cnn_large.cu
	nvcc -O2 -std=c++17 -arch=sm_$(strip $(ARCH)) cnn_large.cu -o $(TARGET)

# Reference outputs of a run without a tool
golden: $(TARGET)
	./$(TARGET) > golden_stdout.txt 2> golden_stderr.txt
	mv output_logits.bin golden_output_logits.bin

clean:
	rm -f $(TARGET) *.o *.bin stdout.txt stderr.txt golden_stdout.txt golden_stderr.txt

.PHONY: all golden clean
