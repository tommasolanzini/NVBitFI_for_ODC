# Offline tests (no GPU, no NVBit needed)
#   make -C tests check                       built-in expected cases
#   tests/extract_sass.sh <binary or .so> | tests/dests_test
#                                             per-opcode summary on real SASS
CXX ?= g++

dests_test: dests_test.cpp ../common/dests.h ../common/arch.h
	$(CXX) -std=c++17 -O2 -Wall -I../common $< -o $@

check: dests_test
	./dests_test --check
	python3 test_fp_signature.py
	python3 test_liveness.py
	python3 test_final_block.py
	python3 test_fault_locations.py

clean:
	rm -f dests_test

.PHONY: check clean
