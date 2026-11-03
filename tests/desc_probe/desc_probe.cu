/*
 * Probe for the memory descriptor of LDG/STG (desc[URn], loaded by
 * LDCU.64 URn, c[0x0][0x358]): does corrupting it change timing or output?
 *
 * One warp (one CTA, 32 threads) so that the single LDCU.64 of the kernel feeds
 * all its memory accesses. usage: desc_probe <working set KiB> <accesses per thread>
 * A small working set stays in the caches, so it is slowed down if a corrupted
 * descriptor disables them; a large one misses and gives the reference for how
 * much caching changes the time. Prints the time of the second launch on stderr,
 * the checksum of the data read and written on stdout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#define STRIDE 16 // uint32 elements (64 bytes, about a cache line)

__global__ void probe(unsigned int *buf, unsigned int mask, unsigned int accesses, unsigned int *result)
{
	unsigned int idx = threadIdx.x * STRIDE * 97u;
	unsigned int sum = 0;
	for (unsigned int i = 0; i < accesses; i++) {
		idx = (idx + STRIDE * 7u) & mask;
		unsigned int v = buf[idx];
		sum += v;
		buf[idx + 1] = v ^ i; // store in the same line, to exercise the write path too
	}
	result[threadIdx.x] = sum;
}

int main(int argc, char **argv)
{
	if (argc != 3) { printf("usage: %s <KiB, power of 2> <accesses>\n", argv[0]); return 2; }
	const unsigned int words = atoi(argv[1]) * 1024u / 4u;
	const unsigned int accesses = atoi(argv[2]);

	std::vector<unsigned int> init(words);
	for (unsigned int i = 0; i < words; i++) init[i] = i * 2654435761u;
	unsigned int *buf, *result;
	unsigned int host_result[32];
	cudaMalloc(&buf, words * 4);
	cudaMalloc(&result, sizeof(host_result));
	cudaMemcpy(buf, init.data(), words * 4, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// First launch outside the timer: an NVBit tool instruments the kernel here
	probe<<<1, 32>>>(buf, (words - 1) & ~(STRIDE - 1u), 1, result);
	cudaDeviceSynchronize();

	auto t0 = std::chrono::steady_clock::now();
	probe<<<1, 32>>>(buf, (words - 1) & ~(STRIDE - 1u), accesses, result);
	cudaDeviceSynchronize();
	auto t1 = std::chrono::steady_clock::now();
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(e)); return 1; }

	cudaMemcpy(host_result, result, sizeof(host_result), cudaMemcpyDeviceToHost);
	// FNV-1a over the loaded sums and the whole buffer, which holds the stores
	cudaMemcpy(init.data(), buf, words * 4, cudaMemcpyDeviceToHost);
	unsigned int check = 2166136261u;
	for (int i = 0; i < 32; i++) check = (check ^ host_result[i]) * 16777619u;
	for (unsigned int i = 0; i < words; i++) check = (check ^ init[i]) * 16777619u;
	printf("checksum %08x\n", check);
	fprintf(stderr, "%.3f\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
	return 0;
}
