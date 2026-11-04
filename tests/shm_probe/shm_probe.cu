/*
 * Experiments on shared memory for the memory-data injector, see README.md.
 *
 *   shm_probe atomic     are atomics on shared memory atomic? Plain read-modify-
 *                        write is the control, it must lose updates.
 *   shm_probe occupancy  how many CTAs are resident on one SM, for each size of
 *                        the shared memory of a CTA.
 *   shm_probe known      known values in shared memory, copied to global memory.
 *                        Used with the tool in tool/, which flips a bit of them.
 *   shm_probe toolxor    shared word left to the tool, which XORs it on every
 *                        global load: tells if the tool's atomics are atomic.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_SMS_MAX 64
#define KNOWN_WORDS 64

static void check(cudaError_t e, const char *what)
{
	if (e != cudaSuccess) {
		printf("{\"error\": \"%s: %s\"}\n", what, cudaGetErrorString(e));
		exit(1);
	}
}

// T threads, R rounds. Bit t%32 is XORed R times by each thread.
__global__ void k_atomic(unsigned *out, int reps)
{
	__shared__ unsigned w[3];
	const unsigned t = threadIdx.x;
	if (t == 0) {
		w[0] = 0;
		w[1] = 0;
		w[2] = 0;
	}
	__syncthreads();
	volatile unsigned *plain = &w[2];
	for (int r = 0; r < reps; r++) {
		atomicXor(&w[0], 1u << (t % 32));
		atomicAdd(&w[1], 1u);
		*plain = *plain + 1;
	}
	__syncthreads();
	if (t == 0) {
		out[blockIdx.x * 3 + 0] = w[0];
		out[blockIdx.x * 3 + 1] = w[1];
		out[blockIdx.x * 3 + 2] = w[2];
	}
}

static int run_atomic()
{
	const int threads = 1023, reps = 1001, ctas = 200;
	unsigned expected_xor = 0;
	for (int t = 0; t < threads; t++) {
		if (reps & 1) expected_xor ^= 1u << (t % 32);
	}
	const unsigned expected_add = (unsigned)threads * reps;

	unsigned *dev;
	static unsigned host[ctas * 3];
	check(cudaMalloc(&dev, sizeof(host)), "malloc");
	k_atomic<<<ctas, threads>>>(dev, reps);
	check(cudaDeviceSynchronize(), "k_atomic");
	check(cudaMemcpy(host, dev, sizeof(host), cudaMemcpyDeviceToHost), "copy");

	int bad_xor = 0, bad_add = 0, bad_plain = 0;
	unsigned long long lost_plain = 0;
	for (int c = 0; c < ctas; c++) {
		bad_xor += host[c * 3] != expected_xor;
		bad_add += host[c * 3 + 1] != expected_add;
		bad_plain += host[c * 3 + 2] != expected_add;
		lost_plain += expected_add - host[c * 3 + 2];
	}
	printf("{\"test\": \"atomic\", \"ctas\": %d, \"threads\": %d, \"rounds\": %d, \"expected_xor\": %u, \"expected_add\": %u, "
		"\"ctas_wrong_atomic_xor\": %d, \"ctas_wrong_atomic_add\": %d, \"ctas_wrong_plain\": %d, \"plain_updates_lost\": %llu}\n",
		ctas, threads, reps, expected_xor, expected_add, bad_xor, bad_add, bad_plain, lost_plain);
	return 0;
}

extern __shared__ unsigned char dyn_shared[];

// Resident means: between the first and the last line of thread 0. The wait
// keeps a CTA resident long enough for the others to arrive.
__global__ void k_occupancy(unsigned *inflight, unsigned *peak, unsigned *total, int size, unsigned wait)
{
	unsigned sm;
	asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
	if (threadIdx.x == 0) {
		const unsigned now = atomicAdd(&inflight[sm], 1u) + 1;
		atomicMax(&peak[sm], now);
		atomicAdd(&total[sm], 1u);
		if (size > 0) {
			dyn_shared[0] = 1;
			dyn_shared[size - 1] = 1;
		}
		const unsigned t0 = clock();
		while (clock() - t0 < wait) {}
		atomicAdd(&inflight[sm], (unsigned)-1);
	}
	__syncthreads();
}

static int run_occupancy()
{
	int sms = 0, optin = 0, perblock = 0;
	cudaDeviceProp prop;
	check(cudaGetDeviceProperties(&prop, 0), "props");
	sms = prop.multiProcessorCount;
	perblock = (int)prop.sharedMemPerBlock;
	optin = (int)prop.sharedMemPerBlockOptin;
	printf("{\"test\": \"device\", \"sms\": %d, \"sharedMemPerBlock\": %d, \"sharedMemPerBlockOptin\": %d, \"sharedMemPerMultiprocessor\": %zu, \"maxBlocksPerMultiProcessor\": %d, \"reservedSharedMemPerBlock\": %zu}\n",
		sms, perblock, optin, prop.sharedMemPerMultiprocessor, prop.maxBlocksPerMultiProcessor, prop.reservedSharedMemPerBlock);

	unsigned *inflight, *peak, *total;
	check(cudaMalloc(&inflight, NUM_SMS_MAX * 4), "malloc");
	check(cudaMalloc(&peak, NUM_SMS_MAX * 4), "malloc");
	check(cudaMalloc(&total, NUM_SMS_MAX * 4), "malloc");

	const int sizes[] = {0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 48 * 1024, 64 * 1024, 96 * 1024, 128 * 1024, 192 * 1024, 228 * 1024};
	for (int size : sizes) {
		if (size > optin) continue;
		check(cudaFuncSetAttribute(k_occupancy, cudaFuncAttributeMaxDynamicSharedMemorySize, size), "attr");
		check(cudaMemset(inflight, 0, NUM_SMS_MAX * 4), "memset");
		check(cudaMemset(peak, 0, NUM_SMS_MAX * 4), "memset");
		check(cudaMemset(total, 0, NUM_SMS_MAX * 4), "memset");
		k_occupancy<<<3000, 32, size>>>(inflight, peak, total, size, 3000000u);
		check(cudaDeviceSynchronize(), "k_occupancy");
		unsigned p[NUM_SMS_MAX], t[NUM_SMS_MAX];
		check(cudaMemcpy(p, peak, sizeof(p), cudaMemcpyDeviceToHost), "copy");
		check(cudaMemcpy(t, total, sizeof(t), cudaMemcpyDeviceToHost), "copy");
		unsigned peak_max = 0, sum = 0;
		for (int s = 0; s < sms; s++) {
			if (p[s] > peak_max) peak_max = p[s];
			sum += t[s];
		}
		printf("{\"test\": \"occupancy\", \"shared_bytes\": %d, \"peak_ctas_on_one_sm\": %u, \"peak_bytes\": %llu, \"ctas\": %u, \"peak_per_sm\": [",
			size, peak_max, (unsigned long long)peak_max * size, sum);
		for (int s = 0; s < sms; s++) printf("%s%u", s ? ", " : "", p[s]);
		printf("]}\n");
		fflush(stdout);
	}
	return 0;
}

// Thread i stores word i, then a marker after a barrier: the tool flips right
// after the marker, when all the words hold their values and none was read.
__global__ void k_known(unsigned *out, unsigned *marker_out)
{
	__shared__ unsigned w[KNOWN_WORDS + 1];
	const unsigned t = threadIdx.x;
	w[t] = 0x10000u + t;
	__syncthreads();
	if (t == 0) w[KNOWN_WORDS] = 0xabcd;
	__syncthreads();
	out[blockIdx.x * KNOWN_WORDS + t] = w[t];
	if (t == 0) marker_out[blockIdx.x] = w[KNOWN_WORDS];
}

static int run_known()
{
	const int ctas = 4;
	unsigned *dev, *mark;
	static unsigned host[ctas * KNOWN_WORDS], hmark[ctas];
	check(cudaMalloc(&dev, sizeof(host)), "malloc");
	check(cudaMalloc(&mark, sizeof(hmark)), "malloc");
	k_known<<<ctas, KNOWN_WORDS>>>(dev, mark);
	check(cudaDeviceSynchronize(), "k_known");
	check(cudaMemcpy(host, dev, sizeof(host), cudaMemcpyDeviceToHost), "copy");
	int wrong = 0;
	for (int c = 0; c < ctas; c++) {
		for (int i = 0; i < KNOWN_WORDS; i++) {
			const unsigned e = 0x10000u + i;
			const unsigned v = host[c * KNOWN_WORDS + i];
			if (v != e) {
				printf("{\"test\": \"known\", \"cta\": %d, \"word\": %d, \"expected\": %u, \"got\": %u, \"xor\": %u}\n", c, i, e, v, v ^ e);
				wrong++;
			}
		}
	}
	printf("{\"test\": \"known\", \"wrong_words\": %d}\n", wrong);
	return 0;
}

// The application only clears w[0] and reads it at the end. In between the tool
// XORs it once per global load, from 1023 threads.
__global__ void k_toolxor(volatile const unsigned *g, unsigned *out, unsigned *sink, int reps)
{
	__shared__ unsigned w[1];
	volatile unsigned *word = &w[0]; // otherwise the compiler folds the read to 0 and removes w
	const unsigned t = threadIdx.x;
	if (t == 0) *word = 0;
	__syncthreads();
	unsigned x = 0;
	for (int r = 0; r < reps; r++) x += g[r & 15];
	__syncthreads();
	if (t == 0) out[blockIdx.x] = *word;
	sink[blockIdx.x] = x;
}

static int run_toolxor()
{
	const int threads = 1023, reps = 1001, ctas = 50;
	unsigned expected = 0;
	for (int t = 0; t < threads; t++) {
		if (reps & 1) expected ^= 1u << (t % 32);
	}
	unsigned *g, *out, *sink;
	static unsigned host[ctas];
	check(cudaMalloc(&g, 16 * 4), "malloc");
	check(cudaMalloc(&out, sizeof(host)), "malloc");
	check(cudaMalloc(&sink, sizeof(host)), "malloc");
	check(cudaMemset(g, 0, 16 * 4), "memset");
	k_toolxor<<<ctas, threads>>>(g, out, sink, reps);
	check(cudaDeviceSynchronize(), "k_toolxor");
	check(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost), "copy");
	int bad = 0;
	for (int c = 0; c < ctas; c++) bad += host[c] != expected;
	printf("{\"test\": \"toolxor\", \"ctas\": %d, \"threads\": %d, \"rounds\": %d, \"expected\": %u, \"ctas_wrong\": %d, \"first\": %u}\n", ctas, threads, reps, expected, bad, host[0]);
	return 0;
}

int main(int argc, char **argv)
{
	setbuf(stdout, NULL);
	const char *what = argc > 1 ? argv[1] : "";
	if (!strcmp(what, "atomic")) return run_atomic();
	if (!strcmp(what, "occupancy")) return run_occupancy();
	if (!strcmp(what, "known")) return run_known();
	if (!strcmp(what, "toolxor")) return run_toolxor();
	printf("usage: shm_probe atomic|occupancy|known|toolxor\n");
	return 2;
}
