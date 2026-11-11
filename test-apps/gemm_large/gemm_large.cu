/*
 * Large variant of test-apps/gemm, for occupancy measurements only (no injection campaign).
 *
 * Same two paths, same inputs generator and same gemm_tc kernel as gemm.cu. Differences:
 *  - N is set at build time (-DN=...), default 4096;
 *  - gemm_core: one thread per element as before, but with 128 threads per CTA and a 2-D
 *    grid (row = blockIdx.x, column = blockIdx.y * 128 + threadIdx.x), since N > 1024
 *    does not fit in one CTA. For N = 128 the grid is the same as gemm.cu;
 *  - the outputs are not written in full (64 MB each at N = 4096): output_hash.txt has
 *    one FNV-1a hash per result matrix, enough to see that a tool did not change them.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#ifndef N
#define N 4096 // matrix size, multiple of CORE_THREADS and TILE
#endif
#define TILE 16 // WMMA fragment size (m16n16k16)
#define CORE_THREADS 128 // threads per CTA of gemm_core, as in gemm.cu
#if N % CORE_THREADS != 0
#error "N must be a multiple of 128"
#endif

__global__ void gemm_core(const float *A, const float *B, float *C, int n)
{
	const int row = blockIdx.x;
	const int col = blockIdx.y * blockDim.x + threadIdx.x;
	float sum = 0.f;
	for (int k = 0; k < n; k++) {
		sum += A[row * n + k] * B[k * n + col];
	}
	C[row * n + col] = sum;
}

__global__ void gemm_tc(const half *A, const half *B, float *C, int n)
{
	const int tile_row = blockIdx.y * TILE;
	const int tile_col = blockIdx.x * TILE;

	wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a;
	wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> b;
	wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c;

	wmma::fill_fragment(c, 0.f);
	for (int k = 0; k < n; k += TILE) {
		wmma::load_matrix_sync(a, A + tile_row * n + k, n);
		wmma::load_matrix_sync(b, B + k * n + tile_col, n);
		wmma::mma_sync(c, a, b, c);
	}
	wmma::store_matrix_sync(C + tile_row * n + tile_col, c, n, wmma::mem_row_major);
}

static float next_input(unsigned int *state)
{
	*state = *state * 1664525u + 1013904223u;
	return (float)((int)((*state >> 16) % 2049u) - 1024) / 1024.f;
}

static void check(const char *what)
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess) {
		printf("Error: %s failed: %s\n", what, cudaGetErrorString(e));
		exit(1);
	}
}

static uint64_t fnv1a(const void *p, size_t bytes)
{
	uint64_t h = 1469598103934665603ull;
	for (size_t i = 0; i < bytes; i++) {
		h ^= ((const unsigned char *)p)[i];
		h *= 1099511628211ull;
	}
	return h;
}

int main()
{
	setbuf(stdout, NULL);
	const size_t nn = (size_t)N * N;
	float *host_a = (float *)malloc(nn * sizeof(float)), *host_b = (float *)malloc(nn * sizeof(float)), *host_c = (float *)malloc(nn * sizeof(float));
	half *host_a_h = (half *)malloc(nn * sizeof(half)), *host_b_h = (half *)malloc(nn * sizeof(half));
	if (!host_a || !host_b || !host_c || !host_a_h || !host_b_h) {
		printf("Error: host allocation failed\n");
		exit(1);
	}

	unsigned int state = 12345u;
	for (size_t i = 0; i < nn; i++) { host_a[i] = next_input(&state); host_a_h[i] = __float2half(host_a[i]); }
	for (size_t i = 0; i < nn; i++) { host_b[i] = next_input(&state); host_b_h[i] = __float2half(host_b[i]); }

	float *a, *b, *c, *c_tc;
	half *a_h, *b_h;
	cudaMalloc(&a, nn * sizeof(float));
	cudaMalloc(&b, nn * sizeof(float));
	cudaMalloc(&c, nn * sizeof(float));
	cudaMalloc(&c_tc, nn * sizeof(float));
	cudaMalloc(&a_h, nn * sizeof(half));
	cudaMalloc(&b_h, nn * sizeof(half));
	cudaMemcpy(a, host_a, nn * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b, host_b, nn * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(a_h, host_a_h, nn * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(b_h, host_b_h, nn * sizeof(half), cudaMemcpyHostToDevice);
	check("setup");
	printf("GEMM large N=%d\n", N);

	gemm_core<<<dim3(N, N / CORE_THREADS), CORE_THREADS>>>(a, b, c, N);
	cudaDeviceSynchronize();
	check("gemm_core");
	cudaMemcpy(host_c, c, nn * sizeof(float), cudaMemcpyDeviceToHost);
	check("copy core");
	const uint64_t h_core = fnv1a(host_c, nn * sizeof(float));

	gemm_tc<<<dim3(N / TILE, N / TILE), 32>>>(a_h, b_h, c_tc, N);
	cudaDeviceSynchronize();
	check("gemm_tc");
	cudaMemcpy(host_c, c_tc, nn * sizeof(float), cudaMemcpyDeviceToHost);
	check("copy tc");
	const uint64_t h_tc = fnv1a(host_c, nn * sizeof(float));

	FILE *f = fopen("output_hash.txt", "w");
	if (!f || fprintf(f, "core %016llx\ntc %016llx\n", (unsigned long long)h_core, (unsigned long long)h_tc) < 0 || fclose(f) != 0) {
		printf("Error: cannot write output_hash.txt\n");
		exit(1);
	}
	printf("done\n");
	return 0;
}
