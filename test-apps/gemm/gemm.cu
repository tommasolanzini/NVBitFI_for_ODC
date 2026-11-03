/*
 * GEMM workload for fault injection: C = A * B on two execution paths.
 *
 *  gemm_core: fp32 on the CUDA cores, one thread per element of C.
 *  gemm_tc:   Tensor Cores through WMMA (fp16 inputs, fp32 accumulator), one
 *             warp per 16x16 tile of C.
 *
 * Inputs come from a fixed generator and the summation order is fixed, so the
 * output is bit-exact across runs. Each result is written in full (row-major
 * fp32) to output_core.bin and output_tc.bin, to locate wrong elements later.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define N 128 // matrix size, multiple of TILE
#define TILE 16 // WMMA fragment size (m16n16k16)

__global__ void gemm_core(const float *A, const float *B, float *C, int n)
{
	const int row = blockIdx.x;
	const int col = threadIdx.x;
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

// Values k/1024 in [-1, 1]: exact in both fp16 and fp32, so the two paths
// multiply the same numbers. For N=128 the products and sums are also exact in
// fp32 (grid 2^-20, magnitude < 16), so both paths give the same bits.
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

static void write_matrix(const char *fname, const float *m)
{
	FILE *f = fopen(fname, "wb");
	if (!f || fwrite(m, sizeof(float), N * N, f) != N * N || fclose(f) != 0) {
		printf("Error: cannot write %s\n", fname);
		exit(1);
	}
}

int main()
{
	setbuf(stdout, NULL);
	static float host_a[N * N], host_b[N * N], host_c[N * N];
	static half host_a_h[N * N], host_b_h[N * N];

	unsigned int state = 12345u;
	for (int i = 0; i < N * N; i++) { host_a[i] = next_input(&state); host_a_h[i] = __float2half(host_a[i]); }
	for (int i = 0; i < N * N; i++) { host_b[i] = next_input(&state); host_b_h[i] = __float2half(host_b[i]); }

	float *a, *b, *c, *c_tc; // one result buffer per path: a kernel that writes nothing must not inherit the other's result
	half *a_h, *b_h;
	cudaMalloc(&a, sizeof(host_a));
	cudaMalloc(&b, sizeof(host_b));
	cudaMalloc(&c, sizeof(host_c));
	cudaMalloc(&c_tc, sizeof(host_c));
	cudaMalloc(&a_h, sizeof(host_a_h));
	cudaMalloc(&b_h, sizeof(host_b_h));
	cudaMemcpy(a, host_a, sizeof(host_a), cudaMemcpyHostToDevice);
	cudaMemcpy(b, host_b, sizeof(host_b), cudaMemcpyHostToDevice);
	cudaMemcpy(a_h, host_a_h, sizeof(host_a_h), cudaMemcpyHostToDevice);
	cudaMemcpy(b_h, host_b_h, sizeof(host_b_h), cudaMemcpyHostToDevice);
	check("setup");
	printf("GEMM N=%d\n", N);

	gemm_core<<<N, N>>>(a, b, c, N);
	cudaDeviceSynchronize();
	check("gemm_core");
	cudaMemcpy(host_c, c, sizeof(host_c), cudaMemcpyDeviceToHost);
	check("copy core");
	write_matrix("output_core.bin", host_c);

	gemm_tc<<<dim3(N / TILE, N / TILE), 32>>>(a_h, b_h, c_tc, N);
	cudaDeviceSynchronize();
	check("gemm_tc");
	cudaMemcpy(host_c, c_tc, sizeof(host_c), cudaMemcpyDeviceToHost);
	check("copy tc");
	write_matrix("output_tc.bin", host_c);

	printf("done\n");
	return 0;
}
