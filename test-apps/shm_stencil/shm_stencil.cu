/*
 * Shared memory workload for fault injection: 1-D stencil on each row of a
 * matrix, with all the working data in the shared memory of the CTA.
 *
 * CTA = row, thread = column, so tid = flat index of the element. Three arrays:
 *   coef     written once, read in every iteration: long-lived data
 *   buf[0/1] written and read in turn: each word is overwritten every second
 *            iteration
 * A word is read only by the threads of its CTA, so a flip in shared memory can
 * only change the row of its CTA. The result is bit-exact across runs: every
 * thread executes the same operations on the same values, the barriers order
 * the iterations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 128     // rows, threads per CTA and words per array
#define ITERS 8

__global__ void shm_stencil(const float *in, const float *in_coef, float *out)
{
	__shared__ float buf[2][N];
	__shared__ float coef[N];
	const int row = blockIdx.x;
	const int t = threadIdx.x;

	buf[0][t] = in[row * N + t];
	coef[t] = in_coef[row * N + t];
	__syncthreads();

	const int left = t > 0 ? t - 1 : t;
	const int right = t < N - 1 ? t + 1 : t;
	for (int it = 0; it < ITERS; it++) {
		const int src = it & 1;
		const float c = coef[t];
		const float center = buf[src][t];
		const float sides = 0.5f * (buf[src][left] + buf[src][right]);
		buf[1 - src][t] = c * center + (1.f - c) * sides;
		__syncthreads();
	}
	out[row * N + t] = buf[ITERS & 1][t];
}

static float next_input(unsigned int *state)
{
	*state = *state * 1664525u + 1013904223u;
	return (float)((int)((*state >> 16) % 2049u) - 1024) / 1024.f;
}

static void check(cudaError_t e, const char *what)
{
	if (e != cudaSuccess) {
		printf("Error: %s failed: %s\n", what, cudaGetErrorString(e));
		exit(1);
	}
}

int main()
{
	setbuf(stdout, NULL);
	static float host_in[N * N], host_coef[N * N], host_out[N * N];
	unsigned int state = 54321u;
	for (int i = 0; i < N * N; i++) host_in[i] = next_input(&state);
	for (int i = 0; i < N * N; i++) host_coef[i] = 0.5f + 0.25f * next_input(&state);

	float *in, *coef, *out;
	check(cudaMalloc(&in, sizeof(host_in)), "malloc");
	check(cudaMalloc(&coef, sizeof(host_coef)), "malloc");
	check(cudaMalloc(&out, sizeof(host_out)), "malloc");
	check(cudaMemcpy(in, host_in, sizeof(host_in), cudaMemcpyHostToDevice), "copy in");
	check(cudaMemcpy(coef, host_coef, sizeof(host_coef), cudaMemcpyHostToDevice), "copy coef");
	printf("SHM_STENCIL N=%d iters=%d\n", N, ITERS);

	shm_stencil<<<N, N>>>(in, coef, out);
	check(cudaDeviceSynchronize(), "shm_stencil");
	check(cudaMemcpy(host_out, out, sizeof(host_out), cudaMemcpyDeviceToHost), "copy out");

	FILE *f = fopen("output_shm.bin", "wb");
	if (!f || fwrite(host_out, sizeof(float), N * N, f) != N * N || fclose(f) != 0) {
		printf("Error: cannot write output_shm.bin\n");
		exit(1);
	}
	printf("done\n");
	return 0;
}
