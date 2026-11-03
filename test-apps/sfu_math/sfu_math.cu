/*
 * Workload for the SFU group (MUFU instructions, Special Function Unit).
 *
 * One kernel per function, each computing out[i] = f(in[i]) on an N x N grid,
 * one thread per element (CTA = row, thread = column). The fast intrinsics are
 * used on purpose: they map to MUFU, while the default math library functions
 * are mostly software polynomials. Each result is written in full (row-major
 * fp32) to output_<function>.bin.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 128

#define SFU_KERNEL(name, expr) \
	__global__ void name(const float *in, float *out, int n) { \
		const int i = blockIdx.x * n + threadIdx.x; \
		const float x = in[i]; \
		out[i] = (expr); \
	}

SFU_KERNEL(k_rsqrt, __frsqrt_rn(x))
SFU_KERNEL(k_rcp, __frcp_rn(x))
SFU_KERNEL(k_sin, __sinf(x))
SFU_KERNEL(k_exp, __expf(x))
SFU_KERNEL(k_log, __logf(x))

static void check(const char *what)
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess) {
		printf("Error: %s failed: %s\n", what, cudaGetErrorString(e));
		exit(1);
	}
}

static float host_in[N * N], host_out[N * N];

static void finish(const char *name, const float *out)
{
	cudaDeviceSynchronize();
	check(name);
	cudaMemcpy(host_out, out, sizeof(host_out), cudaMemcpyDeviceToHost);
	check(name);
	char fname[64];
	snprintf(fname, sizeof(fname), "output_%s.bin", name);
	FILE *f = fopen(fname, "wb");
	if (!f || fwrite(host_out, sizeof(float), N * N, f) != N * N || fclose(f) != 0) {
		printf("Error: cannot write %s\n", fname);
		exit(1);
	}
}

int main()
{
	setbuf(stdout, NULL);
	// Inputs in (0, 2], valid for all the functions, from a fixed generator
	unsigned int state = 12345u;
	for (int i = 0; i < N * N; i++) {
		state = state * 1664525u + 1013904223u;
		host_in[i] = (float)((state >> 16) % 2048u + 1u) / 1024.f;
	}

	float *in, *out;
	cudaMalloc(&in, sizeof(host_in));
	cudaMalloc(&out, sizeof(host_out));
	cudaMemcpy(in, host_in, sizeof(host_in), cudaMemcpyHostToDevice);
	check("setup");
	printf("SFU math N=%d\n", N);

#define RUN(name) \
	cudaMemset(out, 0, sizeof(host_out)); /* a kernel that writes nothing must not inherit the previous result */ \
	k_##name<<<N, N>>>(in, out, N); \
	finish(#name, out);

	RUN(rsqrt)
	RUN(rcp)
	RUN(sin)
	RUN(exp)
	RUN(log)

	printf("done\n");
	return 0;
}
