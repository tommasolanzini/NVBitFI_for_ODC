/*
 * Small convolutional network written by hand, without cuDNN, for fault injection.
 *
 *   input   BATCH images, 1 channel, 32 x 32
 *   k_conv1 3 x 3, C1 filters    -> x1 (30 x 30)
 *   k_relu1 in place
 *   k_pool1 max pooling 2 x 2    -> p1 (15 x 15)
 *   k_conv2 3 x 3, C2 filters    -> x2 (13 x 13)
 *   k_relu2 in place
 *   k_pool2 max pooling 2 x 2    -> p2 (6 x 6)
 *   k_fc    fully connected      -> logits, BATCH x BATCH, row = image
 *
 * Each kernel is launched once and each thread computes one element, so the result
 * is bit-exact across runs. Each intermediate buffer is freed when no longer needed,
 * as a framework would do. Input and weights come from a fixed generator.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BATCH 32
#define C1 4
#define C2 8
#define IN_W 32
#define KERNEL 3
#define W1 (IN_W - KERNEL + 1)  // 30
#define P1 (W1 / 2)             // 15
#define W2 (P1 - KERNEL + 1)    // 13
#define P2 (W2 / 2)             // 6
#define FLAT (C2 * P2 * P2)     // 288
#define CLASSES BATCH           // square output, for the output signature

__global__ void k_conv1(const float *in, const float *w, float *out)
{
	const int img = blockIdx.x, f = blockIdx.y, t = threadIdx.x;
	const int y = t / W1, x = t % W1;
	const float *image = in + img * IN_W * IN_W;
	const float *weights = w + f * KERNEL * KERNEL;
	float sum = 0.f;
	for (int ky = 0; ky < KERNEL; ky++)
		for (int kx = 0; kx < KERNEL; kx++)
			sum += image[(y + ky) * IN_W + x + kx] * weights[ky * KERNEL + kx];
	out[(img * C1 + f) * W1 * W1 + t] = sum;
}

__global__ void k_conv2(const float *in, const float *w, float *out)
{
	const int img = blockIdx.x, f = blockIdx.y, t = threadIdx.x;
	const int y = t / W2, x = t % W2;
	const float *weights = w + f * C1 * KERNEL * KERNEL;
	float sum = 0.f;
	for (int c = 0; c < C1; c++) {
		const float *channel = in + (img * C1 + c) * P1 * P1;
		for (int ky = 0; ky < KERNEL; ky++)
			for (int kx = 0; kx < KERNEL; kx++)
				sum += channel[(y + ky) * P1 + x + kx] * weights[(c * KERNEL + ky) * KERNEL + kx];
	}
	out[(img * C2 + f) * W2 * W2 + t] = sum;
}

__device__ __forceinline__ void relu(float *x, int i)
{
	x[i] = fmaxf(x[i], 0.f);
}

__global__ void k_relu1(float *x)
{
	relu(x, (blockIdx.x * C1 + blockIdx.y) * W1 * W1 + threadIdx.x);
}

__global__ void k_relu2(float *x)
{
	relu(x, (blockIdx.x * C2 + blockIdx.y) * W2 * W2 + threadIdx.x);
}

__device__ __forceinline__ float pool2x2(const float *channel, int width, int y, int x)
{
	const int row0 = 2 * y * width + 2 * x, row1 = row0 + width;
	return fmaxf(fmaxf(channel[row0], channel[row0 + 1]), fmaxf(channel[row1], channel[row1 + 1]));
}

__global__ void k_pool1(const float *in, float *out)
{
	const int c = blockIdx.x * C1 + blockIdx.y, t = threadIdx.x;
	out[c * P1 * P1 + t] = pool2x2(in + c * W1 * W1, W1, t / P1, t % P1);
}

__global__ void k_pool2(const float *in, float *out)
{
	const int c = blockIdx.x * C2 + blockIdx.y, t = threadIdx.x;
	out[c * P2 * P2 + t] = pool2x2(in + c * W2 * W2, W2, t / P2, t % P2);
}

// CTA = image, thread = class, so tid = flat index of the element of the output
__global__ void k_fc(const float *in, const float *w, const float *bias, float *out)
{
	const int img = blockIdx.x, cls = threadIdx.x;
	const float *features = in + img * FLAT;
	const float *weights = w + cls * FLAT;
	float sum = bias[cls];
	for (int i = 0; i < FLAT; i++)
		sum += features[i] * weights[i];
	out[img * CLASSES + cls] = sum;
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

static float *upload(const float *host, size_t count, const char *what)
{
	float *d;
	check(cudaMalloc(&d, count * sizeof(float)), what);
	check(cudaMemcpy(d, host, count * sizeof(float), cudaMemcpyHostToDevice), what);
	return d;
}

static float *alloc(size_t count, const char *what)
{
	float *d;
	check(cudaMalloc(&d, count * sizeof(float)), what);
	return d;
}

static void sync(const char *what)
{
	check(cudaDeviceSynchronize(), what);
}

int main()
{
	setbuf(stdout, NULL);
	static float host_in[BATCH * IN_W * IN_W];
	static float host_w1[C1 * KERNEL * KERNEL];
	static float host_w2[C2 * C1 * KERNEL * KERNEL];
	static float host_wfc[CLASSES * FLAT];
	static float host_bias[CLASSES];
	static float host_out[BATCH * CLASSES];

	unsigned int state = 2718u;
	for (float &v : host_in) { const float r = next_input(&state); v = r < 0.f ? -r : r; } // pixels in [0, 1]
	for (float &v : host_w1) v = 0.5f * next_input(&state);
	for (float &v : host_w2) v = 0.25f * next_input(&state);
	for (float &v : host_wfc) v = 0.05f * next_input(&state);
	for (float &v : host_bias) v = 0.1f * next_input(&state);

	float *in = upload(host_in, BATCH * IN_W * IN_W, "input");
	float *w1 = upload(host_w1, C1 * KERNEL * KERNEL, "w1");
	float *w2 = upload(host_w2, C2 * C1 * KERNEL * KERNEL, "w2");
	float *wfc = upload(host_wfc, CLASSES * FLAT, "wfc");
	float *bias = upload(host_bias, CLASSES, "bias");
	printf("CNN batch=%d classes=%d\n", BATCH, CLASSES);

	float *x1 = alloc(BATCH * C1 * W1 * W1, "x1");
	k_conv1<<<dim3(BATCH, C1), W1 * W1>>>(in, w1, x1);
	sync("k_conv1");
	k_relu1<<<dim3(BATCH, C1), W1 * W1>>>(x1);
	sync("k_relu1");
	float *p1 = alloc(BATCH * C1 * P1 * P1, "p1");
	k_pool1<<<dim3(BATCH, C1), P1 * P1>>>(x1, p1);
	sync("k_pool1");
	check(cudaFree(x1), "free x1");

	float *x2 = alloc(BATCH * C2 * W2 * W2, "x2");
	k_conv2<<<dim3(BATCH, C2), W2 * W2>>>(p1, w2, x2);
	sync("k_conv2");
	check(cudaFree(p1), "free p1");
	k_relu2<<<dim3(BATCH, C2), W2 * W2>>>(x2);
	sync("k_relu2");
	float *p2 = alloc(BATCH * FLAT, "p2");
	k_pool2<<<dim3(BATCH, C2), P2 * P2>>>(x2, p2);
	sync("k_pool2");
	check(cudaFree(x2), "free x2");

	float *out = alloc(BATCH * CLASSES, "out");
	k_fc<<<BATCH, CLASSES>>>(p2, wfc, bias, out);
	sync("k_fc");
	check(cudaFree(p2), "free p2");
	check(cudaMemcpy(host_out, out, sizeof(host_out), cudaMemcpyDeviceToHost), "copy out");

	FILE *f = fopen("output_logits.bin", "wb");
	if (!f || fwrite(host_out, sizeof(float), BATCH * CLASSES, f) != BATCH * CLASSES || fclose(f) != 0) {
		printf("Error: cannot write output_logits.bin\n");
		exit(1);
	}
	check(cudaFree(in), "free in");
	check(cudaFree(w1), "free w1");
	check(cudaFree(w2), "free w2");
	check(cudaFree(wfc), "free wfc");
	check(cudaFree(bias), "free bias");
	check(cudaFree(out), "free out");
	printf("done\n");
	return 0;
}
