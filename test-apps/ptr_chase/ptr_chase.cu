/*
 * Workload with pointers in global memory, for fault injection in memory data.
 *
 * Each element of the output is the sum over one linked list, walked by one
 * thread (CTA = row, thread = column, tid = flat index of the element). The walk
 * has no step limit, on purpose. A flipped bit of a `next` pointer can:
 *   - leave the valid memory or break the alignment: crash
 *   - point to an earlier node of the same list: endless loop, so a timeout
 *   - point to a later node: nodes skipped, so a wrong sum
 * The 64 nodes of a list sit in one block of 1024 bytes, in shuffled order, so bits
 * 4-9 of `next` stay inside the list.
 *
 * Kernels and buffers:
 *   k_prepare  scale -> adj. Then scale is freed, to exercise a target that is
 *              freed; the next allocation (out) can reuse its address.
 *   k_walk     heads, nodes, adj -> out
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define N 64         // output is N x N; also CTAs and threads per CTA
#define LISTS (N * N)
#define LIST_LEN 64

typedef struct Node {
	struct Node *next;
	float value;
	float weight;
} Node;

__global__ void k_prepare(const float *scale, float *adj)
{
	const int id = blockIdx.x * N + threadIdx.x;
	adj[id] = 0.5f + 0.25f * scale[id];
}

__global__ void k_walk(Node *const *heads, const float *adj, float *out)
{
	const int id = blockIdx.x * N + threadIdx.x;
	const float a = adj[id];
	float sum = 0.f;
	for (const Node *p = heads[id]; p != nullptr; p = p->next) {
		sum += a * p->value * p->weight;
	}
	out[id] = sum;
}

static unsigned int next_rand(unsigned int *state)
{
	*state = *state * 1664525u + 1013904223u;
	return *state >> 16;
}

static float next_input(unsigned int *state)
{
	return (float)((int)(next_rand(state) % 2049u) - 1024) / 1024.f;
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
	static float host_scale[LISTS], host_out[LISTS];
	static Node host_nodes[LISTS * LIST_LEN];
	static Node *host_heads[LISTS];

	float *scale, *adj, *out;
	Node *nodes;
	Node **heads;
	check(cudaMalloc(&scale, sizeof(host_scale)), "malloc scale");
	check(cudaMalloc(&adj, sizeof(host_scale)), "malloc adj");
	check(cudaMalloc(&nodes, sizeof(host_nodes)), "malloc nodes");
	check(cudaMalloc(&heads, sizeof(host_heads)), "malloc heads");

	unsigned int state = 2024u;
	for (int i = 0; i < LISTS; i++) host_scale[i] = next_input(&state);
	for (int l = 0; l < LISTS; l++) {
		int order[LIST_LEN];
		for (int k = 0; k < LIST_LEN; k++) order[k] = k;
		for (int k = LIST_LEN - 1; k > 0; k--) {
			const int j = next_rand(&state) % (k + 1);
			const int t = order[k];
			order[k] = order[j];
			order[j] = t;
		}
		const int base = l * LIST_LEN;
		host_heads[l] = nodes + base + order[0];
		for (int k = 0; k < LIST_LEN; k++) {
			Node *n = &host_nodes[base + order[k]];
			n->next = k + 1 < LIST_LEN ? nodes + base + order[k + 1] : nullptr;
			n->value = next_input(&state);
			n->weight = 0.5f + 0.25f * next_input(&state);
		}
	}
	check(cudaMemcpy(scale, host_scale, sizeof(host_scale), cudaMemcpyHostToDevice), "copy scale");
	check(cudaMemcpy(nodes, host_nodes, sizeof(host_nodes), cudaMemcpyHostToDevice), "copy nodes");
	check(cudaMemcpy(heads, host_heads, sizeof(host_heads), cudaMemcpyHostToDevice), "copy heads");
	printf("PTR_CHASE lists=%d len=%d\n", LISTS, LIST_LEN);

	k_prepare<<<N, N>>>(scale, adj);
	check(cudaDeviceSynchronize(), "k_prepare");
	check(cudaFree(scale), "free scale");

	check(cudaMalloc(&out, sizeof(host_out)), "malloc out");
	k_walk<<<N, N>>>(heads, adj, out);
	check(cudaDeviceSynchronize(), "k_walk");
	check(cudaMemcpy(host_out, out, sizeof(host_out), cudaMemcpyDeviceToHost), "copy out");

	FILE *f = fopen("output_ptr.bin", "wb");
	if (!f || fwrite(host_out, sizeof(float), LISTS, f) != LISTS || fclose(f) != 0) {
		printf("Error: cannot write output_ptr.bin\n");
		exit(1);
	}
	check(cudaFree(adj), "free adj");
	check(cudaFree(nodes), "free nodes");
	check(cudaFree(heads), "free heads");
	check(cudaFree(out), "free out");
	printf("done\n");
	return 0;
}
