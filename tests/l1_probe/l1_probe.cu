/*
 * Does an atomicXor executed in L2 update the copies already in L1?
 *
 * All threads meet at a barrier made of atomics, read a word (fill), and wait
 * on their own clock, without touching memory. One thread flips the word with
 * atomicXor. Every thread then rereads it, counts old and new values per SM and
 * measures the read latency in cycles, which tells an L1 hit from a miss.
 * Results are stored only after the rereads, so the readers execute no atomic
 * and no fence between the fill and the rereads, unless the case asks for it.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CTAS 80
#define THREADS 32
#define MAX_SM 32
#define OLD_V 0x11111111u
#define MASK 0x00010000u
#define EVICT_WORDS (4u * 1024u * 1024u / 4u) // 4 MiB, far larger than L1
#define SPIN_CAP 20000000
#define REREADS 8
#define FLIP_DELAY 200000ull    // cycles after the barrier: all fills are done
#define READ_DELAY 400000ull    // cycles after the barrier: the flip is done

enum { F_FILL = 1, F_FLIP = 2, F_VOLATILE = 4, F_FENCE = 8, F_EVICT = 16, F_ATOMIC = 32, F_NEXT = 64 };
struct case_t { const char *name; int flags; };
static const case_t cases[] = {
	{"control_no_flip",        F_FILL},                    // old value, latency of an L1 hit
	{"no_fill",                F_FLIP},                    // no copy can exist: must read the new value
	{"fill",                   F_FILL | F_FLIP},           // the question
	{"fill_volatile",          F_FILL | F_FLIP | F_VOLATILE},
	{"fill_fence",             F_FILL | F_FLIP | F_FENCE},
	{"fill_atomic_by_reader",  F_FILL | F_FLIP | F_ATOMIC},
	{"fill_evict",             F_FILL | F_FLIP | F_EVICT},
	{"fill_next_kernel",       F_FILL | F_FLIP | F_NEXT},
};
#define NUM_CASES (int)(sizeof(cases) / sizeof(cases[0]))

struct result_t {
	unsigned old_reads[MAX_SM], new_reads[MAX_SM], other_reads[MAX_SM], threads[MAX_SM];
	unsigned long long lat_first, lat_second, lat_reread, n_first, n_second, n_reread;
	int flipper_sm;
	unsigned spin_timeouts;
	unsigned junk;
};

__device__ inline unsigned read_smid() {
	unsigned s;
	asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
	return s;
}

// Explicit PTX, so that the kind of load is exactly the one asked for
__device__ inline unsigned load_word(const unsigned *p, bool vol) {
	unsigned v;
	if (vol) asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
	else asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
	return v;
}

// The comparison forces the load to complete before the clock is read again
__device__ inline unsigned timed_load(const unsigned *p, bool vol, unsigned long long *lat, unsigned *junk) {
	unsigned long long t0 = clock64();
	unsigned v = load_word(p, vol);
	*junk += (v == 0xdeadbeefu);
	*lat = clock64() - t0;
	return v;
}

__global__ void first(const unsigned *word, unsigned *sync, const unsigned *big, result_t *r, int flags, bool count) {
	const unsigned sm = read_smid() % MAX_SM;
	const bool vol = flags & F_VOLATILE;
	unsigned junk = 0;
	unsigned long long l1 = 0, l2 = 0, lr = 0, l;
	unsigned o = 0, n = 0, x = 0;

	atomicAdd(&r->threads[sm], 1u);
	atomicAdd(&sync[0], 1u);
	int spins = 0;
	while (atomicAdd(&sync[0], 0u) < CTAS * THREADS && ++spins < SPIN_CAP) {}
	if (spins >= SPIN_CAP) atomicAdd(&r->spin_timeouts, 1u);
	const unsigned long long t0 = clock64();

	if (flags & F_FILL) {
		timed_load(word, vol, &l1, &junk);
		timed_load(word, vol, &l2, &junk);
	}
	if (flags & F_FLIP) {
		const bool flipper = blockIdx.x == 0 && threadIdx.x == 0;
		while (clock64() - t0 < FLIP_DELAY) {}
		if (flipper) {
			r->flipper_sm = sm;
			atomicXor((unsigned *)word, MASK);
		}
	}
	while (clock64() - t0 < READ_DELAY) {}

	if (flags & F_FENCE) __threadfence();
	if (flags & F_ATOMIC) atomicAdd(&sync[1], 1u);
	if (flags & F_EVICT) {
		for (unsigned i = (blockIdx.x * 7919u + threadIdx.x * 64u) % 1024u; i < EVICT_WORDS; i += 16) junk += big[i];
	}
	if (count) {
		for (int i = 0; i < REREADS; i++) {
			unsigned v = timed_load(word, vol, &l, &junk);
			lr += l;
			if (v == OLD_V) o++;
			else if (v == (OLD_V ^ MASK)) n++;
			else x++;
		}
		atomicAdd(&r->old_reads[sm], o);
		atomicAdd(&r->new_reads[sm], n);
		atomicAdd(&r->other_reads[sm], x);
		atomicAdd(&r->lat_reread, lr);
		atomicAdd(&r->n_reread, (unsigned long long)REREADS);
	}
	if (flags & F_FILL) {
		atomicAdd(&r->lat_first, l1);
		atomicAdd(&r->lat_second, l2);
		atomicAdd(&r->n_first, 1ull);
		atomicAdd(&r->n_second, 1ull);
	}
	atomicAdd(&r->junk, junk);
}

__global__ void second(const unsigned *word, result_t *r) {
	const unsigned sm = read_smid() % MAX_SM;
	unsigned o = 0, n = 0, x = 0, junk = 0;
	unsigned long long lr = 0, l;
	for (int i = 0; i < REREADS; i++) {
		unsigned v = timed_load(word, false, &l, &junk);
		lr += l;
		if (v == OLD_V) o++;
		else if (v == (OLD_V ^ MASK)) n++;
		else x++;
	}
	atomicAdd(&r->old_reads[sm], o);
	atomicAdd(&r->new_reads[sm], n);
	atomicAdd(&r->other_reads[sm], x);
	atomicAdd(&r->lat_reread, lr);
	atomicAdd(&r->n_reread, (unsigned long long)REREADS);
	atomicAdd(&r->junk, junk);
}

static void check(const char *what) {
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess) { printf("Error: %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}

int main(int argc, char **argv) {
	const int rounds = argc > 1 ? atoi(argv[1]) : 20;
	unsigned *word, *sync, *big;
	result_t *r;
	cudaMalloc(&word, 256);
	cudaMalloc(&sync, 256);
	cudaMalloc(&big, EVICT_WORDS * sizeof(unsigned));
	cudaMalloc(&r, sizeof(result_t));
	cudaMemset(big, 0, EVICT_WORDS * sizeof(unsigned));
	check("setup");
	const unsigned old_v = OLD_V, zero[8] = {0};

	printf("{\"rounds\": %d, \"ctas\": %d, \"threads\": %d, \"rereads_per_thread\": %d, \"cases\": {\n", rounds, CTAS, THREADS, REREADS);
	for (int c = 0; c < NUM_CASES; c++) {
		const int fl = cases[c].flags;
		unsigned long long o = 0, n = 0, x = 0, fo = 0, fn = 0, to = 0, lf = 0, ls = 0, lr = 0, nf = 0, ns = 0, nr = 0;
		unsigned sm_old = 0, sm_new = 0, used = 0;
		for (int i = 0; i < rounds; i++) {
			cudaMemcpy(word, &old_v, 4, cudaMemcpyHostToDevice);
			cudaMemcpy(sync, zero, sizeof(zero), cudaMemcpyHostToDevice);
			cudaMemset(r, 0, sizeof(result_t));
			first<<<CTAS, THREADS>>>(word, sync, big, r, fl, !(fl & F_NEXT));
			cudaDeviceSynchronize();
			if (fl & F_NEXT) { second<<<CTAS, THREADS>>>(word, r); cudaDeviceSynchronize(); }
			check(cases[c].name);
			result_t h;
			cudaMemcpy(&h, r, sizeof(h), cudaMemcpyDeviceToHost);
			for (int s = 0; s < MAX_SM; s++) {
				o += h.old_reads[s]; n += h.new_reads[s]; x += h.other_reads[s];
				if (h.old_reads[s]) sm_old |= 1u << s;
				if (h.new_reads[s]) sm_new |= 1u << s;
				if (h.threads[s]) used |= 1u << s;
			}
			if (fl & F_FLIP) { fo += h.old_reads[h.flipper_sm]; fn += h.new_reads[h.flipper_sm]; }
			to += h.spin_timeouts;
			lf += h.lat_first; ls += h.lat_second; lr += h.lat_reread;
			nf += h.n_first; ns += h.n_second; nr += h.n_reread;
		}
		printf("%s  \"%s\": {\"old\": %llu, \"new\": %llu, \"other\": %llu, \"flipper_sm_old\": %llu, \"flipper_sm_new\": %llu, "
			"\"sms_used\": %d, \"sms_old\": %d, \"sms_new\": %d, \"spin_timeouts\": %llu, "
			"\"cycles_first\": %.1f, \"cycles_second\": %.1f, \"cycles_reread\": %.1f}",
			c ? ",\n" : "", cases[c].name, o, n, x, fo, fn,
			__builtin_popcount(used), __builtin_popcount(sm_old), __builtin_popcount(sm_new), to,
			nf ? (double)lf / nf : 0.0, ns ? (double)ls / ns : 0.0, nr ? (double)lr / nr : 0.0);
	}
	printf("\n}}\n");
	return 0;
}
