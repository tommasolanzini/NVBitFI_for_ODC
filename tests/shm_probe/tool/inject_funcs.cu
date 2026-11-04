/*
 * Device code of the shared memory probe tool, see ../README.md.
 */

#include <stdint.h>
#include "utils/utils.h"
#include "probe.h"

// Addresses that NVBit passes for shared memory instructions
extern "C" __device__ __noinline__ void probe_record(uint64_t pstate, uint64_t addr, int slot, int pred) {
	if (!pred) return;
	probe_state_t *st = (probe_state_t *)pstate;
	const unsigned n = atomicAdd(&st->count[slot], 1u);
	if (n < PROBE_SAMPLES) st->addr[slot][n] = addr;
}

// Flip at the store to the trigger address: of the address of the store itself,
// or of a fixed offset of the shared memory of this CTA.
extern "C" __device__ __noinline__ void probe_flip(uint64_t pstate, uint64_t addr, int pred) {
	probe_state_t *st = (probe_state_t *)pstate;
	if (!pred || addr != st->trigger) return;
	if (atomicCAS(&st->fired, 0, 1) != 0) return;
	const uint64_t target = st->use_offset ? st->offset : addr;
	unsigned *p = (unsigned *)__cvta_shared_to_generic((unsigned)target);
	st->old_word = atomicXor(p, st->mask);
	st->new_word = *p;
	st->flip_addr = (uint64_t)p;
	st->flip_cta = blockIdx.x;
	st->flip_tid = threadIdx.x;
}

// The offset is set by the host before the kernel. A value written by a thread
// during the kernel could not be used here: a plain load can return a stale L1
// copy of it (tests/l1_probe).
extern "C" __device__ __noinline__ void probe_xor(uint64_t pstate, int plain, int pred) {
	if (!pred) return;
	probe_state_t *st = (probe_state_t *)pstate;
	unsigned *p = (unsigned *)__cvta_shared_to_generic((unsigned)st->offset);
	const unsigned m = 1u << (threadIdx.x % 32);
	if (plain) {
		const unsigned v = *p;
		*p = v ^ m;
	} else {
		atomicXor(p, m);
	}
}
