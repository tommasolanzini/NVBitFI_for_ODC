/*
 * NVBitFI_for_ODC (2026): device code of the memory-data injector.
 *
 *  mem_count_flip counts the dynamic instructions of the target kernel. The
 *    thread that reaches the selected one flips the word with atomicXor.
 *  mem_trace_access checks every global load and store against the flipped
 *    byte. On an overlap with a load it reads the word just before the
 *    instruction, with a plain load: same path as the application, so it sees
 *    what the application sees, stale L1 copies included.
 *
 * The readers must not execute a fence or a volatile load: on this GPU both
 * bypass or invalidate L1 (tests/l1_probe). Atomics do not.
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"
#include "mem_injector.h"

__inline__ __device__ int flat_tid_in_cta() {
	return threadIdx.x + (blockDim.x * (threadIdx.y + (threadIdx.z * blockDim.y)));
}

__inline__ __device__ int flat_cta() {
	return blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y)));
}

extern "C" __device__ __noinline__ void mem_count_flip(uint64_t pinfo, uint64_t pshadow, uint64_t pcounter,
		int offset, int predicate) {
	mem_info_t *info = (mem_info_t *)pinfo;
	if (!info->armed) return;

	const unsigned active_mask = __activemask();
	const unsigned pred_mask = __ballot_sync(active_mask, predicate);
	if (!predicate) return;

	const int lane = get_laneid();
	const int leader = __ffs(pred_mask) - 1;

	// warp-aggregated counting: the leader reserves one index per thread
	unsigned long long base = 0;
	if (lane == leader) base = atomicAdd((unsigned long long *)pcounter, (unsigned long long)__popc(pred_mask));
	base = __shfl_sync(pred_mask, base, leader);
	const unsigned long long myIdx = base + __popc(pred_mask & ((1u << lane) - 1));

	if (myIdx != info->instID) return;

	volatile mem_shadow_t *rec = (volatile mem_shadow_t *)pshadow;
	rec->kernelIdx = info->kernelIdx;
	rec->dynInstr = (long long)myIdx;
	rec->pcOffset = offset;
	rec->tid = flat_tid_in_cta() + flat_cta() * (blockDim.x * blockDim.y * blockDim.z);
	rec->ctaFlat = flat_cta();
	rec->smID = get_smid();
	rec->warpID = get_warpid();
	rec->laneID = lane;
	// The fences only affect the SM of this thread, which sees the new value
	// anyway (tests/l1_probe): the stale copies of the other SMs survive.
	__threadfence_system();
	rec->state = MEM_SELECTED;
	__threadfence_system();

	// The readers wait while FLIP_RUNNING. The slow writes to host memory come
	// after FLIP_DONE, so that the readers do not wait for them.
	atomicExch(&info->fired, FLIP_RUNNING);
	const unsigned old = atomicXor((unsigned *)info->wordAddr, info->appliedMask);
	atomicExch(&info->oldWord, old);
	atomicExch(&info->fired, FLIP_DONE);

	rec->oldWord = old;
	rec->newWord = old ^ info->appliedMask;
	__threadfence_system();
	rec->state = MEM_INJECTED;
	__threadfence_system();
}

extern "C" __device__ __noinline__ void mem_trace_access(uint64_t pinfo, uint64_t pshadow, int predicate,
		uint64_t addr, int size, int flags) {
	mem_info_t *info = (mem_info_t *)pinfo;
	if (!predicate || !info->traceOn) return;
	const uint64_t flip = info->flipByteAddr;
	if (addr > flip || addr + size <= flip) return;

	mem_shadow_t *rec = (mem_shadow_t *)pshadow;
	const unsigned smBit = 1u << (get_smid() & 31);

	// The application load comes right after this call. Everything slow goes
	// before the check of the flip: a load that returns here as "before the
	// flip" must really run before it. Reads before the flip = loads - old - new.
	if (flags & MEM_FLAG_LOAD) {
		atomicAdd(&info->loads, 1u);
		atomicOr(&info->smLoads, smBit);
	}
	int fired = atomicAdd(&info->fired, 0);
	for (int spins = 0; fired == FLIP_RUNNING && spins < 100000; spins++) fired = atomicAdd(&info->fired, 0);
	if (fired == FLIP_NOT_YET) return;

	if (flags & MEM_FLAG_LOAD) {
		const unsigned v = *(const unsigned *)info->wordAddr;
		const unsigned old = atomicAdd(&info->oldWord, 0u);
		atomicCAS(&rec->firstAccess, ACCESS_NONE, ACCESS_READ);
		if ((v ^ old) & info->mask) {
			atomicAdd(&rec->readsNew, 1u);
			atomicOr(&rec->smNew, smBit);
		} else {
			atomicAdd(&rec->readsOld, 1u);
			atomicOr(&rec->smOld, smBit);
		}
		const int k = info->kernelIdx;
		atomicOr(&rec->readKernels, 1ull << (k < 63 ? k : 63));
	}
	if (flags & MEM_FLAG_STORE) {
		atomicCAS(&rec->firstAccess, ACCESS_NONE, ACCESS_WRITE);
		atomicAdd(&rec->writesAfter, 1u);
	}
}

// ---- Shared memory -------------------------------------------------------
// A shared address is relative to the window of the CTA of the thread that uses
// it: only a thread of a CTA can reach its window. atomicXor compiles to ATOMS,
// atomic against the other threads of the CTA (tests/shm_probe).

// Count phase: dynamic stores to shared memory, the index of mode s
extern "C" __device__ __noinline__ void shm_count_store(uint64_t pcounter, int predicate) {
	const unsigned pred_mask = __ballot_sync(__activemask(), predicate);
	if (!predicate || get_laneid() != __ffs(pred_mask) - 1) return;
	atomicAdd((unsigned long long *)pcounter, (unsigned long long)__popc(pred_mask));
}

// Count phase: first and last instruction of each CTA, on the instruction clock
extern "C" __device__ __noinline__ void shm_residency(uint64_t ptable, uint64_t pcounter, int predicate) {
	const unsigned pred_mask = __ballot_sync(__activemask(), predicate);
	if (!predicate || get_laneid() != __ffs(pred_mask) - 1) return;
	const int cta = flat_cta();
	if (cta >= MEM_MAX_CTAS) return;
	mem_cta_t *e = (mem_cta_t *)ptable + cta;
	const unsigned long long now = atomicAdd((unsigned long long *)pcounter, 0ull) + 1;
	atomicCAS(&e->first, 0ull, now);
	atomicMax(&e->last, now);
	e->sm = get_smid();
}

__inline__ __device__ void shm_do_flip(mem_info_t *info, mem_shadow_t *r, unsigned word_off, unsigned byte_off,
		unsigned mask, unsigned long long idx, int offset, int lane) {
	volatile mem_shadow_t *rec = (volatile mem_shadow_t *)r;
	rec->kernelIdx = info->kernelIdx;
	rec->dynInstr = (long long)idx;
	rec->pcOffset = offset;
	rec->tid = flat_tid_in_cta() + flat_cta() * (blockDim.x * blockDim.y * blockDim.z);
	rec->ctaFlat = flat_cta();
	rec->smID = get_smid();
	rec->warpID = get_warpid();
	rec->laneID = lane;
	rec->shmWordOffset = word_off;
	rec->shmMask = mask;
	__threadfence_system();
	rec->state = MEM_SELECTED;
	__threadfence_system();

	atomicExch(&info->wordAddr, (unsigned long long)word_off);
	atomicExch(&info->flipByteAddr, (unsigned long long)byte_off);
	atomicExch(&info->mask, mask);
	atomicExch(&info->flipCta, flat_cta());
	atomicExch(&info->fired, FLIP_RUNNING);
	const unsigned applied = info->applyFlip ? mask : 0u;
	const unsigned old = atomicXor((unsigned *)__cvta_shared_to_generic(word_off), applied);
	atomicExch(&info->oldWord, old);
	atomicExch(&info->fired, FLIP_DONE);

	rec->oldWord = old;
	rec->newWord = old ^ applied;
	__threadfence_system();
	rec->state = MEM_INJECTED;
	__threadfence_system();
}

// Mode s: right after the N-th dynamic store to shared memory, flip one of the
// bits just stored
extern "C" __device__ __noinline__ void shm_flip_after_store(uint64_t pinfo, uint64_t pshadow, uint64_t pcounter,
		int offset, int predicate, uint64_t addr, int size) {
	mem_info_t *info = (mem_info_t *)pinfo;
	if (!info->armed) return;
	const unsigned pred_mask = __ballot_sync(__activemask(), predicate);
	if (!predicate) return;
	const int lane = get_laneid();
	const int leader = __ffs(pred_mask) - 1;
	unsigned long long base = 0;
	if (lane == leader) base = atomicAdd((unsigned long long *)pcounter, (unsigned long long)__popc(pred_mask));
	base = __shfl_sync(pred_mask, base, leader);
	const unsigned long long myIdx = base + __popc(pred_mask & ((1u << lane) - 1));
	if (myIdx != info->instID) return;

	const unsigned bit = info->bitSeed % ((unsigned)size * 8);
	const unsigned byte_off = (unsigned)addr + bit / 8;
	shm_do_flip(info, (mem_shadow_t *)pshadow, byte_off & ~3u, byte_off, 1u << ((byte_off & 3) * 8 + bit % 8), myIdx, offset, lane);
}

// Mode u: time = N-th dynamic instruction of the kernel. The first warp on the
// target SM that executes an instruction after it flips a word of its own CTA.
extern "C" __device__ __noinline__ void shm_flip_uniform(uint64_t pinfo, uint64_t pshadow, uint64_t pcounter,
		int offset, int predicate) {
	mem_info_t *info = (mem_info_t *)pinfo;
	if (!info->armed) return;
	const unsigned pred_mask = __ballot_sync(__activemask(), predicate);
	if (!predicate) return;
	const int lane = get_laneid();
	if (lane != __ffs(pred_mask) - 1) return;
	const unsigned long long base = atomicAdd((unsigned long long *)pcounter, (unsigned long long)__popc(pred_mask));
	if (base < info->instID || (int)get_smid() != info->targetSM) return;
	if (atomicCAS(&info->claimed, 0, 1) != 0) return;
	const unsigned word_off = (unsigned)info->flipOffsetTarget;
	shm_do_flip(info, (mem_shadow_t *)pshadow, word_off, word_off + info->bitSeed / 8, 1u << info->bitSeed, base, offset, lane);
}

// After a store. If the flip came while the store was between its check in
// shm_trace_access and its execution, the check did not see it: without this,
// the data of the store would later be read as the flipped bit.
extern "C" __device__ __noinline__ void shm_trace_store_after(uint64_t pinfo, uint64_t pshadow, int predicate,
		uint64_t addr, int size) {
	mem_info_t *info = (mem_info_t *)pinfo;
	if (!predicate || !info->traceOn) return;
	if (atomicAdd(&info->fired, 0) != FLIP_DONE) return;
	if (flat_cta() != atomicAdd(&info->flipCta, 0)) return;
	const unsigned long long flip = atomicAdd(&info->flipByteAddr, 0ull);
	if (addr > flip || addr + size <= flip) return;
	atomicCAS(&((mem_shadow_t *)pshadow)->firstAccess, ACCESS_NONE, ACCESS_AMBIGUOUS);
}

// The readers of the flipped word: same CTA only. Shared memory has no L1 copy,
// so all of them should see the new value; this measures it.
extern "C" __device__ __noinline__ void shm_trace_access(uint64_t pinfo, uint64_t pshadow, int predicate,
		uint64_t addr, int size, int flags) {
	mem_info_t *info = (mem_info_t *)pinfo;
	if (!predicate || !info->traceOn) return;
	int fired = atomicAdd(&info->fired, 0);
	for (int spins = 0; fired == FLIP_RUNNING && spins < 100000; spins++) fired = atomicAdd(&info->fired, 0);
	if (fired != FLIP_DONE) return;
	if (flat_cta() != atomicAdd(&info->flipCta, 0)) return;
	const unsigned long long flip = atomicAdd(&info->flipByteAddr, 0ull);
	if (addr > flip || addr + size <= flip) return;

	mem_shadow_t *rec = (mem_shadow_t *)pshadow;
	const unsigned smBit = 1u << (get_smid() & 31);
	// After a store to the byte the bit no longer comes from the flip: comparing
	// it would count data of the application as old or new reads.
	const int first = atomicAdd(&rec->firstAccess, 0);
	if ((flags & MEM_FLAG_LOAD) && first != ACCESS_WRITE && first != ACCESS_AMBIGUOUS && atomicAdd(&rec->writesAfter, 0u) == 0) {
		const unsigned word = (unsigned)atomicAdd(&info->wordAddr, 0ull);
		const unsigned v = *(const unsigned *)__cvta_shared_to_generic(word);
		const unsigned old = atomicAdd(&info->oldWord, 0u);
		const unsigned mask = atomicAdd(&info->mask, 0u);
		atomicCAS(&rec->firstAccess, ACCESS_NONE, ACCESS_READ);
		if ((v ^ old) & mask) {
			atomicAdd(&rec->readsNew, 1u);
			atomicOr(&rec->smNew, smBit);
		} else {
			atomicAdd(&rec->readsOld, 1u);
			atomicOr(&rec->smOld, smBit);
		}
		const int k = info->kernelIdx;
		atomicOr(&rec->readKernels, 1ull << (k < 63 ? k : 63));
	}
	if (flags & MEM_FLAG_STORE) {
		atomicCAS(&rec->firstAccess, ACCESS_NONE, ACCESS_WRITE);
		atomicAdd(&rec->writesAfter, 1u);
	}
}
