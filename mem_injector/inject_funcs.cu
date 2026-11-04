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
