/*
 * Copyright 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * NVBitFI_for_ODC (2026): injection split in two calls.
 *
 *  inject_select (IPOINT_BEFORE) counts the injectable dynamic instances of
 *    the selected group exactly like the profiler (same point, same guard
 *    predicate, same eligibility) and marks the selected (thread, PC).
 *    Counting AFTER the instruction, as the original did, miscounts
 *    instructions that overwrite their own guard predicate (@P0 ISETP P0, ...)
 *    and never runs for EXIT/branches.
 *  inject_error (IPOINT_AFTER) corrupts one destination register of the
 *    selected instance: GPR, predicate (P0-P6), uniform register or uniform
 *    predicate, using the NVBit 1.8 register read/write device API.
 *  inject_addr_flip and inject_addr_restore: the address model, G_ADDR. A source
 *    register of the address is corrupted for that instruction only.
 */

#include <stdint.h>
#include <stdio.h>

#include "nvbit_reg_rw.h"
#include "utils/utils.h"
#include "injector.h"
#include "arch.h"

__inline__ __device__ int get_flat_tid_in_cta() {
	return threadIdx.x + (blockDim.x * (threadIdx.y + (threadIdx.z * blockDim.y)));
}

__inline__ __device__ int get_flat_cta() {
	return blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y)));
}

__inline__ __device__ int get_flat_tid() {
	return get_flat_tid_in_cta() + get_flat_cta() * (blockDim.x * blockDim.y * blockDim.z);
}

// get_laneid(), get_smid(), get_warpid() come from NVBit's utils/utils.h

// k ones starting at bit start
__inline__ __device__ unsigned int ones_at(int k, int start) {
	return (k >= 32 ? 0xffffffffu : ((1u << k) - 1)) << start;
}

// A second seed from the bits of bitIDSeed that did not pick the bit: uniform, and
// independent of the bit
__inline__ __device__ float second_seed(float bitIDSeed) {
	const float x = 32.0f * bitIDSeed;
	return x - (float)(int)x;
}

// XOR mask for a 32-bit register. The old value will be XORed with it.
__inline__ __device__ unsigned int get_mask(uint32_t bitFlipModel, int k, float bitIDSeed, unsigned int oldVal) {
	int b1 = min((int)(32 * bitIDSeed), 31);
	int b2 = min((int)(31 * bitIDSeed), 30);
	if (bitFlipModel == MCU_ADJ_BITS) return ones_at(k, min((int)((33 - k) * bitIDSeed), 32 - k));
	if (bitFlipModel == MCU_ADJ_REGS || bitFlipModel == MCU_ADJ_LANES) return 1u << b1;
	return (bitFlipModel == FLIP_SINGLE_BIT) * (1u << b1) +
		(bitFlipModel == FLIP_TWO_BITS) * (3u << b2) +
		(bitFlipModel == RANDOM_VALUE)  * (unsigned int)(((unsigned int)-1) * bitIDSeed) +
		(bitFlipModel == ZERO_VALUE) * oldVal;
}

// Predicate register, multi-cell: k adjacent bits of P0-P6 that include bit p
__inline__ __device__ unsigned int pred_window(int k, int p, float seed) {
	if (k > 7) k = 7;
	const int low = max(0, p - k + 1), high = min(p, 7 - k);
	return ones_at(k, low + min((int)((high - low + 1) * seed), high - low));
}

__inline__ __device__ void record_cell(volatile inj_shadow_t *rec, int idx, int regNo, int lane, unsigned int mask, unsigned int before, unsigned int after) {
	volatile inj_cell_t *c = &rec->cells[idx];
	c->regNo = regNo;
	c->lane = lane;
	c->tid = get_flat_tid();
	c->mask = mask;
	c->before = before;
	c->after = after;
	__threadfence_system();
}

// New value of the whole predicate register for a destination p. Flip models
// invert the bit (a single bit has no "two adjacent bits"), RANDOM_VALUE draws
// 0/1, ZERO_VALUE clears it, MCU_ADJ_BITS and MCU_ADJ_REGS flip k adjacent bits.
__inline__ __device__ unsigned int corrupt_pred(uint32_t bitFlipModel, int k, float bitIDSeed, unsigned int before, int p) {
	if (bitFlipModel == MCU_ADJ_BITS || bitFlipModel == MCU_ADJ_REGS) return before ^ pred_window(k, p, bitIDSeed);
	unsigned int bit = ((before >> p) & 1u) ^ 1u;
	if (bitFlipModel == RANDOM_VALUE) bit = bitIDSeed >= 0.5f ? 1u : 0u;
	if (bitFlipModel == ZERO_VALUE) bit = 0u;
	return (before & ~(1u << p)) | (bit << p);
}

// k-th set bit (0-based) of mask
__inline__ __device__ int nth_set_bit(uint32_t mask, int k) {
	for (int b = 0; b < 32; b++) {
		if (mask >> b & 1) { if (k == 0) return b; k--; }
	}
	return -1;
}

// Store what is about to be injected in host memory before the register is
// corrupted, so that the record survives whatever the corruption causes.
__inline__ __device__ void record_injection(volatile inj_shadow_t *rec, int regType, int regNo,
		unsigned int mask, unsigned int before, unsigned int after) {
	rec->regType = regType;
	rec->regNo = regNo;
	rec->mask = mask;
	rec->beforeVal = before;
	rec->afterVal = after;
	__threadfence_system();
}

// ---------------------------------------------------------------------------
// BEFORE: count and select
// ---------------------------------------------------------------------------
extern "C" __device__ __noinline__ void inject_select(uint64_t piinfo, uint64_t pshadow, uint64_t pcounters,
		int offset, int opcode, int grp, int flags, int predicate) {
	inj_info_t* inj_info = (inj_info_t*)piinfo;
	uint64_t *counters = (uint64_t *)pcounters;

	if (!inj_info->areParamsReady) return; // not the kernel of interest (keep counting after the selection: groupInstrs must equal the profile)

	// Same eligibility rules as the profiler (count_instrs)
	const int igid = inj_info->groupID;
	bool elig;
	int ctr;
	if (igid == G_GPPR) { elig = flags & FLAG_ELIG_GPPR; ctr = NUM_OPCODE_COUNTERS + G_GPPR; }
	else if (igid == G_GP) { elig = flags & FLAG_ELIG_GP; ctr = NUM_OPCODE_COUNTERS + G_GP; }
	else if (igid == G_ADDR) { elig = flags & FLAG_ELIG_ADDR; ctr = NUM_OPCODE_COUNTERS + G_ADDR; }
	else if (igid == G_ADDR_STORE) { elig = flags & FLAG_ELIG_ADDR_STORE; ctr = NUM_OPCODE_COUNTERS + G_ADDR_STORE; }
	else { elig = (grp == igid) && (flags & FLAG_ELIG_OWN); ctr = NUM_OPCODE_COUNTERS + igid; }
	if (!elig) return; // static property: the whole warp returns together

	const unsigned active_mask = __activemask();
	const unsigned pred_mask = __ballot_sync(active_mask, predicate);
	if (!predicate) return; // predicated-out threads do not execute the instruction

	const int lane = get_laneid();
	const int leader = __ffs(pred_mask) - 1;
	const bool uniform = flags & FLAG_UNIFORM;
	const unsigned long long n = uniform ? 1ULL : (unsigned long long)__popc(pred_mask);

	// warp-aggregated counting: the leader reserves n indices, every thread
	// takes base + its rank among the predicate-true lanes
	unsigned long long base = 0;
	if (lane == leader) base = atomicAdd((unsigned long long *)&counters[ctr], n);
	base = __shfl_sync(pred_mask, base, leader);

	if (uniform && lane != leader) return; // a uniform instruction is one instance per warp
	const unsigned long long myIdx = base + (uniform ? 0 : __popc(pred_mask & ((1u << lane) - 1)));

	if (myIdx == inj_info->instID) {
		volatile inj_shadow_t *rec = (volatile inj_shadow_t *)pshadow;
		rec->opcode = opcode;
		rec->pcOffset = offset;
		rec->tid = get_flat_tid();
		rec->ctaFlat = get_flat_cta();
		rec->smID = get_smid();
		rec->warpID = get_warpid();
		rec->laneID = lane;
		// Lanes of the model: a window of k lanes that holds this one, restricted to
		// those running now. An inactive lane does not run tool code, so it cannot
		// be corrupted: the record has the lanes asked and the lanes written.
		unsigned int laneMask = 1u << lane;
		rec->requestedCells = 1;
		if (inj_info->bitFlipModel == MCU_ADJ_LANES) {
			const int k = inj_info->mcuK;
			const int low = max(0, lane - k + 1), high = min(lane, 32 - k);
			const int start = low + min((int)((high - low + 1) * second_seed(inj_info->bitIDSeed)), high - low);
			laneMask = ones_at(k, start) & active_mask;
			rec->requestedCells = k;
		}
		__threadfence_system(); // the fields must reach host memory before the state
		rec->state = SHADOW_SELECTED;
		__threadfence_system();

		inj_info->selPcOffset = offset;
		inj_info->selCtaFlat = get_flat_cta();
		inj_info->selTidInCta = get_flat_tid_in_cta();
		inj_info->selWarp = get_warpid();
		inj_info->selLane = lane;
		inj_info->selLaneMask = laneMask;
		__threadfence();
		inj_info->selected = true;
	}
}

// nvbit_write_upred_reg() sets R5 of the calling thread to 0 (measured, always R5, see
// README). The other GPRs are compared too, because nothing documents this. R1 stays out:
// it is the stack pointer of the tool code and it is never written from here.
#define MAX_GPRS_KEPT 64
__inline__ __device__ void write_upred_keep_gprs(unsigned int value, int maxGpr) {
	unsigned int saved[MAX_GPRS_KEPT];
	const int n = maxGpr < MAX_GPRS_KEPT ? maxGpr : MAX_GPRS_KEPT;
	for (int q = 0; q < n; q++) if (q != STACK_POINTER_GPR) saved[q] = nvbit_read_reg((uint64_t)q);
	nvbit_write_upred_reg((int32_t)value);
	for (int q = 0; q < n; q++) {
		if (q != STACK_POINTER_GPR && (unsigned int)nvbit_read_reg((uint64_t)q) != saved[q]) nvbit_write_reg((uint64_t)q, saved[q]);
	}
}

// DUMMY=1: nothing is written. DUMMY=2: the unchanged value is written back, to test
// the write path too. In both cases the values stay the original ones.
#define WRITE_REGS (DUMMY != 1)

// ---------------------------------------------------------------------------
// AFTER: corrupt one destination of the selected instance
// ---------------------------------------------------------------------------
extern "C" __device__ __noinline__ void inject_error(uint64_t piinfo, uint64_t pshadow, uint64_t pverbose_device,
		int offset, uint32_t gprRange, uint32_t prMask, uint32_t urRange, uint32_t upMask, int maxGpr, int maxUr, int skipGpr, int regval, int upredValue) {
	inj_info_t* inj_info = (inj_info_t*)piinfo;
	volatile inj_shadow_t *rec = (volatile inj_shadow_t *)pshadow;
	uint32_t verbose_device = *((uint32_t *)pverbose_device);

	if (!inj_info->selected || inj_info->errorInjected) return;
	if (offset != inj_info->selPcOffset) return;
	if (get_flat_cta() != inj_info->selCtaFlat) return;
	const uint32_t model = inj_info->bitFlipModel;
	const int k = inj_info->mcuK;
	const int lane = get_laneid();
	const unsigned int laneMask = inj_info->selLaneMask;
	if (model == MCU_ADJ_LANES) {
		// Lockstep: these lanes run this call together, so all of them see
		// errorInjected false above and set it below.
		if ((int)get_warpid() != inj_info->selWarp || !((laneMask >> lane) & 1u)) return;
	} else if (get_flat_tid_in_cta() != inj_info->selTidInCta) {
		return;
	}
	const bool leader = (lane == inj_info->selLane); // the selected thread writes the common fields
	const int laneCell = (model == MCU_ADJ_LANES) ? __popc(laneMask & ((1u << lane) - 1)) : 0;

	const int igid = inj_info->groupID;
	const int gBase = gprRange >> 8, nG = gprRange & 0xff;
	const int uBase = urRange >> 8, nU = urRange & 0xff;
	int nP = __popc(prMask), nUP = __popc(upMask);
	// candidate destinations for the selected group
	int candG = nG, candP = nP, candU = nU, candUP = nUP;
	if (igid == G_GP) { candP = 0; candU = 0; candUP = 0; }
	else if (igid == G_GPPR) { candU = 0; candUP = 0; }
	const int total = candG + candP + candU + candUP;

	inj_info->debug[0] = 1;

	if (total == 0) { // should not happen: eligibility guarantees a destination
		inj_info->errorInjected = true; // the record stays SHADOW_SELECTED: nothing injected
		return;
	}

	int d = min((int)(total * inj_info->opIDSeed), total - 1);
	int numCells = 1;
	float seed = inj_info->bitIDSeed;

	if (d < candG) {                                   // ---- GPR ----
		int r = gBase + d;
		if (verbose_device) {
			inj_info->debug[1] = regval;                // value passed by NVBit for the first dest
			inj_info->debug[2] = nvbit_read_reg((uint64_t)gBase);
			inj_info->debug[3] = gBase;
		}
		unsigned int before = nvbit_read_reg((uint64_t)r);
		unsigned int mask = get_mask(model, k, seed, before);
		// skipGpr: the stack pointer of an application that never reads it. The flip stays
		// in the record and in the denominator, but it cannot change a real run.
		unsigned int after = (DUMMY || r == skipGpr) ? before : (before ^ mask);
		if (leader) record_injection(rec, REG_GPR, r, mask, before, after);
		record_cell(rec, laneCell, r, lane, mask, before, after);
		if (WRITE_REGS && r != skipGpr) nvbit_write_reg((uint64_t)r, after);
		if (model == MCU_ADJ_REGS) {
			rec->requestedCells = k;
			for (int j = 1; j < k && r + j < maxGpr; j++) {
				const unsigned int b = nvbit_read_reg((uint64_t)(r + j));
				const unsigned int a = (DUMMY || r + j == skipGpr) ? b : (b ^ mask);
				record_cell(rec, j, r + j, lane, mask, b, a);
				if (WRITE_REGS && r + j != skipGpr) nvbit_write_reg((uint64_t)(r + j), a);
				numCells++;
			}
		}
	} else if ((d -= candG) < candP) {                 // ---- predicate P0-P6 ----
		int p = nth_set_bit(prMask, d);
		unsigned int before = (unsigned int)nvbit_read_pred_reg();
		unsigned int after = corrupt_pred(model, k, seed, before, p);
		unsigned int shown = DUMMY ? before : after;
		if (leader) record_injection(rec, REG_PR, p, before ^ after, before, shown);
		record_cell(rec, laneCell, p, lane, before ^ after, before, shown);
		if (WRITE_REGS) nvbit_write_pred_reg((int32_t)shown);
	} else if ((d -= candP) < candU) {                 // ---- uniform register ----
		int r = uBase + d;
		unsigned int before = nvbit_read_ureg((uint64_t)r);
		unsigned int mask = get_mask(model, k, seed, before);
		unsigned int after = DUMMY ? before : (before ^ mask);
		record_injection(rec, REG_UR, r, mask, before, after);
		record_cell(rec, 0, r, lane, mask, before, after);
		if (WRITE_REGS) nvbit_write_ureg((uint64_t)r, after);
		if (model == MCU_ADJ_REGS) {
			rec->requestedCells = k;
			for (int j = 1; j < k && r + j < maxUr; j++) {
				const unsigned int b = nvbit_read_ureg((uint64_t)(r + j));
				const unsigned int a = DUMMY ? b : (b ^ mask);
				record_cell(rec, j, r + j, lane, mask, b, a);
				if (WRITE_REGS) nvbit_write_ureg((uint64_t)(r + j), a);
				numCells++;
			}
		}
	} else {                                           // ---- uniform predicate UP0-UP6 ----
		d -= candU;
		int p = nth_set_bit(upMask, d);
		unsigned int before = (unsigned int)upredValue; // not nvbit_read_upred_reg(): it changes the run, see README
		unsigned int after = corrupt_pred(model, k, seed, before, p);
		unsigned int shown = DUMMY ? before : after;
		record_injection(rec, REG_UP, p, before ^ after, before, shown);
		record_cell(rec, 0, p, lane, before ^ after, before, shown);
		if (WRITE_REGS) write_upred_keep_gprs(shown, maxGpr);
	}
	if (leader) {
		rec->numCells = (model == MCU_ADJ_LANES) ? __popc(laneMask) : numCells;
		__threadfence_system();
		rec->state = SHADOW_INJECTED;
		__threadfence_system();
	}
	inj_info->errorInjected = true;
	// no device-side printf: it makes the tool fail to load with NVBit 1.8 on sm_120
}

// ---------------------------------------------------------------------------
// Address model. Generic over the register class and the width: it picks one
// register and one bit, so another source operand needs only a new way to list
// the candidates.
// ---------------------------------------------------------------------------
__inline__ __device__ bool is_selected_thread(inj_info_t *inj_info, int offset) {
	return inj_info->selected && offset == inj_info->selPcOffset &&
		get_flat_cta() == inj_info->selCtaFlat && get_flat_tid_in_cta() == inj_info->selTidInCta;
}

__inline__ __device__ unsigned int addr_read(int kind, int r) {
	return kind == AP_RA ? (unsigned int)nvbit_read_reg((uint64_t)r) : (unsigned int)nvbit_read_ureg((uint64_t)r);
}

__inline__ __device__ void addr_write(int kind, int r, unsigned int v) {
	if (kind == AP_RA) nvbit_write_reg((uint64_t)r, v);
	else nvbit_write_ureg((uint64_t)r, v);
}

// BEFORE, after inject_select. addr stays before the 32-bit arguments: after nine of
// them NVBit passed only its high word. raMeta: bit 0 = .64, bit 1 = .U32, bits 8 up = stride.
// urMeta: bit 0 = .64.
extern "C" __device__ __noinline__ void inject_addr_flip(uint64_t piinfo, uint64_t pshadow, uint64_t addr, int offset,
		uint32_t partRa, uint32_t partUr, uint32_t partDesc, uint32_t raMeta, uint32_t urMeta, int access,
		uint32_t gprRange, uint32_t urRange) {
	inj_info_t *inj_info = (inj_info_t *)piinfo;
	if (inj_info->errorInjected || !is_selected_thread(inj_info, offset)) return;
	volatile inj_shadow_t *rec = (volatile inj_shadow_t *)pshadow;

	const int sigRa = partRa & 0xff, sigUr = partUr & 0xff, sigDesc = partDesc & 0xff;
	const int total = sigRa + sigUr + sigDesc;
	int bit = min((int)(total * inj_info->bitIDSeed), total - 1);
	int kind, base;
	if (bit < sigRa) { kind = AP_RA; base = partRa >> 8; }
	else if (bit < sigRa + sigUr) { kind = AP_UR; base = partUr >> 8; bit -= sigRa; }
	else { kind = AP_DESC; base = partDesc >> 8; bit -= sigRa + sigUr; }

	const int r = base + bit / 32;
	const unsigned int mask = 1u << (bit % 32);
	const unsigned int before = addr_read(kind, r);
	const unsigned int after = before ^ mask;

	// The address with the flip, as gen_mref_addr in nvbit_tool.h computes it
	long long delta = 0;
	if (kind == AP_RA) {
		const long long stride = (long long)(raMeta >> 8);
		if (raMeta & 1) {
			delta = bit < 32 ? ((long long)(unsigned long long)after - (long long)(unsigned long long)before) * stride
				: (long long)(((unsigned long long)after - (unsigned long long)before) << 32);
		} else if (raMeta & 2) {
			delta = ((long long)(unsigned long long)after - (long long)(unsigned long long)before) * stride;
		} else {
			delta = (long long)(int)(after * (unsigned int)stride) - (long long)(int)(before * (unsigned int)stride);
		}
	} else if (kind == AP_UR) {
		delta = (urMeta & 1) ? (bit < 32 ? (long long)(unsigned long long)after - (long long)(unsigned long long)before
				: (long long)(((unsigned long long)after - (unsigned long long)before) << 32))
			: (long long)(int)after - (long long)(int)before;
	}
	const unsigned long long corrupted = addr + (unsigned long long)delta;

	// The instruction writes this register: its value afterwards is the result
	bool isDest;
	if (kind == AP_RA) isDest = r >= (int)(gprRange >> 8) && r < (int)(gprRange >> 8) + (int)(gprRange & 0xff);
	else isDest = r >= (int)(urRange >> 8) && r < (int)(urRange >> 8) + (int)(urRange & 0xff);

	const int regType = kind == AP_RA ? REG_GPR : REG_UR;
	const unsigned int shown = DUMMY ? before : after;
	rec->addrOrig = addr;
	rec->addrNew = corrupted;
	rec->addrPart = kind;
	rec->addrBit = bit;
	rec->memAccess = access;
	rec->requestedCells = 1;
	rec->numCells = 1;
	record_cell(rec, 0, r, get_laneid(), mask, before, shown);
	record_injection(rec, regType, r, mask, before, shown);
	rec->state = SHADOW_INJECTED;
	__threadfence_system();

	inj_info->addrKind = kind;
	inj_info->addrReg = r;
	inj_info->addrOld = before;
	inj_info->addrRestore = !isDest;
	inj_info->addrDone = false;
	inj_info->errorInjected = true;
	if (WRITE_REGS) addr_write(kind, r, shown);
}

// AFTER: back to the original value, unless the instruction wrote that register
extern "C" __device__ __noinline__ void inject_addr_restore(uint64_t piinfo, uint64_t pshadow, int offset) {
	inj_info_t *inj_info = (inj_info_t *)piinfo;
	if (!inj_info->errorInjected || inj_info->addrDone || !is_selected_thread(inj_info, offset)) return;
	volatile inj_shadow_t *rec = (volatile inj_shadow_t *)pshadow;
	inj_info->addrDone = true;
	if (inj_info->addrRestore) {
		if (WRITE_REGS) addr_write(inj_info->addrKind, inj_info->addrReg, inj_info->addrOld);
		rec->addrRestored = 1;
	} else {
		rec->addrOverwritten = 1;
	}
	__threadfence_system();
}
