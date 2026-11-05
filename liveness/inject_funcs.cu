/*
 * Device code of the liveness tool. No printf here: see CLAUDE.md, a tool module with
 * device printf does not load on this GPU.
 */
#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"
#include "liveness.h"

__inline__ __device__ void read_slot(live_ctx_t *c, live_slot_t *s, int cls, uint32_t t, uint32_t cyc) {
	if (s->site == 0) {
		atomicAdd((unsigned long long *)&c->readsNoWrite[cls], 1ULL);
		return;
	}
	s->rTime = t;
	s->rCyc = cyc;
	s->reads++;
}

// The write ends the previous value of the register: its numbers go to its site
__inline__ __device__ void write_slot(live_ctx_t *c, live_slot_t *s, uint32_t site, uint32_t t, uint32_t cyc) {
	if (s->site != 0) {
		live_site_t *o = &c->sites[s->site - 1];
		atomicAdd((unsigned long long *)&o->count, 1ULL);
		if (s->reads == 0) {
			atomicAdd((unsigned long long *)&o->dead, 1ULL);
		} else {
			atomicAdd((unsigned long long *)&o->live, (unsigned long long)(s->rTime - s->wTime));
			atomicAdd((unsigned long long *)&o->liveCycles, (unsigned long long)(uint32_t)(s->rCyc - s->wCyc));
			atomicAdd((unsigned long long *)&o->reads, (unsigned long long)s->reads);
		}
	}
	s->site = site + 1;
	s->wTime = t;
	s->wCyc = cyc;
	s->rTime = t;
	s->rCyc = cyc;
	s->reads = 0;
}

__inline__ __device__ void tick(live_time_t *tt, uint64_t now, uint32_t &t) {
	t = ++tt->time;
	if (t == 1) {
		tt->firstCycle = now;
		tt->sm = get_smid();
	}
	tt->lastCycle = now;
}

// Called BEFORE every instruction. Sources first: an instruction that reads and
// writes a register reads the old value.
extern "C" __device__ __noinline__ void live_track(uint64_t pctx, uint32_t idx, int pred) {
	live_ctx_t *c = (live_ctx_t *)pctx;
	const live_instr_t *in = &c->instrs[idx];
	const unsigned active = __activemask();
	const unsigned predMask = __ballot_sync(active, pred);
	const int lane = get_laneid();
	const uint64_t now = clock64();
	const uint32_t cyc = (uint32_t)now;

	const uint32_t cta = blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y)));
	const uint32_t local = threadIdx.x + (blockDim.x * (threadIdx.y + (threadIdx.z * blockDim.y)));
	if ((uint32_t)lane != local % 32) atomicAdd((unsigned long long *)&c->laneMismatch, 1ULL);

	// thread: GPR and predicate registers
	const uint32_t tid = cta * c->threadsPerCta + local;
	uint32_t t;
	tick(&c->threadTime[tid], now, t);
	live_slot_t *ts = &c->threadSlots[(uint64_t)tid * (c->numGpr + LIVE_NUM_PRED)];
	live_slot_t *tp = ts + c->numGpr;

	if (in->guardPr >= 0) read_slot(c, tp + in->guardPr, LIVE_PR, t, cyc);
	if (pred) {
		for (int w = 0; w < 2; w++) {
			uint64_t m = in->gprR[w];
			while (m) {
				const int b = __ffsll((long long)m) - 1;
				m &= m - 1;
				read_slot(c, ts + w * 64 + b, LIVE_GPR, t, cyc);
			}
		}
		for (uint32_t m = in->prR; m; m &= m - 1) read_slot(c, tp + __ffs(m) - 1, LIVE_PR, t, cyc);
		uint32_t site = in->siteG;
		for (int w = 0; w < 2; w++) {
			uint64_t m = in->gprW[w];
			while (m) {
				const int b = __ffsll((long long)m) - 1;
				m &= m - 1;
				write_slot(c, ts + w * 64 + b, site++, t, cyc);
			}
		}
		site = in->siteP;
		for (uint32_t m = in->prW; m; m &= m - 1) write_slot(c, tp + __ffs(m) - 1, site++, t, cyc);
	}

	// warp: uniform registers and uniform predicates, one lane does the work
	if (lane != __ffs(active) - 1) return;
	const uint32_t wid = cta * c->warpsPerCta + local / 32;
	uint32_t wt;
	tick(&c->warpTime[wid], now, wt);
	live_slot_t *ws = &c->warpSlots[(uint64_t)wid * (c->numUr + LIVE_NUM_PRED)];
	live_slot_t *wp = ws + c->numUr;

	if (in->guardUp >= 0) read_slot(c, wp + in->guardUp, LIVE_UP, wt, cyc);
	if (predMask == 0) return;
	for (int w = 0; w < 2; w++) {
		uint64_t m = in->urR[w];
		while (m) {
			const int b = __ffsll((long long)m) - 1;
			m &= m - 1;
			read_slot(c, ws + w * 64 + b, LIVE_UR, wt, cyc);
		}
	}
	for (uint32_t m = in->upR; m; m &= m - 1) read_slot(c, wp + __ffs(m) - 1, LIVE_UP, wt, cyc);
	uint32_t site = in->siteU;
	for (int w = 0; w < 2; w++) {
		uint64_t m = in->urW[w];
		while (m) {
			const int b = __ffsll((long long)m) - 1;
			m &= m - 1;
			write_slot(c, ws + w * 64 + b, site++, wt, cyc);
		}
	}
	site = in->siteUP;
	for (uint32_t m = in->upW; m; m &= m - 1) write_slot(c, wp + __ffs(m) - 1, site++, wt, cyc);
}
