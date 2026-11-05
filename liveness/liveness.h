/*
 * Register liveness: state shared by the host and the device code.
 *
 * A value lives from its write to its last read before the next write. Time is
 * counted on the thread (GPR, predicate registers) or on the warp (uniform registers,
 * uniform predicates), in executed instructions and in clock cycles.
 */
#ifndef LIVENESS_H
#define LIVENESS_H

#include <stdint.h>

#define LIVE_MAX_REGS 128   // GPR and UR numbers per kernel covered by the masks
#define LIVE_MAX_INSTRS 65536
#define LIVE_MAX_SITES 262144
#define LIVE_NUM_PRED 7     // P0-P6, UP0-UP6: PT and UPT are not storage

enum { LIVE_GPR = 0, LIVE_PR = 1, LIVE_UR = 2, LIVE_UP = 3, LIVE_CLASSES = 4 };

// One static instruction
typedef struct {
	uint64_t gprR[2], gprW[2], urR[2], urW[2];
	uint32_t siteG, siteP, siteU, siteUP; // first site of each class of destinations
	uint8_t prR, prW, upR, upW;
	int8_t guardPr, guardUp;              // -1: no guard of that kind
} live_instr_t;

// The value now in one register of one thread or warp
typedef struct {
	uint32_t wTime, rTime, wCyc, rCyc;
	uint32_t site;  // writer + 1, 0: not written yet
	uint32_t reads;
} live_slot_t;

// One destination of one static instruction: all the values it wrote
typedef struct {
	uint64_t count, live, liveCycles, reads, dead;
} live_site_t;

// Time of one thread or warp
typedef struct {
	uint32_t time;
	uint32_t sm;
	uint64_t firstCycle, lastCycle;
} live_time_t;

typedef struct {
	live_instr_t *instrs;
	live_site_t *sites;
	live_slot_t *threadSlots, *warpSlots;
	live_time_t *threadTime, *warpTime;
	uint32_t numGpr, numUr;
	uint32_t threadsPerCta, warpsPerCta;
	uint64_t readsNoWrite[LIVE_CLASSES]; // reads of a register no instruction wrote yet
	uint64_t laneMismatch;               // lane != thread % 32: the warp key is wrong
} live_ctx_t;

#endif
