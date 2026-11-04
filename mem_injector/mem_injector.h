/*
 * NVBitFI_for_ODC (2026): data shared by the host and device parts of the
 * memory-data injector. See README.md in this directory.
 */

#ifndef MEM_INJECTOR_H
#define MEM_INJECTOR_H

#include <stdint.h>

// Progress of the injection, stored in mem_shadow_t::state
enum { MEM_NONE = 0, MEM_SELECTED = 1, MEM_INJECTED = 2 };

// Progress of the flip seen by the readers, stored in mem_info_t::fired
enum { FLIP_NOT_YET = 0, FLIP_RUNNING = 1, FLIP_DONE = 2 };

// First access to the flipped byte after the flip, stored in firstAccess
enum { ACCESS_NONE = 0, ACCESS_READ = 1, ACCESS_WRITE = 2 };

// Flags of an instrumented memory instruction
enum { MEM_FLAG_LOAD = 1, MEM_FLAG_STORE = 2 };

// Parameters for the device code, in managed memory. Written by the host at
// kernel launch, when no instrumented kernel is running.
//
// fired and oldWord change during a kernel: the device reads them with atomics,
// which do not hit L1. A plain load could return a stale copy (see README.md).
typedef struct {
	int armed;                // count instructions and flip in this launch (timing mode b)
	int traceOn;              // check memory instructions against the flipped byte
	int kernelIdx;            // index of the running kernel
	int fired;                // FLIP_*
	unsigned int loads;       // loads of the flipped byte seen by the trace, before or after the flip
	unsigned int smLoads;     // mask of the SMs that made them
	unsigned int oldWord;     // value of the word before the flip
	unsigned int mask;        // XOR mask of the flip, one bit
	unsigned int appliedMask; // mask actually applied: 0 with DUMMY=1
	unsigned long long instID;       // dynamic instruction that flips (timing mode b)
	unsigned long long wordAddr;     // address of the 32-bit word
	unsigned long long flipByteAddr; // address of the byte holding the flipped bit
} mem_info_t;

// Injection record, in host-mapped pinned memory: the host can read it after a
// kernel error or while a kernel hangs. The device updates the trace counters
// with atomics and no fence, because a fence invalidates L1 and would destroy
// the stale copies this record measures (see README.md).
typedef struct {
	int state;                 // MEM_*
	int kernelIdx;             // kernel of the flip, -1 if it happened between kernels
	long long dynInstr;        // dynamic instruction of the flip, -1 if between kernels
	int pcOffset;
	int tid;
	int ctaFlat;
	int smID;
	int warpID;
	int laneID;
	unsigned int oldWord;
	unsigned int newWord;

	unsigned int readsOld;     // device loads that returned the old bit
	unsigned int readsNew;     // device loads that returned the flipped bit
	unsigned int writesAfter;  // device stores after the flip
	unsigned int smOld;        // mask of the SMs that returned the old bit
	unsigned int smNew;        // mask of the SMs that returned the flipped bit
	int firstAccess;           // ACCESS_* set by the first device or host access
	unsigned int hostReads;    // host copies that read the flipped byte
	unsigned int hostWrites;   // host copies or memsets that wrote it
	unsigned long long readKernels; // bit k: kernel k read the byte after the flip (bit 63: k >= 63)
} mem_shadow_t;

#endif
