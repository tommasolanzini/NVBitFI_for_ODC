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
// ACCESS_AMBIGUOUS (shared memory): a store checked before the flip and finished
// after it. It cannot be told whether the store came before or after the flip.
enum { ACCESS_NONE = 0, ACCESS_READ = 1, ACCESS_WRITE = 2, ACCESS_AMBIGUOUS = 3 };

// Multi-cell upsets: the words of one injection, and the first bytes of the window
// of a CTA, reserved by the runtime (cudaDevAttrReservedSharedMemoryPerBlock)
#define MEM_MAX_CELLS 8
#define MEM_SHM_RESERVED 1024

// k ones from bit start. Every word of an injection gets the same mask.
#define MEM_MCU_MASK(k, start) (((k) >= 32 ? 0xffffffffu : ((1u << (k)) - 1)) << (start))

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
	unsigned int mask;        // XOR mask of the flip: k adjacent bits, the same in every word
	unsigned int appliedMask; // mask actually applied: 0 with DUMMY=1
	unsigned long long instID;       // dynamic instruction that flips (timing mode b)
	unsigned long long wordAddr;     // address of the first 32-bit word; the others are at cellDelta
	int numCells;                    // words of this injection, at least 1
	int cellDelta[MEM_MAX_CELLS];    // their distance in bytes from wordAddr
	unsigned int cellOld[MEM_MAX_CELLS]; // value of each word before the flip
	int mcuK;                        // adjacent bits per word

	// Shared memory (modes s and u). The offsets are relative to the shared
	// window of a CTA. flipCta, wordAddr and mask are set by the flipping thread
	// before FLIP_RUNNING and read with atomics, as fired. A word outside
	// [shmLo, shmHi) is not flipped: mode s only knows its word at run time.
	unsigned int shmLo;
	unsigned int shmHi;
	int applyFlip;            // 0 with DUMMY=1
	int targetSM;             // mode u: SM of the flip
	int claimed;              // mode u: a thread took the flip
	int flipCta;              // CTA of the flip: only its threads can see it
	unsigned int bitSeed;     // mode s: picks the bit inside the stored bytes; mode u: bit of the word
	unsigned long long flipOffsetTarget; // mode u: offset of the word in the window of the CTA
	unsigned int pickSeed;    // mode u: picks a CTA among the resident ones
	int chooserClaim;         // mode u: a thread is choosing the CTA
	int chosenCta;            // mode u: CTA of the flip, CTA_NOT_CHOSEN until chosen
	unsigned long long chooserInstr; // mode u: instruction index when the CTA was chosen
} mem_info_t;

#define CTA_NOT_CHOSEN (-1)

// Residency of a CTA. Count phase: first and last are index + 1 on the clock of
// the dynamic instructions, 0 = never seen. Mode u: first != 0 once the CTA has
// started, exits = threads that reached EXIT.
#define MEM_MAX_CTAS 4096
typedef struct {
	unsigned long long first;
	unsigned long long last;
	int sm;
	unsigned int exits;
	int slot;
	int pad;
} mem_cta_t;

// Mode u: the CTAs resident on an SM, updated when a CTA enters and leaves, so that
// choosing one takes two atomic reads. A scan of a table per CTA delayed the flip by
// up to 30% of the kernel. 32 slots: this GPU has at most 24 CTAs per SM (tests/shm_probe).
#define MEM_MAX_SMS 64
typedef struct {
	unsigned int mask; // bit k: slot k is in use
	int cta[32];
} mem_sm_t;

// One flipped word and its trace. After a store to the word, loads no longer count:
// the bit then comes from the application, not from the flip.
typedef struct {
	int delta;                 // bytes from the first word
	int applied;               // flipped: 0 if outside the window of the CTA
	unsigned int oldWord;
	unsigned int newWord;
	unsigned int readsOld;     // device loads that returned the old bits
	unsigned int readsNew;     // device loads that returned the flipped bits
	unsigned int writesAfter;  // device stores after the flip
	unsigned int smOld;        // mask of the SMs that returned the old bits
	unsigned int smNew;        // mask of the SMs that returned the flipped bits
	int firstAccess;           // ACCESS_* set by the first device or host access
	unsigned int hostReads;    // host copies that read the flipped bytes
	unsigned int hostWrites;   // host copies or memsets that wrote them
} mem_cell_t;

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

	int numCells;              // words flipped
	mem_cell_t cells[MEM_MAX_CELLS];
	unsigned long long readKernels; // bit k: kernel k read the byte after the flip (bit 63: k >= 63)
	unsigned int shmWordOffset; // shared memory: offset of the flipped word in the window of the CTA
	unsigned int mask;          // XOR mask of each word
	int residents;              // mode u: CTAs resident on the SM when the CTA was chosen, -1 = not chosen
	int chosenCta;              // mode u: CTA chosen among them
	int chosenExits;            // mode u: threads of that CTA already gone at that moment
} mem_shadow_t;

#endif
