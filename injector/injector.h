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
 */

#ifndef INJECTOR_H
#define INJECTOR_H

#include "arch.h"

#define NUM_DEBUG_VALS 4
#define MAX_KNAME_SIZE 1000

typedef struct {
	// Parameters to inject an error (read from nvbitfi-injection-info.txt)
	bool areParamsReady;
	char kernelName[MAX_KNAME_SIZE];
	int32_t kernelCount;
	int32_t groupID; // instruction group (GroupType in arch.h)
	unsigned long long instID; // index of the dynamic instance among the injectable instances of the group
	float opIDSeed; // destination selection seed (0-1)
	uint32_t bitFlipModel; // bit-flip model (BitFlipModel in arch.h)
	int mcuK; // size of a multi-cell model, otherwise 1
	float bitIDSeed; // bit selection seed (0-1)

	// Selection (written by the BEFORE call of the selected dynamic instance)
	bool selected;       // the selected instance has been reached
	int selPcOffset;     // its PC offset
	int selCtaFlat;      // its CTA (flattened blockIdx)
	int selTidInCta;     // its thread (flattened threadIdx)
	int selWarp;         // its warp slot
	int selLane;         // its lane
	unsigned int selLaneMask; // lanes that corrupt the register, MCU_ADJ_LANES

	bool errorInjected;  // the AFTER call of the selected instance has run

	// Address model (G_ADDR): the register to restore after the instruction
	bool addrDone;       // the instruction has finished
	bool addrRestore;    // restore it: the instruction does not write this register
	int addrKind;        // AddrPartKind
	int addrReg;
	unsigned int addrOld;

	int debug[NUM_DEBUG_VALS]; // [0]=1 if the instrumented kernel ran, [1]=expected GPR value, [2]=nvbit_read_reg value, [3]=reg
} inj_info_t;

// Progress of the injection, stored in inj_shadow_t::state
enum { SHADOW_NONE = 0, SHADOW_SELECTED = 1, SHADOW_INJECTED = 2 };

typedef struct {
	int regNo;
	int lane;
	int tid;
	unsigned int mask;
	unsigned int before;
	unsigned int after;
} inj_cell_t;

// Injection record, in host-mapped pinned memory (cudaHostAlloc) instead of
// managed memory: the host can read it even after a kernel error or while a
// kernel hangs, when touching managed memory may fail or block. Written by one
// thread only, about twice per run: it is not read by the device, so its cost
// over PCIe does not depend on the number of instrumented instructions.
typedef struct {
	int state;            // SHADOW_*; each step is published after its fields

	// Written by the BEFORE call of the selected instance
	int opcode;           // opcode index (instTypeNames)
	int pcOffset;         // PC offset of the instruction in the function
	int tid;              // flattened global thread id
	int ctaFlat;          // flattened CTA id
	int smID;             // %smid of the SM that executed the injection
	int warpID;           // %warpid (hardware warp slot in the SM)
	int laneID;           // %laneid (0-31)

	// Written by the AFTER call, before the register is corrupted
	int regType;          // RegType in arch.h: GPR, PR, UR, UP (-1 = none)
	int regNo;            // register number (Rn, Pn, URn, UPn)
	unsigned int mask;    // XOR mask actually applied
	unsigned int beforeVal; // register value before injection (whole predicate register for PR/UP)
	unsigned int afterVal;  // register value after injection

	// All the registers corrupted, the selected one included. Each lane writes
	// its own entry. A predicate model with k bits is one entry with k bits in mask.
	int requestedCells;     // cells the model asks for: inactive lanes and registers out of range are not written
	int numCells;
	inj_cell_t cells[MAX_MCU_CELLS];

	// Address model. Written before the register is corrupted, except the last two
	unsigned long long addrOrig;     // address of the instruction, from NVBit
	unsigned long long addrNew;      // address with the flip: with DUMMY=1 only computed, not used
	int addrPart;                    // AddrPartKind, -1 = not this model
	int addrBit;                     // bit inside the part
	int memAccess;                   // MemAccess
	int addrRestored;                // the AFTER call put the register back
	int addrOverwritten;             // the instruction wrote that register: not restored
} inj_shadow_t;

#endif
