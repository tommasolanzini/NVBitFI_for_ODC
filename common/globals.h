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

#ifndef GLOBALS_H
#define GLOBALS_H

#include <assert.h>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream> 
#include <iterator>

#include <set>
#include "arch.h"
#include "dests.h"

//////////////////////////////////////////////////////////////////////////////////
// Kernel launch helpers (NVBit >= 1.7, CUDA >= 12)
//
// CUDA 12 runtimes launch many kernels through cuLaunchKernelEx, whose params
// struct starts with a CUlaunchConfig* (not with the CUfunction). The original
// NVBitFI only intercepted cuLaunch/cuLaunchKernel and cast every params struct
// to cuLaunch_params, so kernels launched via cuLaunchKernelEx were never
// instrumented (or the cast read garbage).
//////////////////////////////////////////////////////////////////////////////////
inline bool is_kernel_launch(nvbit_api_cuda_t cbid) {
	return cbid == API_CUDA_cuLaunch ||
		cbid == API_CUDA_cuLaunchGrid ||
		cbid == API_CUDA_cuLaunchGridAsync ||
		cbid == API_CUDA_cuLaunchKernel ||
		cbid == API_CUDA_cuLaunchKernel_ptsz ||
		cbid == API_CUDA_cuLaunchCooperativeKernel ||
		cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
		cbid == API_CUDA_cuLaunchKernelEx ||
		cbid == API_CUDA_cuLaunchKernelEx_ptsz;
}

inline bool is_launch_ex(nvbit_api_cuda_t cbid) {
	return cbid == API_CUDA_cuLaunchKernelEx || cbid == API_CUDA_cuLaunchKernelEx_ptsz;
}

// CUfunction of the kernel being launched
inline CUfunction get_launched_function(nvbit_api_cuda_t cbid, void *params) {
	if (is_launch_ex(cbid)) {
		return ((cuLaunchKernelEx_params *)params)->f;
	}
	return ((cuLaunch_params *)params)->f; // every other launch struct starts with CUfunction f
}

// Number of CTAs (thread blocks) of the launch, 0 if the API does not carry it
inline int get_num_ctas(nvbit_api_cuda_t cbid, void *params) {
	if (is_launch_ex(cbid)) {
		const CUlaunchConfig *c = ((cuLaunchKernelEx_params *)params)->config;
		return c->gridDimX * c->gridDimY * c->gridDimZ;
	}
	if (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz ||
			cbid == API_CUDA_cuLaunchCooperativeKernel || cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
		cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
		return p->gridDimX * p->gridDimY * p->gridDimZ;
	}
	return 0;
}

// Output log file 
std::string injOutputFilename = "nvbitfi-injection-log-temp.txt";
std::ofstream  fout;

// kernel_id is a counter we used to keep track of which kernel needs
// injection. We do not maintain 2 counters for kernel launches.  Ideally we
// want to maintain one for kernel launch and one for kernel exit because
// kernels can be launched currently and end in the order that's different than
// the order in which they were launched. This is a TODO for later time.
int kernel_id = 0;

void open_output_file(std::string filename) {
	static bool open_flag = false; // file will be parsed only once - performance enhancement
	if (!open_flag) {
 		fout.open(filename.c_str(), std::ifstream::out);
		open_flag = true;
	}
}


__managed__ uint64_t counters[NUM_COUNTERS]; // instruction counters (layout below)

// Counter layout:
//   counters[0 .. NUM_OPCODE_COUNTERS)   dynamic thread-instructions per opcode
//                                        (last slot = UNKNOWN_OPCODE); uniform
//                                        instructions count once per warp
//   counters[NUM_OPCODE_COUNTERS + g]    INJECTABLE dynamic instances of group g
//                                        (instances with at least one destination
//                                        of the right register class, see dests.h);
//                                        the injection index (instID) of a site is
//                                        an index into this counter
// Executed instructions per group can be derived from the per-opcode counts.
uint64_t get_inst_count() {
	uint64_t sum = 0;
	for (int i=0; i<NUM_OPCODE_COUNTERS; i++) {
		sum += counters[i];
	}
	return sum;
}
void init_counters() {
	cudaDeviceSynchronize();
	for (int i=0; i<NUM_COUNTERS; i++) {
		counters[i] = 0;
	}
}

// remove spaces from a string - used mainly for kernel names
std::string removeSpaces(const char * str) {
	std::string s = str;
	s.erase(remove_if(s.begin(), s.end(), isspace), s.end());
	return s;
}

//////////////////////////////////////////////////////////////////////////////////
// Architecture-related functions
//
// The opcode table, the opcode -> group map and the destination analysis live
// in dests.h (shared with the offline test). Below: the NVBit front-end that
// turns Instr operands into the abstract operand list used by dests.h, and the
// per-instruction "static info" passed to the device functions.
//////////////////////////////////////////////////////////////////////////////////

inline uint32_t eligibility_flags(int grp, const DestInfo &d, const AddrInfo &a) {
	uint32_t f = 0;
	if (a.total_bits() > 0) f |= FLAG_ELIG_ADDR;
	if (grp == G_UNIFORM) f |= FLAG_UNIFORM;
	if (grp == G_NODEST || grp == G_UNKNOWN) {
		f |= FLAG_ELIG_OWN; // counted (statistics only), never injected
	} else if (d.any()) {
		f |= FLAG_ELIG_OWN;
	}
	bool regular = (grp != G_NODEST && grp != G_UNKNOWN && grp != G_UNIFORM);
	if (regular && (d.num_gpr > 0 || d.pr_mask)) f |= FLAG_ELIG_GPPR;
	if (regular && d.num_gpr > 0) f |= FLAG_ELIG_GP;
	return f;
}

// Pack a register range (base, count) into one 32-bit argument
inline uint32_t pack_range(int base, int num) {
	return (base < 0 || num <= 0) ? 0 : (((uint32_t)base << 8) | (uint32_t)num);
}

inline std::string strip_operand_modifiers(const char *str) {
	std::string s = str;
	while (!s.empty() && (s[0] == '-' || s[0] == '|' || s[0] == '!' || s[0] == '~' || s[0] == ' ')) s.erase(0, 1);
	return s;
}

// NVBit operands -> abstract operand list (guard predicate is not an operand).
// Discard targets are recognised by name: RZ, URZ, PT, UPT.
inline std::vector<OpDesc> nvbit_operands(Instr *i) {
	std::vector<OpDesc> v;
	for (int k = 0; k < i->getNumOperands(); k++) {
		const InstrType::operand_t *op = i->getOperand(k);
		std::string s = strip_operand_modifiers(op->str);
		switch (op->type) {
			case InstrType::OperandType::REG:
				v.push_back({OK_REG, (s.rfind("RZ", 0) == 0 || op->u.reg.num == InstrType::RZ) ? DISCARD_REG : op->u.reg.num});
				break;
			case InstrType::OperandType::UREG:
				v.push_back({OK_UREG, (s.rfind("URZ", 0) == 0) ? DISCARD_REG : op->u.reg.num});
				break;
			case InstrType::OperandType::PRED:
				if (s.rfind("PR", 0) == 0 && s != "PT" && (s.size() == 2 || s[2] == '.')) v.push_back({OK_PR_ALL, 0});
				else if (op->u.pred.num == InstrType::PR) v.push_back({OK_PR_ALL, 0});
				else v.push_back({OK_PRED, (s.rfind("PT", 0) == 0 || op->u.pred.num == InstrType::PT) ? DISCARD_REG : op->u.pred.num});
				break;
			case InstrType::OperandType::UPRED:
				if (s.rfind("UPR", 0) == 0 || op->u.pred.num == InstrType::UPR) v.push_back({OK_UPR_ALL, 0});
				else v.push_back({OK_UPRED, (s.rfind("UPT", 0) == 0 || op->u.pred.num == InstrType::UPT) ? DISCARD_REG : op->u.pred.num});
				break;
			case InstrType::OperandType::IMM_UINT64:
				v.push_back({OK_IMM, (int64_t)op->u.imm_uint64.value});
				break;
			case InstrType::OperandType::IMM_DOUBLE:
				v.push_back({OK_IMM, 0});
				break;
			case InstrType::OperandType::GENERIC:
				if (s == "PR") v.push_back({OK_PR_ALL, 0});
				else if (s == "UPR") v.push_back({OK_UPR_ALL, 0});
				else v.push_back({OK_OTHER, 0});
				break;
			default:
				v.push_back({OK_OTHER, 0});
				break;
		}
	}
	return v;
}

// The memory operand of an instruction, from NVBit's MREF and MEM_DESC operands.
// Zero registers are recognised by name, as for the destinations.
inline MrefDesc nvbit_mref(Instr *i) {
	MrefDesc m;
	for (int k = 0; k < i->getNumOperands(); k++) {
		const InstrType::operand_t *op = i->getOperand(k);
		if (op->type == InstrType::OperandType::MEM_DESC) {
			m.desc = op->u.mem_desc.ureg_num;
		} else if (op->type == InstrType::OperandType::MREF) {
			const std::string s = op->str;
			if (op->u.mref.has_ra && s.find("[RZ") == std::string::npos && op->u.mref.ra_num != InstrType::RZ) {
				const InstrType::RegModifierType mod = op->u.mref.ra_mod;
				m.ra = op->u.mref.ra_num;
				m.ra64 = (mod == InstrType::RegModifierType::U64);
				m.raUnsigned = (mod == InstrType::RegModifierType::U32);
				m.raStride = mod == InstrType::RegModifierType::X4 ? 4 : mod == InstrType::RegModifierType::X8 ? 8 : mod == InstrType::RegModifierType::X16 ? 16 : 1;
			}
			if (op->u.mref.has_ur && s.find("URZ") == std::string::npos) {
				m.ur = op->u.mref.ur_num;
				m.ur64 = (op->u.mref.ur_mod == InstrType::RegModifierType::U64);
			}
			if (op->u.mref.has_desc) m.desc = op->u.mref.desc_ureg_num;
		} else if (op->type == InstrType::OperandType::CBANK && op->u.cbank.has_reg_offset) {
			const std::string s = op->str;
			const size_t b = s.rfind('[');
			m.cbank_ur = b != std::string::npos && s.compare(b + 1, 2, "UR") == 0;
			if (op->u.cbank.reg_offset != InstrType::RZ && s.find("RZ") == std::string::npos) m.cbank_reg = op->u.cbank.reg_offset;
		}
	}
	return m;
}

// Static information about one SASS instruction, computed once at
// instrumentation time and passed to the device functions as constants.
struct StaticInstInfo {
	int opcode;        // index in the opcode table (UNKNOWN_OPCODE if not found)
	int group;         // exclusive group
	DestInfo dests;    // destination registers
	AddrInfo addr;     // registers of the address, for a memory instruction
	uint32_t flags;    // FLAG_* bits
};

inline StaticInstInfo analyze_instr(Instr *i) {
	StaticInstInfo si;
	std::string full = i->getOpcode();
	si.opcode = opcode_index(short_opcode(full));
	si.group = opcode_group(si.opcode);
	si.dests = analyze_dests(full, si.opcode, nvbit_operands(i));
	si.addr = analyze_addr(si.opcode, nvbit_mref(i));
	si.flags = eligibility_flags(si.group, si.dests, si.addr);
	return si;
}

inline SrcInfo instr_sources(Instr *i, const StaticInstInfo &si) {
	return analyze_sources(i->getOpcode(), si.opcode, nvbit_operands(i), si.dests, nvbit_mref(i));
}

// The ABI stack pointer. NVBit's own instrumentation uses it to reach memory, whether
// or not the application does: see "The stack pointer" in README.md.
static const int STACK_POINTER_GPR = 1;

inline bool reads_gpr(const std::vector<Instr *> &instrs, int reg) {
	for (Instr *i : instrs) {
		const SrcInfo s = instr_sources(i, analyze_instr(i));
		if (std::find(s.gpr.begin(), s.gpr.end(), reg) != s.gpr.end()) return true;
	}
	return false;
}

// One past the highest GPR and UR named by the instructions of a function, from
// the decoded operands and destinations. MCU_ADJ_REGS stops here, not at
// func_config_t::num_registers: on gemm_core that says 40, the code uses up to R37,
// and two DUMMY=1 runs that read R38 ended in "illegal instruction".
struct UsedRegs { int gpr = 0; int ur = 0; };

inline UsedRegs used_regs(const std::vector<Instr *> &instrs) {
	UsedRegs u;
	for (Instr *i : instrs) {
		for (const OpDesc &op : nvbit_operands(i)) {
			if (op.kind == OK_REG && op.num >= 0 && op.num < RZ_NUM) u.gpr = std::max(u.gpr, (int)op.num + 1);
			if (op.kind == OK_UREG && op.num >= 0 && op.num < MAX_UR) u.ur = std::max(u.ur, (int)op.num + 1);
		}
		const StaticInstInfo si = analyze_instr(i);
		if (si.dests.num_gpr) u.gpr = std::max(u.gpr, si.dests.gpr_base + si.dests.num_gpr);
		if (si.dests.num_ur) u.ur = std::max(u.ur, si.dests.ur_base + si.dests.num_ur);
		const MrefDesc m = nvbit_mref(i);
		if (m.ra >= 0 && m.ra < RZ_NUM) u.gpr = std::max(u.gpr, m.ra + (m.ra64 ? 2 : 1));
		if (m.ur >= 0) u.ur = std::max(u.ur, m.ur + (m.ur64 ? 2 : 1));
		if (m.desc >= 0) u.ur = std::max(u.ur, m.desc + 2);
	}
	return u;
}

// Report (once per opcode) SASS opcodes that are missing from the table in
// arch.h: they are counted in G_UNKNOWN and never injected.
inline void warn_unknown_opcode(Instr *i) {
	static std::set<std::string> seen;
	std::string s = short_opcode(i->getOpcode());
	if (seen.insert(s).second) {
		fprintf(stderr, "NVBitFI warning: opcode %s is not in common/arch.h (counted as UNKNOWN, never injected): %s\n", s.c_str(), i->getSass());
	}
}

#endif
