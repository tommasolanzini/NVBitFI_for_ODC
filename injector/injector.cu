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
 * NVBitFI_for_ODC (2026): ported to NVBit 1.8; destination registers from
 * NVBit's decoded operands (common/dests.h); injection into GPRs, predicate
 * registers, uniform registers and uniform predicates; two-phase
 * select (BEFORE) / inject (AFTER); hardware location (SM, warp slot, lane)
 * in the injection log.
 */

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <signal.h>
#include <unistd.h>
#include <unordered_set>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

#include "globals.h"
#include "injector.h"

int verbose;
__managed__ int verbose_device;

// injection parameters input filename: This file is created the the script
// that launched error injections
std::string injInputFilename = "nvbitfi-injection-info.txt";

pthread_mutex_t mutex;

__managed__ inj_info_t inj_info;

// Injection record in host-mapped pinned memory (see inj_shadow_t)
static volatile inj_shadow_t *shadow = NULL;
static void *shadow_device_ptr = NULL;

// Everything the record needs from the host side, kept in plain host memory so
// that write_inj_record() never touches managed memory on the crash paths.
static bool target_launched = false; // the kernel selected for injection has been launched
static int target_index = -1;
static std::string target_kname;
static int target_ctas = -1;
static bool record_written = false;  // the log gets exactly one record

void reset_inj_info() {
	inj_info.areParamsReady = false;
	inj_info.kernelName[0] = '\0';
	inj_info.kernelCount = -1;
	inj_info.groupID = 0;
	inj_info.instID = 0;
	inj_info.opIDSeed = 0;
	inj_info.bitIDSeed = 0;
	inj_info.bitFlipModel = 0;
	inj_info.selected = false;
	inj_info.selPcOffset = -1;
	inj_info.selCtaFlat = -1;
	inj_info.selTidInCta = -1;
	inj_info.errorInjected = false;
	for (int i=0; i<NUM_DEBUG_VALS; i++) {
		inj_info.debug[i] = -1;
	}
}

// Write the injection record to the log, once. Safe when the CUDA context is
// dead or a kernel hangs: it only reads host memory. groupInstrs lives in
// managed memory, so only the normal path passes it (< 0 = not available).
void write_inj_record(long long groupInstrs) {
	if (record_written) return; // e.g. a later kernel crashed after the record was written
	record_written = true;
	assert(fout.good());

	if (target_launched) {
		fout << "Injection data" << std::endl;
		fout << "index: " << target_index << std::endl;
		fout << "kernel_name: " << target_kname << std::endl;
		fout << "ctas: " << target_ctas << std::endl;
	}
	// injectable instances of the selected group in this kernel (must match the profile)
	if (groupInstrs >= 0) fout << "groupInstrs: " << groupInstrs << std::endl;

	const int state = shadow ? shadow->state : SHADOW_NONE;
	if (state >= SHADOW_SELECTED) {
		fout << "opcode: " << instTypeNames[shadow->opcode] << std::endl;
		fout << "pcOffset: 0x" << std::hex << shadow->pcOffset << std::dec << std::endl;
		fout << "tid: " << shadow->tid << std::endl;
		fout << "ctaID: " << shadow->ctaFlat << std::endl;
		fout << "smID: " << shadow->smID << std::endl;
		fout << "warpID: " << shadow->warpID << std::endl;
		fout << "laneID: " << shadow->laneID << std::endl;
		const int regType = shadow->regType;
		if (regType >= 0 && regType < NUM_REG_TYPES) {
			fout << "mask: 0x" << std::hex << shadow->mask << std::endl;
			fout << "beforeVal: 0x" << shadow->beforeVal << ";";
			fout << "afterVal: 0x" << shadow->afterVal << std::dec << std::endl;
			fout << "regType: " << regTypeNames[regType] << std::endl;
			fout << "regNo: " << shadow->regNo << std::endl;
		}
	}
	if (state != SHADOW_INJECTED) {
		fout << "Error not injected\n";
	}
	fout.flush();
}

void print_inj_info() {
	assert(fout.good());
	fout << "kernelName=" << inj_info.kernelName << std::endl;
	fout << "kernelCount=" << inj_info.kernelCount << std::endl;
	fout << "groupID=" << inj_info.groupID << " (" << instGrouptNames[inj_info.groupID] << ")" << std::endl;
	fout << "bitFlipModel=" << inj_info.bitFlipModel  << std::endl;
	fout << "instID=" << inj_info.instID << std::endl;
	fout << "opIDSeed=" << inj_info.opIDSeed << std::endl;
	fout << "bitIDSeed=" << inj_info.bitIDSeed << std::endl;
}

// Parse error injection site info from a file. This should be done on host side.
void parse_params(std::string filename) {
	static bool parse_flag = false; // file will be parsed only once - performance enhancement
	if (!parse_flag) {
		parse_flag = true;
		reset_inj_info();

		std::ifstream ifs (filename.c_str(), std::ifstream::in);
		if (ifs.is_open()) {
			ifs >> inj_info.groupID; // instruction group
			assert(inj_info.groupID >=0 && inj_info.groupID < NUM_INST_GROUPS); // ensure that the value is in the expected range
			// nothing can be injected in these groups
			assert(inj_info.groupID != G_NODEST && inj_info.groupID != G_UNKNOWN);

			ifs >> inj_info.bitFlipModel; // fault model: single bit flip, all bit flip, random value
			assert(inj_info.bitFlipModel < NUM_BFM_TYPES); // ensure that the value is in the expected range

			ifs >> inj_info.kernelName;
			ifs >> inj_info.kernelCount;
			ifs >> inj_info.instID; // instruction id

			ifs >> inj_info.opIDSeed; // destination id seed (float, 0-1)
			assert(inj_info.opIDSeed >=0 && inj_info.opIDSeed < 1.01); // ensure that the value is in the expected range

			ifs >> inj_info.bitIDSeed; // bit location seed (float, 0-1)
			assert(inj_info.bitIDSeed >= 0 && inj_info.bitIDSeed < 1.01); // ensure that the value is in the expected range
		} else {
			printf(" File %s does not exist!", filename.c_str());
			printf(" This file should contain enough information about the fault site to perform an error injection run: ");
			printf("(1) instruction group id, (2) bit flip model, (3) kernel name, (4) kernel count, (5) instruction id, (6) seed to select destination id, (7) sed to select bit location.\n");
			assert(false);
		}
		ifs.close();

		if (verbose) {
			print_inj_info();
		}
	}
}

void INThandler(int sig) {
	signal(sig, SIG_IGN); // disable Ctrl-C

	fout << "ERROR FAIL Detected Signal SIGKILL\n";
	write_inj_record(-1);
	_exit(-1); // not exit(): the CUDA teardown can block if a kernel is hanging
}

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init() {
	/* just make sure all managed variables are allocated on GPU */
	setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC","1",1);

	/* we get some environment variables that are going to be use to selectively
	 * instrument (within a interval of kernel indexes and instructions). By
	 * default we instrument everything. */
	if (getenv("TOOL_VERBOSE")) {
		verbose = atoi(getenv("TOOL_VERBOSE"));
	} else {
		verbose = 0;
	}

	signal(SIGINT, INThandler); // install Ctrl-C handler

	open_output_file(injOutputFilename);
	if (verbose) printf("nvbit_at_init:end\n");
}

/* Called before the first kernel launch: the shadow record must exist when the
 * kernel is instrumented, because its address is passed to the device calls. */
void nvbit_tool_init(CUcontext ctx) {
	if (shadow) return;
	void *host_ptr = NULL;
	if (cudaHostAlloc(&host_ptr, sizeof(inj_shadow_t), cudaHostAllocMapped) != cudaSuccess ||
			cudaHostGetDevicePointer(&shadow_device_ptr, host_ptr, 0) != cudaSuccess) {
		fprintf(stderr, "NVBitFI: cannot allocate the host-mapped injection record\n");
		exit(1);
	}
	inj_shadow_t init = {};
	init.state = SHADOW_NONE;
	init.opcode = -1;
	init.pcOffset = -1;
	init.tid = -1;
	init.ctaFlat = -1;
	init.smID = -1;
	init.warpID = -1;
	init.laneID = -1;
	init.regType = -1;
	init.regNo = -1;
	memcpy(host_ptr, &init, sizeof(init));
	shadow = (volatile inj_shadow_t *)host_ptr;
}

// Does this instruction need to be instrumented for the selected group?
// Only instructions that can be counted for that group (otherwise the BEFORE
// call returns immediately anyway; skipping them keeps the overhead low).
static bool relevant_for_group(const StaticInstInfo &si, int igid) {
	if (igid == G_GPPR) return si.flags & FLAG_ELIG_GPPR;
	if (igid == G_GP) return si.flags & FLAG_ELIG_GP;
	return si.group == igid && (si.flags & FLAG_ELIG_OWN);
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {

	parse_params(injInputFilename.c_str());  // injParams are updated based on injection seed file
	cudaDeviceSynchronize();
	verbose_device = verbose;
	cudaDeviceSynchronize();

	/* Only the kernel selected for injection is instrumented, together with
	 * all its related (device) functions: the profiler counts the
	 * instructions of non-inlined device functions (CALL) as part of the
	 * kernel, so the injector must count them too. (The original matched the
	 * name of every related function against the kernel name, so device
	 * functions were never instrumented and the counts could disagree.) */
	std::string launched = removeSpaces(nvbit_get_func_name(ctx, func));
	if (strcmp(inj_info.kernelName, launched.c_str()) != 0) {
		if (verbose)
			printf(":::NVBit-inject-error; NOT inspecting: %s; %d, %d:::", launched.c_str(), kernel_id, inj_info.kernelCount);
		return;
	}

	std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
	related_functions.push_back(func); /* add kernel itself to the related function vector */

	for (auto f : related_functions) {
		/* skip functions that were already instrumented */
		if (!already_instrumented.insert(f).second) {
			continue;
		}

		std::string kname = removeSpaces(nvbit_get_func_name(ctx,f));
		assert(fout.good()); // ensure that the log file is good.

		/* Get the vector of instruction composing the loaded CUFunction "f" */
		const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
		if (verbose) {
			printf("inspecting %s - num instrs %ld\n", nvbit_get_func_name(ctx, f), instrs.size());
		}

		func_config_t config;
		nvbit_get_func_config(ctx, f, &config);
		fout << "inspecting: " << kname << "\nnum_static_instrs: " << instrs.size() << "\nmaxregs: " << config.num_registers << std::endl;

		int num_instrumented = 0;
		/* We iterate on the vector of instruction */
		for (auto i : instrs) {
			StaticInstInfo si = analyze_instr(i);
			if (si.opcode == UNKNOWN_OPCODE) warn_unknown_opcode(i);
			if (!relevant_for_group(si, inj_info.groupID)) continue;
			num_instrumented++;

			if (verbose) {
				printf("0x%04x %-10s grp=%-8s dests=%-16s flags=0x%x  %s\n", i->getOffset(),
						instTypeNames[si.opcode], instGrouptNames[si.group], si.dests.str().c_str(), si.flags, i->getSass());
			}

			/* BEFORE: count injectable instances and select the target */
			nvbit_insert_call(i, "inject_select", IPOINT_BEFORE);
			nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_info);
			nvbit_add_call_arg_const_val64(i, (uint64_t)shadow_device_ptr);
			nvbit_add_call_arg_const_val64(i, (uint64_t)counters);
			nvbit_add_call_arg_const_val32(i, i->getOffset()); // offset (for pc) info
			nvbit_add_call_arg_const_val32(i, si.opcode); // opcode info
			nvbit_add_call_arg_const_val32(i, si.group); // instruction group info
			nvbit_add_call_arg_const_val32(i, si.flags); // eligibility flags
			nvbit_add_call_arg_guard_pred_val(i); // predicate value

			/* AFTER: corrupt one destination of the selected instance */
			nvbit_insert_call(i, "inject_error", IPOINT_AFTER);
			nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_info);
			nvbit_add_call_arg_const_val64(i, (uint64_t)shadow_device_ptr);
			nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);
			nvbit_add_call_arg_const_val32(i, i->getOffset());
			nvbit_add_call_arg_const_val32(i, pack_range(si.dests.gpr_base, si.dests.num_gpr));
			nvbit_add_call_arg_const_val32(i, si.dests.pr_mask);
			nvbit_add_call_arg_const_val32(i, pack_range(si.dests.ur_base, si.dests.num_ur));
			nvbit_add_call_arg_const_val32(i, si.dests.up_mask);
			if (si.dests.num_gpr > 0) {
				nvbit_add_call_arg_reg_val(i, si.dests.gpr_base); // first destination GPR value (sanity check)
			} else {
				nvbit_add_call_arg_const_val32(i, (unsigned int)-1);
			}
		}
		fout << "num_instrumented_instrs: " << num_instrumented << std::endl;
	}
}

/* This call-back is triggered every time a CUDA event is encountered.
 * Here, we identify CUDA kernel launch events and reset the "counter" before
 * th kernel is launched, and print the counter after the kernel has completed
 * (we make sure it has completed by using cudaDeviceSynchronize()). To
 * selectively run either the original or instrumented kernel we used
 * nvbit_enable_instrumented() before launching the kernel. */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
		const char *name, void *params, CUresult *pStatus) {
	/* Identify all the possible CUDA launch events */
	if (is_kernel_launch(cbid)) {
		/* CUfunction of the launched kernel (also for cuLaunchKernelEx) */
		CUfunction func = get_launched_function(cbid, params);

		if(!is_exit) {
			pthread_mutex_lock(&mutex);
			instrument_function_if_needed(ctx, func);
			init_counters();
			cudaDeviceSynchronize();
			parse_params(injInputFilename);  // injParams are updated based on injection seed file
			inj_info.errorInjected = false;
			inj_info.selected = false;
			inj_info.areParamsReady = (inj_info.kernelCount== kernel_id); // areParamsReady = true for the selected kernel
			inj_info.debug[0] = -1; // set to 1 by the AFTER call when the instrumented kernel ran
			if (inj_info.areParamsReady) {
				target_launched = true;
				target_index = kernel_id;
				target_kname = removeSpaces(nvbit_get_func_name(ctx, func));
				target_ctas = get_num_ctas(cbid, params);
			}
			if (verbose) printf("setting areParamsReady=%d, inj_info.kernelCount=%d, kernel_id=%d\n", inj_info.areParamsReady, inj_info.kernelCount, kernel_id);
			cudaDeviceSynchronize();

			nvbit_enable_instrumented(ctx, func, inj_info.areParamsReady); // should we run the un-instrumented code?
			cudaDeviceSynchronize();
		}  else {
			if (verbose) printf("is_exit\n");
			cudaDeviceSynchronize();

			cudaError_t le = cudaGetLastError();
			if ( cudaSuccess != le ) {
				assert(fout.good());
				std::cout << "ERROR FAIL in kernel execution (" << cudaGetErrorString(le) << "); ";
				fout << "ERROR FAIL in kernel execution (" << cudaGetErrorString(le) << ")" << std::endl;
				write_inj_record(-1); // the context is dead: host memory only
				exit(1);
			}

			std::string kname = removeSpaces(nvbit_get_func_name(ctx,func));
			if (inj_info.areParamsReady) {
				inj_info.areParamsReady = false;
				write_inj_record(counters[NUM_OPCODE_COUNTERS + inj_info.groupID]);

				// sanity check of the NVBit register read path (TOOL_VERBOSE=1):
				// the value NVBit passes as argument must equal nvbit_read_reg()
				if (verbose != 0 && shadow->regType == REG_GPR && inj_info.debug[1] != inj_info.debug[2]) {
					fout << "ERROR FAIL in kernel execution; Expected reg value doesn't match; \n";
					fout << "reg: R" << inj_info.debug[3] << ", expected_val: " << std::hex << inj_info.debug[1]
						<< ", nvbit_read_reg: " << inj_info.debug[2] << std::dec << "\n";
					std::cout << "NVBit-inject-error; ERROR FAIL: expected reg value doesn't match (R" << inj_info.debug[3] << ")\n";
				}
				fout.flush();
			}
			if (verbose) printf("\n index: %d; kernel_name: %s; used_instrumented=%d; \n", kernel_id, kname.c_str(), inj_info.debug[0]);
			kernel_id++; // always increment kernel_id on kernel exit

			cudaDeviceSynchronize();
			pthread_mutex_unlock(&mutex);
		}
	}
}

/* No record yet at normal termination: the selected kernel never completed
 * (e.g. wrong kernel count). Without this line such a run looks masked. The
 * shadow is not read here: no GPU resource is guaranteed to exist any more. */
void nvbit_at_term() {
	if (!record_written) {
		record_written = true;
		fout << "Error not injected" << std::endl;
	}
}
