/*
 * Minimal NVBit tool for the memory descriptor experiment. Unlike the injector
 * it instruments only the LDCU that loads the chosen constant, so it adds no
 * overhead to the loop being timed.
 *
 * DESC_MATCH    text the instruction must contain (default c[0x0][0x358])
 * DESC_MASK_LO  XOR mask for the destination register (hex, default 0)
 * DESC_MASK_HI  XOR mask for the next register, LDCU.64 only (hex, default 0)
 * Prints "ldcu <reg> before <hi:lo> after <hi:lo>" on stderr at the end.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "nvbit_tool.h"
#include "nvbit.h"

static std::string match = "c[0x0][0x358]";
static uint32_t mask_lo = 0, mask_hi = 0;
static volatile uint32_t *record = NULL;
static void *record_dev = NULL;
static int found_reg = -1;
static bool done = false;

void nvbit_at_init() {
	if (getenv("DESC_MATCH")) match = getenv("DESC_MATCH");
	if (getenv("DESC_MASK_LO")) mask_lo = strtoul(getenv("DESC_MASK_LO"), NULL, 16);
	if (getenv("DESC_MASK_HI")) mask_hi = strtoul(getenv("DESC_MASK_HI"), NULL, 16);
}

void nvbit_tool_init(CUcontext ctx) {
	void *p = NULL;
	if (cudaHostAlloc(&p, 5 * sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess ||
			cudaHostGetDevicePointer(&record_dev, p, 0) != cudaSuccess) {
		fprintf(stderr, "desc_tool: cannot allocate the record\n");
		exit(1);
	}
	record = (volatile uint32_t *)p;
	for (int i = 0; i < 5; i++) record[i] = 0;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name, void *params, CUresult *pStatus) {
	if (is_exit || done || (cbid != API_CUDA_cuLaunchKernel && cbid != API_CUDA_cuLaunchKernelEx && cbid != API_CUDA_cuLaunchKernel_ptsz && cbid != API_CUDA_cuLaunchKernelEx_ptsz)) return;
	done = true;
	CUfunction f = (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz)
		? ((cuLaunchKernel_params *)params)->f : ((cuLaunchKernelEx_params *)params)->f;
	for (auto i : nvbit_get_instrs(ctx, f)) {
		std::string sass = i->getSass();
		if (sass.rfind("LDCU", 0) != 0 || sass.find(match) == std::string::npos) continue;
		found_reg = i->getOperand(0)->u.reg.num;
		nvbit_insert_call(i, "xor_ureg", IPOINT_AFTER);
		nvbit_add_call_arg_const_val32(i, found_reg);
		nvbit_add_call_arg_const_val32(i, sass.rfind("LDCU.64", 0) == 0);
		nvbit_add_call_arg_const_val32(i, mask_lo);
		nvbit_add_call_arg_const_val32(i, mask_hi);
		nvbit_add_call_arg_const_val64(i, (uint64_t)record_dev);
		break;
	}
	nvbit_enable_instrumented(ctx, f, true);
}

void nvbit_at_term() {
	if (record && record[4])
		fprintf(stderr, "ldcu %d before %08x:%08x after %08x:%08x\n", found_reg, record[1], record[0], record[3], record[2]);
	else
		fprintf(stderr, "ldcu not reached (reg %d)\n", found_reg);
}
