/*
 * Host part of the shared memory probe tool, see ../README.md.
 *
 * PROBE_MODE:
 *   addr        record the addresses NVBit gives for shared loads and stores,
 *               before and after the instruction
 *   flip_store  flip the word of the store to address PROBE_TRIGGER, at
 *               IPOINT_AFTER, bit PROBE_BIT
 *   flip_offset same trigger, but flip the word at offset PROBE_OFFSET
 *   xor_atomic  XOR the word at PROBE_OFFSET with atomicXor on every global load
 *   xor_plain   same with a plain read-modify-write, as a control
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_set>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"
#include "globals.h"
#include "probe.h"

__managed__ probe_state_t st;

static std::string mode;
static std::unordered_set<CUfunction> done;

static unsigned long long env_ull(const char *name) {
	const char *v = getenv(name);
	return v ? strtoull(v, NULL, 0) : 0;
}

void nvbit_at_init() {
	setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
	mode = getenv("PROBE_MODE") ? getenv("PROBE_MODE") : "addr";
}

static void add_addr_call(Instr *i) {
	nvbit_add_call_arg_const_val64(i, (uint64_t)&st);
	nvbit_add_call_arg_mref_addr64(i, 0);
}

static void instrument(CUcontext ctx, CUfunction func) {
	for (auto i : nvbit_get_instrs(ctx, func)) {
		const InstrType::MemorySpace s = i->getMemorySpace();
		const bool shared = s == InstrType::MemorySpace::SHARED;
		const bool global = s == InstrType::MemorySpace::GLOBAL;
		if (shared || mode == "addr") {
			if (getenv("PROBE_VERBOSE")) printf("{\"instr\": \"%s\", \"space\": %d, \"size\": %d}\n", i->getOpcode(), (int)s, i->getSize());
		}
		if (mode == "addr" && shared) {
			const int kind = i->isStore() ? 1 : 0;
			nvbit_insert_call(i, "probe_record", IPOINT_BEFORE);
			add_addr_call(i);
			nvbit_add_call_arg_const_val32(i, kind);
			nvbit_add_call_arg_guard_pred_val(i);
			nvbit_insert_call(i, "probe_record", IPOINT_AFTER);
			add_addr_call(i);
			nvbit_add_call_arg_const_val32(i, kind + 2);
			nvbit_add_call_arg_guard_pred_val(i);
		} else if ((mode == "flip_store" || mode == "flip_offset") && shared && i->isStore()) {
			nvbit_insert_call(i, "probe_flip", IPOINT_AFTER);
			add_addr_call(i);
			nvbit_add_call_arg_guard_pred_val(i);
		} else if ((mode == "xor_atomic" || mode == "xor_plain") && global && i->isLoad()) {
			nvbit_insert_call(i, "probe_xor", IPOINT_BEFORE);
			nvbit_add_call_arg_const_val64(i, (uint64_t)&st);
			nvbit_add_call_arg_const_val32(i, mode == "xor_plain");
			nvbit_add_call_arg_guard_pred_val(i);
		}
	}
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name, void *params, CUresult *pStatus) {
	if (!is_kernel_launch(cbid)) return;
	CUfunction func = get_launched_function(cbid, params);
	if (!is_exit) {
		if (done.insert(func).second) {
			memset((void *)&st, 0, sizeof(st));
			st.trigger = env_ull("PROBE_TRIGGER");
			st.offset = env_ull("PROBE_OFFSET");
			st.mask = 1u << env_ull("PROBE_BIT");
			st.use_offset = mode == "flip_offset";
			printf("{\"shmem_base\": %llu}\n", (unsigned long long)nvbit_get_shmem_base_addr(ctx));
			func_config_t cfg;
			nvbit_get_func_config(ctx, func, &cfg);
			printf("{\"kernel\": \"%s\", \"static_bytes\": %u, \"dynamic_bytes\": %u}\n", nvbit_get_func_name(ctx, func), cfg.shmem_static_nbytes, cfg.shmem_dynamic_nbytes);
			instrument(ctx, func);
		}
		nvbit_enable_instrumented(ctx, func, true);
		return;
	}
	cudaDeviceSynchronize();
	static const char *slot[] = {"load_before", "store_before", "load_after", "store_after"};
	for (int s = 0; s < PROBE_SLOTS; s++) {
		printf("{\"slot\": \"%s\", \"count\": %u, \"addrs\": [", slot[s], st.count[s]);
		for (unsigned n = 0; n < PROBE_SAMPLES && n < st.count[s]; n++) printf("%s%llu", n ? ", " : "", st.addr[s][n]);
		printf("]}\n");
	}
	printf("{\"fired\": %d, \"old\": %u, \"new\": %u, \"flip_addr\": %llu, \"cta\": %d, \"tid\": %d}\n", st.fired, st.old_word, st.new_word, st.flip_addr, st.flip_cta, st.flip_tid);
	fflush(stdout);
}
