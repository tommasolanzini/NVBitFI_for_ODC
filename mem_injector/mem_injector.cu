/*
 * NVBitFI_for_ODC (2026): memory-data injector, global and shared memory.
 *
 * Global memory: corrupts one bit of a live allocation where it is stored,
 * between kernels (timing mode a, host read-XOR-write) or during a kernel
 * (timing mode b, atomicXor at the N-th dynamic instruction).
 * Shared memory: corrupts one bit of the window of a CTA with atomicXor, right
 * after the N-th dynamic store (mode s) or on a sampled SM, offset and time
 * (mode u). See README.md.
 *
 * MEMINJ_PHASE selects what the tool does:
 *   timing: no instrumentation, log the CUDA events with their times
 *   count:  log the events and the dynamic instructions of every kernel
 *   inject (default): one flip, described by nvbitfi-injection-info.txt
 */

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <unordered_set>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

#include "globals.h"
#include "mem_injector.h"

#ifndef DUMMY
#define DUMMY 0
#endif

enum Phase { PHASE_TIMING, PHASE_COUNT, PHASE_INJECT };
static Phase phase = PHASE_INJECT;

enum EventKind { EV_NONE, EV_KERNEL, EV_ALLOC, EV_FREE, EV_H2D, EV_D2H, EV_D2D, EV_SET };
static const char *event_names[] = {"none", "kernel", "alloc", "free", "h2d", "d2h", "d2d", "memset"};

__managed__ mem_info_t info;
__managed__ mem_cta_t cta_table[MEM_MAX_CTAS];
__managed__ mem_sm_t sm_table[MEM_MAX_SMS];

static pthread_mutex_t mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static thread_local bool in_tool = false; // CUDA calls made by the tool itself are not events
static FILE *events_file = NULL;
static long long event_id = 0;
static long long ev_t0 = 0;

struct Buffer { uint64_t addr; size_t size; bool live; };
static std::vector<Buffer> buffers; // in allocation order: ids are stable across runs

// Target of the injection (inject phase)
static struct {
	std::string mode;        // "a" = between kernels, "b" = during a kernel, "s" = after a shared store, "u" = uniform in shared memory
	int bufferId = -1;
	size_t bufferSize = 0;
	size_t offset = 0;       // of the 32-bit word
	int bit = 0;
	long long event = -1;    // mode a: flip on entry of this event
	int kernel = -1;         // mode b: flip in this kernel
	std::string kernelName;
	unsigned long long instID = 0; // mode s: index among the shared stores
	int sm = -1;             // mode u
	unsigned shmOffset = 0;  // mode u: word in the window of the CTA
	unsigned pick = 0;       // mode u: picks a CTA among the resident ones
} target;

static bool is_shared_mode() { return target.mode == "s" || target.mode == "u"; }

static int unseen_shared_instrs = 0; // instructions that may reach shared memory without being traced

static bool flip_done = false;      // host view: flip applied (mode a) or kernel of the flip started (mode b)
static uint64_t flip_byte = 0;
static uint64_t word_addr = 0;
static bool target_freed = false;   // the target buffer has been freed

// Record in host-mapped memory, and its copy in plain host memory taken after
// every event: at nvbit_at_term the mapped memory may no longer exist.
static volatile mem_shadow_t *shadow = NULL;
static void *shadow_device_ptr = NULL;
static mem_shadow_t snapshot;
static bool record_written = false;
static bool loads_known = false;    // the two values below come from managed memory
static unsigned int loads_seen = 0;
static unsigned int sm_loads_seen = 0;

static long long now_ns() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}

static void init_shadow_fields(mem_shadow_t *s) {
	memset(s, 0, sizeof(*s));
	s->kernelIdx = -1;
	s->dynInstr = -1;
	s->pcOffset = -1;
	s->tid = -1;
	s->ctaFlat = -1;
	s->smID = -1;
	s->warpID = -1;
	s->laneID = -1;
	s->residents = -1;
	s->chosenCta = -1;
	s->chosenExits = -1;
}

static void ensure_shadow() {
	if (shadow) return;
	void *host_ptr = NULL;
	if (cudaHostAlloc(&host_ptr, sizeof(mem_shadow_t), cudaHostAllocMapped) != cudaSuccess ||
			cudaHostGetDevicePointer(&shadow_device_ptr, host_ptr, 0) != cudaSuccess) {
		fprintf(stderr, "mem_injector: cannot allocate the host-mapped record\n");
		exit(1);
	}
	mem_shadow_t init;
	init_shadow_fields(&init);
	memcpy(host_ptr, &init, sizeof(init));
	shadow = (volatile mem_shadow_t *)host_ptr;
}

static void take_snapshot() {
	if (shadow) memcpy(&snapshot, (const void *)shadow, sizeof(snapshot));
}

// Writes the record once. Host memory only: safe when the context is dead.
static void write_record(const mem_shadow_t &r) {
	if (record_written || phase != PHASE_INJECT) return;
	record_written = true;
	fout << "structure: " << (is_shared_mode() ? "shared" : "global") << "\n";
	fout << "mode: " << target.mode << "\n";
	if (is_shared_mode()) {
		fout << "targetSM: " << target.sm << "\n";
		fout << "unseenSharedInstrs: " << unseen_shared_instrs << "\n";
		fout << "residents: " << r.residents << "\n";
		fout << "chosenCta: " << r.chosenCta << "\n";
		if (r.state >= MEM_SELECTED) {
			fout << "shmWordOffset: " << r.shmWordOffset << "\n";
			fout << "shmMask: 0x" << std::hex << r.shmMask << std::dec << "\n";
		}
	}
	fout << "bufferId: " << target.bufferId << "\n";
	fout << "bufferSize: " << target.bufferSize << "\n";
	fout << "offset: " << target.offset << "\n";
	fout << "bit: " << target.bit << "\n";
	fout << "targetEvent: " << target.event << "\n";
	if (r.state >= MEM_SELECTED) {
		fout << "flipKernel: " << r.kernelIdx << "\n";
		fout << "dynInstr: " << r.dynInstr << "\n";
		fout << "pcOffset: 0x" << std::hex << r.pcOffset << std::dec << "\n";
		fout << "tid: " << r.tid << "\n";
		fout << "ctaID: " << r.ctaFlat << "\n";
		fout << "smID: " << r.smID << "\n";
		fout << "warpID: " << r.warpID << "\n";
		fout << "laneID: " << r.laneID << "\n";
	}
	if (r.state == MEM_INJECTED) {
		fout << "oldWord: 0x" << std::hex << r.oldWord << "\n";
		fout << "newWord: 0x" << r.newWord << std::dec << "\n";
	}
	fout << "readsOld: " << r.readsOld << "\n";
	fout << "readsNew: " << r.readsNew << "\n";
	if (loads_known) fout << "loads: " << loads_seen << "\n"; // managed memory: not available after a crash
	fout << "writesAfter: " << r.writesAfter << "\n";
	fout << "smOld: 0x" << std::hex << r.smOld << "\n";
	fout << "smNew: 0x" << r.smNew << "\n";
	fout << "smLoads: 0x" << sm_loads_seen << std::dec << "\n";
	fout << "firstAccess: " << r.firstAccess << "\n";
	fout << "hostReads: " << r.hostReads << "\n";
	fout << "hostWrites: " << r.hostWrites << "\n";
	fout << "readKernels: 0x" << std::hex << r.readKernels << std::dec << "\n";
	if (r.state != MEM_INJECTED) fout << "Error not injected\n";
	fout.flush();
}

static void write_record_from_shadow() {
	take_snapshot();
	write_record(snapshot);
}

static void INThandler(int sig) {
	signal(sig, SIG_IGN);
	fout << "ERROR FAIL Detected Signal SIGKILL\n";
	write_record_from_shadow();
	_exit(-1); // not exit(): the CUDA teardown can block if a kernel is hanging
}

static void parse_target() {
	std::ifstream ifs("nvbitfi-injection-info.txt");
	if (!ifs.is_open()) {
		printf("mem_injector: nvbitfi-injection-info.txt does not exist\n");
		exit(1);
	}
	std::string key;
	while (ifs >> key) {
		if (key == "mode") ifs >> target.mode;
		else if (key == "buffer") ifs >> target.bufferId;
		else if (key == "size") ifs >> target.bufferSize;
		else if (key == "offset") ifs >> target.offset;
		else if (key == "bit") ifs >> target.bit;
		else if (key == "event") ifs >> target.event;
		else if (key == "kernel") ifs >> target.kernel;
		else if (key == "kernel_name") ifs >> target.kernelName;
		else if (key == "inst") ifs >> target.instID;
		else if (key == "sm") ifs >> target.sm;
		else if (key == "shm_offset") ifs >> target.shmOffset;
		else if (key == "pick") ifs >> target.pick;
		else { printf("mem_injector: unknown key %s\n", key.c_str()); exit(1); }
	}
	assert(target.mode == "a" || target.mode == "b" || is_shared_mode());
	assert(target.offset % 4 == 0 && target.shmOffset % 4 == 0 && target.bit >= 0);
	assert(target.mode == "s" || target.bit < 32); // in mode s it only picks a bit of the stored bytes
}

void nvbit_at_init() {
	setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
	const char *ph = getenv("MEMINJ_PHASE");
	if (ph && !strcmp(ph, "timing")) phase = PHASE_TIMING;
	else if (ph && !strcmp(ph, "count")) phase = PHASE_COUNT;
	else phase = PHASE_INJECT;

	init_shadow_fields(&snapshot);
	if (phase == PHASE_INJECT) {
		signal(SIGINT, INThandler);
		open_output_file(injOutputFilename);
		parse_target();
	} else {
		events_file = fopen(getenv("MEMINJ_EVENTS") ? getenv("MEMINJ_EVENTS") : "nvbitfi-mem-events.jsonl", "w");
		assert(events_file);
	}
}

void nvbit_tool_init(CUcontext ctx) {
	if (phase != PHASE_INJECT) return;
	bool was = in_tool;
	in_tool = true;
	ensure_shadow();
	in_tool = was;
}

// The word and the byte of the flip, once the target buffer exists
static bool resolve_target() {
	if (word_addr) return true;
	if (target.bufferId < 0 || target.bufferId >= (int)buffers.size()) return false;
	const Buffer &b = buffers[target.bufferId];
	if (!b.live || b.size != target.bufferSize || target.offset + 4 > b.size) return false;
	word_addr = b.addr + target.offset;
	flip_byte = word_addr + target.bit / 8; // little endian
	return true;
}

// Mode a: host read-XOR-write, on entry of the target event
static void host_flip() {
	if (!resolve_target()) return; // the record stays MEM_NONE: "Error not injected"
	ensure_shadow();
	cudaDeviceSynchronize();
	unsigned old = 0;
	const unsigned applied = DUMMY ? 0u : (1u << target.bit);
	shadow->state = MEM_SELECTED;
	if (cudaMemcpy(&old, (const void *)word_addr, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
	const unsigned neu = old ^ applied;
	shadow->oldWord = old;
	shadow->newWord = neu;
	if (applied && cudaMemcpy((void *)word_addr, &neu, 4, cudaMemcpyHostToDevice) != cudaSuccess) return;
	shadow->state = MEM_INJECTED;
	flip_done = true;
}

static std::unordered_set<CUfunction> already_instrumented;

static bool is_shared_store(Instr *i) {
	return i->getMemorySpace() == InstrType::MemorySpace::SHARED && i->isStore() && !i->isLoad();
}

static void add_shared_trace_call(Instr *i) {
	const InstrType::MemorySpace s = i->getMemorySpace();
	if (s == InstrType::MemorySpace::GENERIC || s == InstrType::MemorySpace::GLOBAL_TO_SHARED || s == InstrType::MemorySpace::DISTRIBUTED_SHARED) {
		unseen_shared_instrs++;
		return;
	}
	if (s != InstrType::MemorySpace::SHARED) return;
	const int flags = (i->isLoad() ? MEM_FLAG_LOAD : 0) | (i->isStore() ? MEM_FLAG_STORE : 0);
	if (!flags) return;
	nvbit_insert_call(i, "shm_trace_access", IPOINT_BEFORE);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&info);
	nvbit_add_call_arg_const_val64(i, (uint64_t)shadow_device_ptr);
	nvbit_add_call_arg_guard_pred_val(i);
	nvbit_add_call_arg_mref_addr64(i, 0);
	nvbit_add_call_arg_const_val32(i, std::max(i->getSize(), 1));
	nvbit_add_call_arg_const_val32(i, flags);
	if (flags == MEM_FLAG_STORE) {
		nvbit_insert_call(i, "shm_trace_store_after", IPOINT_AFTER);
		nvbit_add_call_arg_const_val64(i, (uint64_t)&info);
		nvbit_add_call_arg_const_val64(i, (uint64_t)shadow_device_ptr);
		nvbit_add_call_arg_guard_pred_val(i);
		nvbit_add_call_arg_mref_addr64(i, 0);
		nvbit_add_call_arg_const_val32(i, std::max(i->getSize(), 1));
	}
}

static void add_flip_after_store(Instr *i) {
	nvbit_insert_call(i, "shm_flip_after_store", IPOINT_AFTER);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&info);
	nvbit_add_call_arg_const_val64(i, (uint64_t)shadow_device_ptr);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&counters[0]);
	nvbit_add_call_arg_const_val32(i, i->getOffset());
	nvbit_add_call_arg_guard_pred_val(i);
	nvbit_add_call_arg_mref_addr64(i, 0);
	nvbit_add_call_arg_const_val32(i, std::max(i->getSize(), 1));
}

static void add_flip_uniform(Instr *i) {
	nvbit_insert_call(i, "shm_flip_uniform", IPOINT_BEFORE);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&info);
	nvbit_add_call_arg_const_val64(i, (uint64_t)shadow_device_ptr);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&counters[0]);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&sm_table[0]);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&cta_table[0]);
	nvbit_add_call_arg_const_val32(i, i->getOffset());
	nvbit_add_call_arg_guard_pred_val(i);
}

static void add_cta_call(Instr *i, const char *fn) {
	nvbit_insert_call(i, fn, IPOINT_BEFORE);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&info);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&cta_table[0]);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&sm_table[0]);
	nvbit_add_call_arg_guard_pred_val(i);
}

// Count phase: stores to shared memory in counters[1], residency of the CTAs
static void add_profile_calls(Instr *i) {
	nvbit_insert_call(i, "shm_residency", IPOINT_BEFORE);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&cta_table[0]);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&counters[0]);
	nvbit_add_call_arg_guard_pred_val(i);
	if (is_shared_store(i)) {
		nvbit_insert_call(i, "shm_count_store", IPOINT_BEFORE);
		nvbit_add_call_arg_const_val64(i, (uint64_t)&counters[1]);
		nvbit_add_call_arg_guard_pred_val(i);
	}
}

static void add_trace_call(Instr *i) {
	const InstrType::MemorySpace s = i->getMemorySpace();
	if (s != InstrType::MemorySpace::GLOBAL && s != InstrType::MemorySpace::GENERIC) return;
	int flags = (i->isLoad() ? MEM_FLAG_LOAD : 0) | (i->isStore() ? MEM_FLAG_STORE : 0);
	if (!flags) return;
	for (int n = 0; n < i->getNumOperands(); n++) {
		if (i->getOperand(n)->type != InstrType::OperandType::MREF) continue;
		nvbit_insert_call(i, "mem_trace_access", IPOINT_BEFORE);
		nvbit_add_call_arg_const_val64(i, (uint64_t)&info);
		nvbit_add_call_arg_const_val64(i, (uint64_t)shadow_device_ptr);
		nvbit_add_call_arg_guard_pred_val(i);
		nvbit_add_call_arg_mref_addr64(i, 0);
		nvbit_add_call_arg_const_val32(i, std::max(i->getSize(), 1));
		nvbit_add_call_arg_const_val32(i, flags);
		break; // one memory reference per instruction
	}
}

static void add_count_call(Instr *i) {
	nvbit_insert_call(i, "mem_count_flip", IPOINT_BEFORE);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&info);
	nvbit_add_call_arg_const_val64(i, (uint64_t)shadow_device_ptr);
	nvbit_add_call_arg_const_val64(i, (uint64_t)&counters[0]);
	nvbit_add_call_arg_const_val32(i, i->getOffset());
	nvbit_add_call_arg_guard_pred_val(i);
}

static void instrument(CUcontext ctx, CUfunction func) {
	std::vector<CUfunction> fs = nvbit_get_related_functions(ctx, func);
	fs.push_back(func);
	for (auto f : fs) {
		if (!already_instrumented.insert(f).second) continue;
		for (auto i : nvbit_get_instrs(ctx, f)) {
			// every kernel can be the target: the armed flag decides at run time
			if (phase == PHASE_COUNT) {
				add_count_call(i);
				add_profile_calls(i);
			} else if (target.mode == "u") {
				// this order: a CTA is registered before its first flip call, and
				// a thread is counted as gone only after its last one
				if (f == func && i->getOffset() == 0) add_cta_call(i, "shm_cta_enter");
				add_flip_uniform(i);
				if (!strncmp(i->getOpcode(), "EXIT", 4)) add_cta_call(i, "shm_cta_exit");
				add_shared_trace_call(i);
			} else if (target.mode == "s") {
				// this order: the store that triggers the flip must not look ambiguous
				add_shared_trace_call(i);
				if (is_shared_store(i)) add_flip_after_store(i);
			} else {
				add_count_call(i);
				add_trace_call(i);
			}
		}
	}
}

// Host state -> managed memory, only while no kernel runs
static void sync_info(bool arm) {
	cudaDeviceSynchronize();
	info.armed = arm;
	info.kernelIdx = kernel_id;
	if (is_shared_mode()) {
		// the window of a CTA does not outlive the kernel: trace only this one
		info.traceOn = arm;
		if (arm) {
			info.instID = target.instID;
			info.targetSM = target.sm;
			info.bitSeed = target.bit;
			info.flipOffsetTarget = target.shmOffset;
			info.applyFlip = DUMMY ? 0 : 1;
			info.pickSeed = target.pick;
			info.chosenCta = CTA_NOT_CHOSEN;
		}
		cudaDeviceSynchronize();
		return;
	}
	if (word_addr) {
		info.wordAddr = word_addr;
		info.flipByteAddr = flip_byte;
		info.mask = 1u << target.bit;
		info.appliedMask = DUMMY ? 0u : (1u << target.bit);
		info.instID = target.instID;
	}
	if (target.mode == "a" && flip_done) {
		info.fired = FLIP_DONE;
		info.oldWord = shadow->oldWord;
		info.traceOn = 1;
	}
	if (target.mode == "b" && flip_done) info.traceOn = 1;
	if (target_freed) info.traceOn = 0;
	cudaDeviceSynchronize();
}

static void on_kernel_entry(CUcontext ctx, nvbit_api_cuda_t cbid, void *params) {
	CUfunction func = get_launched_function(cbid, params);
	if (phase == PHASE_TIMING) {
		nvbit_enable_instrumented(ctx, func, false);
		return;
	}
	ensure_shadow();
	instrument(ctx, func);
	cudaDeviceSynchronize();
	init_counters();
	if (phase == PHASE_COUNT || target.mode == "u") {
		memset((void *)cta_table, 0, sizeof(cta_table));
		memset((void *)sm_table, 0, sizeof(sm_table));
	}

	bool arm = false;
	if (phase == PHASE_COUNT) {
		info.instID = ~0ull; // never reached: only count
		arm = true;
	} else if (target.mode != "a" && kernel_id == target.kernel && !flip_done) {
		std::string launched = removeSpaces(nvbit_get_func_name(ctx, func));
		if (launched == target.kernelName && (is_shared_mode() || resolve_target())) {
			arm = true;
			flip_done = true;
		}
	}
	sync_info(arm);
	nvbit_enable_instrumented(ctx, func, true);
	cudaDeviceSynchronize();
}

static void log_event(EventKind kind, long long t1, const std::string &extra) {
	if (!events_file) return;
	fprintf(events_file, "{\"e\": %lld, \"kind\": \"%s\", \"t0\": %lld, \"t1\": %lld%s}\n",
			event_id, event_names[kind], ev_t0, t1, extra.c_str());
	fflush(events_file);
}

static void on_kernel_exit(CUcontext ctx, nvbit_api_cuda_t cbid, void *params) {
	CUfunction func = get_launched_function(cbid, params);
	cudaDeviceSynchronize();
	const long long t1 = now_ns();

	cudaError_t le = cudaGetLastError();
	if (le != cudaSuccess) {
		std::cout << "ERROR FAIL in kernel execution (" << cudaGetErrorString(le) << "); ";
		if (phase == PHASE_INJECT) {
			loads_known = false; // the value in hand is from an earlier kernel
			fout << "ERROR FAIL in kernel execution (" << cudaGetErrorString(le) << ")" << std::endl;
			write_record_from_shadow(); // the context is dead: host memory only
		}
		exit(1);
	}

	if (phase == PHASE_INJECT && target.mode == "u" && kernel_id == target.kernel) {
		// Check of the residency: at the end of a kernel every CTA has left its SM
		int entered = 0, slots_in_use = 0;
		unsigned exits_min = ~0u, exits_max = 0;
		for (int c = 0; c < MEM_MAX_CTAS; c++) {
			if (!cta_table[c].first) continue;
			entered++;
			exits_min = std::min(exits_min, cta_table[c].exits);
			exits_max = std::max(exits_max, cta_table[c].exits);
		}
		for (int s = 0; s < MEM_MAX_SMS; s++) slots_in_use += __builtin_popcount(sm_table[s].mask);
		fout << "ctasEntered: " << entered << "\nctaExitsMin: " << exits_min << "\nctaExitsMax: " << exits_max << "\nslotsInUse: " << slots_in_use << "\n";
		fout << "chosenAt: " << info.chooserInstr << "\nchosenExitsAtChoice: " << shadow->chosenExits << "\n";
	}
	if (phase == PHASE_INJECT) {
		loads_seen = info.loads;
		sm_loads_seen = info.smLoads;
		loads_known = true;
	}
	std::string extra = ", \"k\": " + std::to_string(kernel_id) + ", \"name\": \"" +
		removeSpaces(nvbit_get_func_name(ctx, func)) + "\"";
	if (phase == PHASE_COUNT) {
		extra += ", \"instrs\": " + std::to_string(counters[0]);
		func_config_t cfg;
		nvbit_get_func_config(ctx, func, &cfg);
		const unsigned shm = cfg.shmem_static_nbytes + cfg.shmem_dynamic_nbytes;
		if (shm) {
			extra += ", \"shm\": " + std::to_string(shm) + ", \"stores\": " + std::to_string(counters[1]) + ", \"ctas\": [";
			bool first = true;
			for (int c = 0; c < MEM_MAX_CTAS; c++) {
				if (!cta_table[c].first) continue;
				extra += std::string(first ? "" : ", ") + "[" + std::to_string(cta_table[c].sm) + ", " +
					std::to_string(cta_table[c].first) + ", " + std::to_string(cta_table[c].last) + "]";
				first = false;
			}
			extra += "]";
		}
	}
	log_event(EV_KERNEL, t1, extra);
	kernel_id++;
}

static EventKind classify(nvbit_api_cuda_t cbid) {
	if (is_kernel_launch(cbid)) return EV_KERNEL;
	switch (cbid) {
		case API_CUDA_cuMemAlloc_v2:
		case API_CUDA_cuMemAllocManaged:
			return EV_ALLOC;
		case API_CUDA_cuMemFree_v2:
			return EV_FREE;
		case API_CUDA_cuMemcpyHtoD_v2:
		case API_CUDA_cuMemcpyHtoDAsync_v2:
			return EV_H2D;
		case API_CUDA_cuMemcpyDtoH_v2:
		case API_CUDA_cuMemcpyDtoHAsync_v2:
			return EV_D2H;
		case API_CUDA_cuMemcpyDtoD_v2:
		case API_CUDA_cuMemcpyDtoDAsync_v2:
			return EV_D2D;
		case API_CUDA_cuMemsetD8_v2:
		case API_CUDA_cuMemsetD8Async:
		case API_CUDA_cuMemsetD16_v2:
		case API_CUDA_cuMemsetD16Async:
		case API_CUDA_cuMemsetD32_v2:
		case API_CUDA_cuMemsetD32Async:
			return EV_SET;
		default:
			return EV_NONE;
	}
}

// Device range(s) touched by a copy or memset: {src, dst, size}; 0 = none
struct Range { uint64_t src = 0, dst = 0; size_t size = 0; };

static Range get_range(nvbit_api_cuda_t cbid, void *params) {
	Range r;
	switch (cbid) {
		case API_CUDA_cuMemcpyHtoD_v2: { auto p = (cuMemcpyHtoD_v2_params *)params; r.dst = p->dstDevice; r.size = p->ByteCount; break; }
		case API_CUDA_cuMemcpyHtoDAsync_v2: { auto p = (cuMemcpyHtoDAsync_v2_params *)params; r.dst = p->dstDevice; r.size = p->ByteCount; break; }
		case API_CUDA_cuMemcpyDtoH_v2: { auto p = (cuMemcpyDtoH_v2_params *)params; r.src = p->srcDevice; r.size = p->ByteCount; break; }
		case API_CUDA_cuMemcpyDtoHAsync_v2: { auto p = (cuMemcpyDtoHAsync_v2_params *)params; r.src = p->srcDevice; r.size = p->ByteCount; break; }
		case API_CUDA_cuMemcpyDtoD_v2: { auto p = (cuMemcpyDtoD_v2_params *)params; r.src = p->srcDevice; r.dst = p->dstDevice; r.size = p->ByteCount; break; }
		case API_CUDA_cuMemcpyDtoDAsync_v2: { auto p = (cuMemcpyDtoDAsync_v2_params *)params; r.src = p->srcDevice; r.dst = p->dstDevice; r.size = p->ByteCount; break; }
		case API_CUDA_cuMemsetD8_v2: { auto p = (cuMemsetD8_v2_params *)params; r.dst = p->dstDevice; r.size = p->N; break; }
		case API_CUDA_cuMemsetD8Async: { auto p = (cuMemsetD8Async_params *)params; r.dst = p->dstDevice; r.size = p->N; break; }
		case API_CUDA_cuMemsetD16_v2: { auto p = (cuMemsetD16_v2_params *)params; r.dst = p->dstDevice; r.size = p->N * 2; break; }
		case API_CUDA_cuMemsetD16Async: { auto p = (cuMemsetD16Async_params *)params; r.dst = p->dstDevice; r.size = p->N * 2; break; }
		case API_CUDA_cuMemsetD32_v2: { auto p = (cuMemsetD32_v2_params *)params; r.dst = p->dstDevice; r.size = p->N * 4; break; }
		case API_CUDA_cuMemsetD32Async: { auto p = (cuMemsetD32Async_params *)params; r.dst = p->dstDevice; r.size = p->N * 4; break; }
		default: break;
	}
	return r;
}

static void account_host_access(const Range &r) {
	if (!flip_done || !word_addr || target_freed || !shadow || shadow->state != MEM_INJECTED) return;
	const bool rd = r.src && r.src <= flip_byte && flip_byte < r.src + r.size;
	const bool wr = r.dst && r.dst <= flip_byte && flip_byte < r.dst + r.size;
	if (rd) {
		shadow->hostReads = shadow->hostReads + 1;
		if (shadow->firstAccess == ACCESS_NONE) shadow->firstAccess = ACCESS_READ;
	}
	if (wr) {
		shadow->hostWrites = shadow->hostWrites + 1;
		if (shadow->firstAccess == ACCESS_NONE) shadow->firstAccess = ACCESS_WRITE;
	}
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
		const char *name, void *params, CUresult *pStatus) {
	if (in_tool) return;
	const EventKind kind = classify(cbid);
	if (kind == EV_NONE) return;

	pthread_mutex_lock(&mutex);
	in_tool = true;

	if (!is_exit) {
		ev_t0 = now_ns();
		if (phase == PHASE_INJECT && target.mode == "a" && event_id == target.event && !flip_done) host_flip();
		if (phase == PHASE_INJECT) {
			Range r = get_range(cbid, params);
			account_host_access(r);
		}
		if (kind == EV_KERNEL) on_kernel_entry(ctx, cbid, params);
		ev_t0 = now_ns(); // tool work is not part of the event
	} else {
		std::string extra;
		if (kind == EV_KERNEL) {
			on_kernel_exit(ctx, cbid, params);
		} else {
			const long long t1 = now_ns();
			if (kind == EV_ALLOC) {
				uint64_t addr = 0;
				size_t size = 0;
				if (cbid == API_CUDA_cuMemAlloc_v2) { auto p = (cuMemAlloc_v2_params *)params; addr = *p->dptr; size = p->bytesize; }
				else { auto p = (cuMemAllocManaged_params *)params; addr = *p->dptr; size = p->bytesize; }
				extra = ", \"id\": " + std::to_string(buffers.size()) + ", \"size\": " + std::to_string(size);
				buffers.push_back({addr, size, true});
			} else if (kind == EV_FREE) {
				const uint64_t addr = ((cuMemFree_v2_params *)params)->dptr;
				for (size_t b = 0; b < buffers.size(); b++) {
					if (buffers[b].live && buffers[b].addr == addr) {
						buffers[b].live = false;
						extra = ", \"id\": " + std::to_string(b);
						// the address can be reused: stop tracing the freed word
						if ((int)b == target.bufferId) target_freed = true;
					}
				}
			} else {
				const Range r = get_range(cbid, params);
				extra = ", \"src\": " + std::to_string(r.src) + ", \"dst\": " + std::to_string(r.dst) + ", \"size\": " + std::to_string(r.size);
			}
			log_event(kind, t1, extra);
		}
		event_id++;
		take_snapshot();
	}

	in_tool = false;
	pthread_mutex_unlock(&mutex);
}

// Normal end of the run: no GPU call, the shadow may not exist any more
void nvbit_at_term() {
	if (phase == PHASE_INJECT) {
		write_record(snapshot);
	} else if (events_file) {
		fprintf(events_file, "{\"e\": %lld, \"kind\": \"term\", \"t0\": %lld, \"t1\": %lld}\n", event_id, now_ns(), now_ns());
		fclose(events_file);
	}
}
