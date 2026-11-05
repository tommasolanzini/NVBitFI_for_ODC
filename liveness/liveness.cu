/*
 * Register liveness tool. For every value written to a GPR, predicate register,
 * uniform register or uniform predicate: the time from the write to the last read
 * before the next write, in instructions and cycles. One JSON line per kernel launch
 * in liveness.jsonl (or $LIVE_OUT). scripts/liveness_report.py reads it.
 */
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <map>
#include <string>
#include <vector>
#include <unordered_set>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "globals.h"
#include "utils/utils.h"
#include "liveness.h"

pthread_mutex_t mutex;

__managed__ live_ctx_t ctx_dev;

struct SiteInfo {
	std::string kernel, sass;
	uint32_t pc;
	int cls, reg, opcode, group;
};

static std::vector<live_instr_t> host_instrs;
static std::vector<SiteInfo> host_sites;
static std::unordered_set<CUfunction> instrumented;
static std::map<std::string, int> launches;
static std::map<CUfunction, UsedRegs> used_of;
static live_instr_t *dev_instrs;
static live_site_t *dev_sites;
static FILE *out;

static void check(cudaError_t e, const char *what) {
	if (e != cudaSuccess) {
		fprintf(stderr, "liveness: %s failed: %s\n", what, cudaGetErrorString(e));
		exit(1);
	}
}

void nvbit_at_init() {
	setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
	const char *path = getenv("LIVE_OUT");
	out = fopen(path ? path : "liveness.jsonl", "w");
	assert(out);
}

static void set_bit(uint64_t m[2], int r) {
	if (r >= LIVE_MAX_REGS) {
		fprintf(stderr, "liveness: register %d beyond LIVE_MAX_REGS\n", r);
		exit(1);
	}
	m[r / 64] |= 1ULL << (r % 64);
}

static uint32_t add_sites(const std::string &kname, Instr *i, const StaticInstInfo &si, int cls, const std::vector<int> &regs) {
	const uint32_t first = host_sites.size();
	for (int r : regs) host_sites.push_back({kname, i->getSass(), i->getOffset(), cls, r, si.opcode, si.group});
	return first;
}

static void instrument(CUcontext ctx, CUfunction func) {
	std::vector<CUfunction> funcs = nvbit_get_related_functions(ctx, func);
	funcs.push_back(func);
	const std::string kname = removeSpaces(nvbit_get_func_name(ctx, func));

	// One register layout for the kernel and what it calls: they share the registers
	UsedRegs used;
	for (auto f : funcs) {
		const UsedRegs u = used_regs(nvbit_get_instrs(ctx, f));
		used.gpr = std::max(used.gpr, u.gpr);
		used.ur = std::max(used.ur, u.ur);
	}
	used_of[func] = used;

	for (auto f : funcs) {
		if (!instrumented.insert(f).second) continue;
		for (Instr *i : nvbit_get_instrs(ctx, f)) {
			const StaticInstInfo si = analyze_instr(i);
			if (si.opcode == UNKNOWN_OPCODE) warn_unknown_opcode(i);
			const SrcInfo src = instr_sources(i, si);

			live_instr_t in = {};
			in.guardPr = in.guardUp = -1;
			if (i->hasPred() && i->getPredNum() < PT_NUM) {
				(i->isPredUniform() ? in.guardUp : in.guardPr) = i->getPredNum();
			}
			for (int r : src.gpr) set_bit(in.gprR, r);
			for (int r : src.ur) set_bit(in.urR, r);
			in.prR = src.pr_mask;
			in.upR = src.up_mask;

			std::vector<int> g, p, u, up;
			for (int k = 0; k < si.dests.num_gpr; k++) {
				g.push_back(si.dests.gpr_base + k);
				set_bit(in.gprW, si.dests.gpr_base + k);
			}
			for (int k = 0; k < si.dests.num_ur; k++) {
				u.push_back(si.dests.ur_base + k);
				set_bit(in.urW, si.dests.ur_base + k);
			}
			for (int k = 0; k < LIVE_NUM_PRED; k++) {
				if (si.dests.pr_mask >> k & 1) p.push_back(k);
				if (si.dests.up_mask >> k & 1) up.push_back(k);
			}
			in.prW = si.dests.pr_mask;
			in.upW = si.dests.up_mask;
			in.siteG = add_sites(kname, i, si, LIVE_GPR, g);
			in.siteP = add_sites(kname, i, si, LIVE_PR, p);
			in.siteU = add_sites(kname, i, si, LIVE_UR, u);
			in.siteUP = add_sites(kname, i, si, LIVE_UP, up);
			if (host_instrs.size() >= LIVE_MAX_INSTRS || host_sites.size() >= LIVE_MAX_SITES) {
				fprintf(stderr, "liveness: too many instructions or sites\n");
				exit(1);
			}

			nvbit_insert_call(i, "live_track", IPOINT_BEFORE);
			nvbit_add_call_arg_const_val64(i, (uint64_t)&ctx_dev);
			nvbit_add_call_arg_const_val32(i, (uint32_t)host_instrs.size());
			nvbit_add_call_arg_guard_pred_val(i);
			host_instrs.push_back(in);
		}
	}
}

// Threads in a CTA of this launch, 0 if the API does not say
static uint32_t threads_per_cta(nvbit_api_cuda_t cbid, void *params) {
	if (is_launch_ex(cbid)) {
		const CUlaunchConfig *c = ((cuLaunchKernelEx_params *)params)->config;
		return c->blockDimX * c->blockDimY * c->blockDimZ;
	}
	if (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz ||
			cbid == API_CUDA_cuLaunchCooperativeKernel || cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
		cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
		return p->blockDimX * p->blockDimY * p->blockDimZ;
	}
	return 0;
}

static std::string json_escape(const std::string &s) {
	std::string o;
	for (char c : s) {
		if (c == '"' || c == '\\') o += '\\';
		if ((unsigned char)c >= 32) o += c;
	}
	return o;
}

template <typename T>
static T *dev_alloc(size_t n, const char *what) {
	T *p = nullptr;
	check(cudaMalloc(&p, n * sizeof(T)), what);
	check(cudaMemset(p, 0, n * sizeof(T)), what);
	return p;
}

static void before_launch(CUcontext ctx, CUfunction func, nvbit_api_cuda_t cbid, void *params) {
	instrument(ctx, func);
	if (!dev_instrs) {
		dev_instrs = dev_alloc<live_instr_t>(LIVE_MAX_INSTRS, "instruction table");
		dev_sites = dev_alloc<live_site_t>(LIVE_MAX_SITES, "site table");
	}
	check(cudaMemcpy(dev_instrs, host_instrs.data(), host_instrs.size() * sizeof(live_instr_t), cudaMemcpyHostToDevice), "copy instructions");
	check(cudaMemset(dev_sites, 0, host_sites.size() * sizeof(live_site_t)), "reset sites");

	const uint32_t tpc = threads_per_cta(cbid, params);
	const uint64_t ctas = get_num_ctas(cbid, params);
	if (!tpc || !ctas) {
		fprintf(stderr, "liveness: launch API %d does not carry the grid\n", (int)cbid);
		exit(1);
	}
	const UsedRegs u = used_of[func];
	const uint32_t wpc = (tpc + 31) / 32;
	const uint64_t threads = ctas * tpc, warps = ctas * wpc;

	cudaDeviceSynchronize();
	ctx_dev.instrs = dev_instrs;
	ctx_dev.sites = dev_sites;
	ctx_dev.numGpr = u.gpr;
	ctx_dev.numUr = u.ur;
	ctx_dev.threadsPerCta = tpc;
	ctx_dev.warpsPerCta = wpc;
	ctx_dev.threadSlots = dev_alloc<live_slot_t>(threads * (u.gpr + LIVE_NUM_PRED), "thread slots");
	ctx_dev.warpSlots = dev_alloc<live_slot_t>(warps * (u.ur + LIVE_NUM_PRED), "warp slots");
	ctx_dev.threadTime = dev_alloc<live_time_t>(threads, "thread times");
	ctx_dev.warpTime = dev_alloc<live_time_t>(warps, "warp times");
	for (int k = 0; k < LIVE_CLASSES; k++) ctx_dev.readsNoWrite[k] = 0;
	ctx_dev.laneMismatch = 0;
}

template <typename T>
static std::vector<T> dev_read(T *p, size_t n) {
	std::vector<T> v(n);
	check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "read back");
	check(cudaFree(p), "free");
	return v;
}

// Values still in the registers when the thread or warp ended
static void close_slots(const std::vector<live_slot_t> &slots, std::vector<live_site_t> &sites) {
	for (const live_slot_t &s : slots) {
		if (!s.site) continue;
		live_site_t &o = sites[s.site - 1];
		o.count++;
		if (s.reads == 0) {
			o.dead++;
		} else {
			o.live += s.rTime - s.wTime;
			o.liveCycles += (uint32_t)(s.rCyc - s.wCyc);
			o.reads += s.reads;
		}
	}
}

static void after_launch(CUcontext ctx, CUfunction func, nvbit_api_cuda_t cbid, void *params) {
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "liveness: the kernel failed\n");
		exit(1);
	}
	const uint32_t tpc = ctx_dev.threadsPerCta, wpc = ctx_dev.warpsPerCta;
	const uint64_t ctas = get_num_ctas(cbid, params);
	const uint64_t threads = ctas * tpc, warps = ctas * wpc;
	const uint32_t ng = ctx_dev.numGpr, nu = ctx_dev.numUr;

	std::vector<live_site_t> sites(host_sites.size());
	check(cudaMemcpy(sites.data(), dev_sites, sites.size() * sizeof(live_site_t), cudaMemcpyDeviceToHost), "read sites");
	close_slots(dev_read(ctx_dev.threadSlots, threads * (ng + LIVE_NUM_PRED)), sites);
	close_slots(dev_read(ctx_dev.warpSlots, warps * (nu + LIVE_NUM_PRED)), sites);
	const std::vector<live_time_t> tt = dev_read(ctx_dev.threadTime, threads);
	const std::vector<live_time_t> wt = dev_read(ctx_dev.warpTime, warps);

	uint64_t threadInstrs = 0, threadCycles = 0, warpInstrs = 0, warpCycles = 0;
	for (const live_time_t &t : tt) if (t.time) { threadInstrs += t.time; threadCycles += t.lastCycle - t.firstCycle; }
	std::map<uint32_t, std::pair<uint64_t, uint64_t>> smSpan;
	std::map<uint32_t, uint64_t> smWarpCycles;
	for (const live_time_t &t : wt) {
		if (!t.time) continue;
		warpInstrs += t.time;
		warpCycles += t.lastCycle - t.firstCycle;
		auto it = smSpan.find(t.sm);
		if (it == smSpan.end()) smSpan[t.sm] = {t.firstCycle, t.lastCycle};
		else { it->second.first = std::min(it->second.first, t.firstCycle); it->second.second = std::max(it->second.second, t.lastCycle); }
		smWarpCycles[t.sm] += t.lastCycle - t.firstCycle;
	}

	func_config_t config;
	nvbit_get_func_config(ctx, func, &config);
	const std::string kname = removeSpaces(nvbit_get_func_name(ctx, func));
	fprintf(out, "{\"kernel\":\"%s\",\"launch\":%d,\"ctas\":%lu,\"threads_per_cta\":%u,\"threads\":%lu,\"warps\":%lu,\"num_registers\":%d,\"used_gpr\":%u,\"used_ur\":%u,"
		"\"thread_instrs\":%lu,\"thread_cycles\":%lu,\"warp_instrs\":%lu,\"warp_cycles\":%lu,\"reads_no_write\":[%lu,%lu,%lu,%lu],\"lane_mismatch\":%lu,\"sm\":[",
		json_escape(kname).c_str(), launches[kname]++, ctas, tpc, threads, warps, config.num_registers, ng, nu, threadInstrs, threadCycles, warpInstrs, warpCycles,
		ctx_dev.readsNoWrite[0], ctx_dev.readsNoWrite[1], ctx_dev.readsNoWrite[2], ctx_dev.readsNoWrite[3], ctx_dev.laneMismatch);
	bool first = true;
	for (const auto &kv : smSpan) {
		fprintf(out, "%s{\"sm\":%u,\"span\":%lu,\"warp_cycles\":%lu}", first ? "" : ",", kv.first, kv.second.second - kv.second.first, smWarpCycles[kv.first]);
		first = false;
	}
	fprintf(out, "],\"sites\":[");
	first = true;
	for (size_t k = 0; k < sites.size(); k++) {
		const live_site_t &s = sites[k];
		if (s.count == 0) continue;
		const SiteInfo &m = host_sites[k];
		fprintf(out, "%s{\"pc\":%u,\"cls\":%d,\"reg\":%d,\"group\":\"%s\",\"count\":%lu,\"live\":%lu,\"live_cycles\":%lu,\"reads\":%lu,\"dead\":%lu,\"sass\":\"%s\"}",
			first ? "" : ",", m.pc, m.cls, m.reg, instGrouptNames[m.group], s.count, s.live, s.liveCycles, s.reads, s.dead, json_escape(m.sass).c_str());
		first = false;
	}
	fprintf(out, "]}\n");
	fflush(out);
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name, void *params, CUresult *pStatus) {
	if (!is_kernel_launch(cbid)) return;
	CUfunction func = get_launched_function(cbid, params);
	if (!is_exit) {
		pthread_mutex_lock(&mutex);
		before_launch(ctx, func, cbid, params);
		nvbit_enable_instrumented(ctx, func, true);
	} else {
		after_launch(ctx, func, cbid, params);
		pthread_mutex_unlock(&mutex);
	}
}

void nvbit_at_term() {
	if (out) fclose(out);
}
