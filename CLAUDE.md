# NVBitFI_for_ODC: context for Claude Code

## Who / what
- User: Tommaso Lanzini, MSc Space Systems Engineering (TU Delft). Thesis on radiation effects in GPUs for orbital data centers (ODC).
- Goal of the thesis: build a neural network that combines (a) NVBitFI software fault-injection results, where the injection location is known, and (b) proton-beam tests with RAS/ECC telemetry, covering the subsystems NVBitFI cannot reach. Use it to map the sensitivity of every GPU subsystem, i.e. where the GPU needs more protection.
- This repo is a fork of NVlabs/nvbitfi, ported to NVBit 1.8 / CUDA 12 / Blackwell (sm_120). See README.md (top section) for the full list of changes.

## How to talk to the user
- Answer in **Italian** (code, comments and commit messages in English).
- Every time you mention a hardware/software subsystem, add a very short explanation in parentheses of what it does, e.g. "predicate registers (1-bit per-thread flags that enable or disable individual instructions)". Keep it short but precise: never lump things together as "registri". Keep GPR, predicate registers, uniform registers, uniform predicates, architectural state, etc. distinct, and do the same for every piece of hardware.
- The user is learning the toolchain. Explain what a command does before or after running it, briefly.
- **Commits: Tommaso makes them himself.** Prepare the changes, show the diff, propose a commit message. Do not commit or push unless he explicitly asks.
- When asked to commit or push: Tommaso Lanzini must be the only author, with no Claude attribution (no `Co-Authored-By`, no "Generated with Claude Code"). Stage files by name.

## Machine / environment (verified 2026-11-03)
- HP ZBook X G1i 16, dual boot Windows 11 (BitLocker) + **Ubuntu 24.04**, kernel 7.0 HWE, Secure Boot ON.
- GPU: **NVIDIA RTX PRO 1000 Blackwell Laptop**, sm_120, 20 SMs, 8 GB GDDR, **no ECC**, minimal RAS (fine for FI development; beam/RAS work will need a data-center GPU).
- Driver **580.178.04** (`nvidia-driver-580-open`, apt-mark hold). The NVBit 1.8 README says "driver <= 575", but 580 works: NVBit samples instr_count and mov_replace run fine.
- CUDA toolkit **12.9** in `/usr/local/cuda-12.9` (NVIDIA apt repo, `cuda-toolkit-12-9` only, never install `cuda`/`cuda-drivers`).
- NVBit 1.8: `~/thesis/nvbit_release_x86_64`. This repo: `~/thesis/nvbit_release_x86_64/tools/nvbitfi`, branch `odc-nvbit18`.
- Env for campaigns: `export NVBITFI_HOME=~/thesis/nvbit_release_x86_64/tools/nvbitfi`. `dmesg` needs `sudo sysctl kernel.dmesg_restrict=0` (resets at reboot) to record Xid errors.

## Hard constraints learned on this GPU
- **No `printf` in device code of any NVBit tool** (inject_funcs.cu). With NVBit 1.8 on sm_120 the tool module then fails to load: `nvbit_imp.cpp line 716: CUDA_ERROR_INVALID_SOURCE device kernel image is invalid` (the NVBit sample mem_printf fails the same way). Record everything in device structs and print from the host.
- NVBit register device API (`nvbit_read/write_reg`, `_ureg`, `_pred_reg`, `_upred_reg`) loads and works (mov_replace sample OK).
- NVBit prints a `nvbit_at_ctx_init()` WARNING on stderr. Runs must set `ACK_CTX_INIT_LIMITATION=1` and `NOBANNER=1`, otherwise every run differs from the golden output. `scripts/common_functions.py:set_env` already does this.
- `make` rebuilds by itself when `ARCH`, `DUMMY` or the other build flags change (stamp file `.build_flags`, see `common/build.mk`), so `make clean` is not needed. Always pass `ARCH=sm_120`: a plain `make` means `ARCH=all` and rebuilds for every architecture.

## Code map
- `common/arch.h`: X-macro opcode table (opcode -> exclusive group), groups (fp64, fp32, ld, pr, nodest, others, tc, sfu, fp16, uniform, unknown, gppr=11, gp=12), RegType (GPR/PR/UR/UP), FLAG_* eligibility bits, bit-flip models.
- `common/dests.h`: destination analysis from decoded operands (GPR/PR/UR/UP, widths, MMA shapes, discard regs by name). Offline test: `make -C tests check` (73 cases).
- `common/globals.h`: launch helpers (incl. cuLaunchKernelEx), NVBit operand front-end, `analyze_instr`, eligibility flags, counters layout.
- `profiler/`: counts executed instances per opcode and **injectable** instances per group (same rules as the injector).
- `injector/`: transient FI. `inject_select` (IPOINT_BEFORE) counts and selects the target, `inject_error` (IPOINT_AFTER) corrupts one destination. The log has opcode, PC, regType/regNo, mask, before/after, tid, CTA, smID, warpID, laneID, groupInstrs. The record lives in host-mapped pinned memory (`inj_shadow_t`, `cudaHostAlloc`), written by the device at selection and before the register is corrupted, so crashes and hangs keep it. `groupInstrs` stays in managed memory: it is -1 in crashed or timed-out runs.
- `pf_injector/`: permanent FI (SM, lane, opcode, XOR mask). Not wired into the scripts.
- `scripts/`: params.py (apps, groups x models, NUM_INJECTIONS/THRESHOLD_JOBS, outcome categories), run_profiler.py, generate_injection_list.py, run_injections.py, run_one_injection.py (classification + `results-*.jsonl`, one JSON record per injection for the ML pipeline, including `exit_code`, `cuda_error`, `xids`), parse_results.py. `app_igids` in params.py selects the groups per app. run_injections.py skips the jobs already in the JSONL, so an interrupted campaign can be resumed. `generate_injection_list.py <app>` regenerates the list of one app only.
- `test-apps/simple_add/`: tiny test kernel (run.sh, sdc_check.sh, golden).
- `test-apps/sfu_math/`: rsqrt, rcp, sin, exp, log through the fast intrinsics (MUFU), one kernel per function, N=128, one thread per element. Bit-exact golden with all outputs. Separate from gemm so that its golden and the pilot stay valid.
- `test-apps/gemm/`: C = A x B, N=128. `gemm_core` = fp32 on CUDA cores, one thread per element (CTA = row, thread = column, so `tid` = flat index of the element). `gemm_tc` = Tensor Cores via WMMA (HMMA.16816.F32), one warp per 16x16 tile. Bit-exact golden with both full matrices (`make golden`, git-ignored: run it after a clone). `signature.py` is called by run_one_injection.py if the app has one (shared code in `scripts/fp_signature.py`); its result goes in the JSONL field `signature`, with the bit class computed on the final output element, not on the injected bit (that is `mask`).

## Validation status
- `make -C tests check`: 73/73 + 18/18 (signature).
- simple_add, G_GP, DUMMY=1: 25/25 Masked; `groupInstrs` (injector) == `gp` in the profile == 51200 (10 CTAs x 1024 threads x 5 GPR-writing instructions).
- simple_add, G_GP, DUMMY=0, 25 injections: 20 Masked, 5 `DUE: Kernel crash` with full record (all LDC, GPR R1, PC 0x0, value 0xfffcf0, probably the stack pointer; not verified on SASS), `xids` [13, 13, 13, 43], dmesg Xid 13 "SM Warp Exception ... Out Of Range Address" + Xid 43.
- Outcome rules in `classify_injection`: CUDA kernel error = 20 `DUE: Kernel crash` (does not depend on dmesg), other non-zero exit = 5 `DUE: Non Zero Exit Status`, "Error not injected" = 19 `Uncategorized`. Categories 6 and 7 are no longer assigned. Non-zero exit means DUE, so workloads must exit 0 on success.
- gemm, 25 injections per group, single bit. DUMMY=1: all Masked, `groupInstrs` == profile (fp32 2097152; tc 32768; gp 11173888 core, 364544 tc). DUMMY=0: fp32 23 SDC / 2 Masked; tc 25 SDC; gp 19 SDC / 4 crash / 2 Masked.
- gemm location check on those 67 SDCs: 40/40 `gemm_core` wrong element == `tid`, 27/27 `gemm_tc` errors inside the tile of the logged CTA. A corrupted load in `gemm_tc` gives a 16-element row (A) or column (B); a corrupted HMMA result gives one element.
- gemm pilot (2026-11-03, single bit, raw results in `logs/pilot-2026-11-03-gemm/`, not the final campaign). DUMMY=1, 25 per group: all Masked, counts == profile. DUMMY=0, 100 per group (SDC/crash/Masked): fp32 95/0/5, ld 88/1/11, pr 88/7/5, tc 95/0/5, gp 68/23/9. uniform, 1000: 127/199/674 (`gemm_core` 9/5/606, `gemm_tc` 118/194/68). gemm has no sfu instructions: see sfu_math.
- **Propagation verified on GPU.** PR: 88/88 SDCs change only the element of the logged thread. UR: 9/9 `gemm_core` SDCs (S2UR UR8, written by lane 0) change exactly the 32 elements of the whole warp. UP: 9/9 `gemm_tc` SDCs change the whole 256-element tile.
- gemm_core uniform Masked rate is a workload artifact: UR4 (561 of 620 hits) is read only by the remainder loop, which N=128 never runs.
- **Memory descriptor** (`LDCU.64 URn, c[0x0][0x358]`, the `desc[URn]` of LDG/STG; full report in `tests/desc_probe/README.md`). Value 0. Sweep of all 64 bits, not documented by NVIDIA: output never changes for single-bit flips that do not crash. Bit 61 (high word bit 29): 64 KiB working set 3x slower, same time as 1 MiB, probably L1 bypass (inferred from timing, no counters). Bit 59 (high word bit 27) or all 64 bits: always `illegal instruction`, Xid 13 "Illegal Instruction Parameter". The other 62 bits: no effect on this pattern, not proven harmless. So Masked on these registers means "output unchanged", not "no effect".
- Not exercised yet: the SIGKILL fallback in `is_timeout`, a crash in a kernel after the injected one.
- Results in `logs/simple_add/` from before commit 1550c4b have the old labels and format: rerun the campaign for homogeneous data.
- simple_add note: atomicAdd is warp-aggregated and the leader is the **highest** active lane (`UFLO` = find leading one), not lane 0. Corruptions in the other lanes are mostly masked by construction.

## TODO (in this order)
1. Done: crash-safe injection record.
2. Done: classification of crashes (DUE, exit code, CUDA error, Xids).
3. Done: Makefile rebuilds on flag and header changes.
4. Done: GEMM workload and output signatures.
5. **In progress: final campaigns**, 1000 injections per group, single bit: gemm with gp, pr (predicates), tc (tensor cores), uniform (UR/UP, reusing the pilot's 1000), fp32, ld; sfu_math with sfu. Results and configuration go in `logs/final-v1-<date>-gemm/`. Propagation of UR/UP/PR corruption is verified (see Validation status).
6. **Memory-data injector (`mem_injector/`)**: corrupt data where it is stored, not only the copy in one thread's register.
   - **Goal of the thesis:** show that the FI toolchain (NVBitFI + mem_injector) covers the large majority (target ~90%) of the *expected* upset rate of this GPU, and that the remaining part comes from structures software cannot reach (control logic, pipeline latches, scheduler, physical L2 location). Those structures may be many but have few bits / low reachability and are covered by the beam tests (RAS, Xid).
   - **Coverage metric (to compute later, not now):** for each structure s, expected upset rate ∝ σ_bit(s) × bits(s) × AVF(s, workload). Coverage = sum over the structures reachable by FI / sum over all structures. σ_bit from literature or beam data, bits from the chip specs (GB207: 32 MB L2, 20 SM × 256 KB register file, 20 SM × 128 KB L1/shared, 8 GB GDDR7, no ECC), AVF from the FI campaigns and from the lifetime of the allocations. Keep the inputs in one table/config so the number can be recomputed.
   - **Global memory (VRAM + L2):** track live allocations through NVBit's CUDA callbacks (cuMemAlloc*/cuMemFree*, sizes, alloc/free times). Choose a target (buffer, offset, bit, time) weighted by bits × lifetime. Two timing modes: (a) between kernels, host read-XOR-write of the word; (b) during a kernel, at the N-th dynamic instruction one thread executes `atomicXor` on the address (atomics are executed in L2, so the flip is coherent and visible to all SMs and later kernels). Bytes never allocated are masked by definition: count them analytically, do not spend injections on them.
   - **Shared memory:** after an STS, XOR the stored word at its shared address (NVBit mref address / nvbit_get_shmem_base_addr), so every thread of the CTA that reads it sees the corrupted value.
   - **Known limit to document:** software sees addresses, not cells. It cannot pick an L2 slice/set/way and cannot reproduce an upset on a *clean* L2 line that disappears on eviction. The class is "corrupted data in global memory": it slightly overestimates L2 upsets. L2 vs VRAM separation is left to the beam (ECC on/off on a data-center GPU). This GPU has no ECC, so software flips are realistic here.
   - **Record in the JSONL:** structure (global/shared), buffer id and size, offset, bit, old/new word, timing mode, kernel index and dynamic instruction of the flip, list of kernels that read the corrupted address afterwards (optional, via memory tracing), outcome and output signature as for the other campaigns.
   - Same conventions as the injector: no device-side printf, crash-safe record in host-mapped memory, DUMMY mode, offline tests where possible.

## Later
- gemm: per-layer checkpoints/checksums (only features also observable under beam).

## Campaign quick reference
```bash
export NVBITFI_HOME=~/thesis/nvbit_release_x86_64/tools/nvbitfi
sudo sysctl kernel.dmesg_restrict=0
make -C $NVBITFI_HOME/injector ARCH=sm_120 DUMMY=0
make -C $NVBITFI_HOME/profiler ARCH=sm_120
cd $NVBITFI_HOME/test-apps/gemm && make && make golden
cd $NVBITFI_HOME/test-apps/sfu_math && make && make golden
cd $NVBITFI_HOME/scripts
python3 run_profiler.py && python3 generate_injection_list.py
python3 run_injections.py standalone && python3 parse_results.py
# results: logs/<app>/results-*.txt and results-*.jsonl, per-run dirs logs/<app>/<app>-group<G>-model<M>-icount<N>/
```
