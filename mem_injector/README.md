# Memory-data injector

Corrupts one bit of data where it is stored, not the copy in one thread's
register. Global memory and shared memory are implemented; see "Shared memory" below
for the second one.

## What it does

- **Tracks allocations** through NVBit's CUDA callbacks (`cuMemAlloc_v2`,
  `cuMemAllocManaged`, `cuMemFree_v2`). Buffer ids follow the allocation order,
  because addresses can change between runs. Copies and memsets are also events.
- **Chooses a target** (buffer, word, bit, time) weighted by bits x lifetime,
  separately for each timing mode (`scripts/mem_plan.py`).
- **Timing mode a, between kernels:** on entry of a chosen CUDA event, the host
  reads the word, XORs it and writes it back.
- **Timing mode b, during a kernel:** at the N-th dynamic instruction, the thread
  that reaches it executes `atomicXor` on the word.
- **Traces the flipped byte.** Every global load and store is checked against it,
  and host copies are checked in the callbacks. A load overlapping the byte is
  compared with the old and new bit.

- **Multi-cell upsets**, modes a, b, s and u, chosen at list time with `--mcu`:
  `adj_bits --k K`, k adjacent bits of one word; `adj_words --k K --step S`, the same bit
  in k words S bytes apart, in a direction drawn per target. S = 4 or 8 for neighbours, any multiple of 4
  for a row distance. No value is fixed in the code, because the interleaving is not known.
  A word outside the buffer or the window of the CTA is not flipped: `cells_requested` against `cells`.
  Each word has its own trace in `cells`; the Masked subclass is the strongest among
  them, in the order `read_masked`, `ambiguous`, `stale_only`, `overwritten`, `no_read`. Default
  sizes for weighting afterwards: `scripts/mcu_sizes.py`.

`MEMINJ_PHASE` selects the behaviour: `timing` (no instrumentation, event times),
`count` (dynamic instructions per kernel), or `inject` (default, one flip).
Do not prefix the workload with `timeout` in `LD_PRELOAD` runs: the tool also
loads into `timeout`, and that process truncates the events file.

## L1 copies

An `atomicXor` executed in L2 does not update the L1 copies of the other SMs, so
they keep returning the old value until the line is evicted, a fence runs in the
reader, or the kernel ends. Only the SM that issued the atomic sees the new value.
Measurements are in `tests/l1_probe/README.md`.

A real upset in L2 or VRAM would not update those copies either, so such runs are
valid Masked results, but they do not show that the workload tolerates the error.
Hence the subclasses (`masked_subclass` in the JSONL):

| Subclass | Meaning | Workload tolerates the error? |
|---|---|---|
| `no_read` | nothing accessed the byte after the flip | no evidence |
| `overwritten` | written before any read | no evidence |
| `stale_only` | device loads returned only the old bit | no evidence |
| `read_masked` | flipped bit read, output unchanged | yes |

Mode a is the control: no stale copy can survive the kernel boundary.

**Reading `stale_only` with care.** The old/new trace comes from a load of the tool, separate from the load
of the application, so the two can disagree. In run 42 of the second mode b pilot on ptr_chase the only
reader of the word traced the old value and the run crashed, which points to the application having read
the new one (inferred from one run). The fraction of stale reads also depends on the access pattern: 2 of
31 traced reads in ptr_chase, where each word is read once, against 95% in `tests/l1_probe`, where the same
word is reread. The cause of the difference is not verified.

## Shared memory

Shared memory is the working memory of a CTA, addressed as a window of that CTA and
stored in the same RAM of the SM as L1 data. It exists only while the CTA is resident,
so there is no mode between kernels and a flip cannot leave its CTA or kernel. It does
not go through L1: no stale copies are expected, and the trace measures it. Measurements
behind this section are in `tests/shm_probe/README.md`.

| Mode | Flip | Use |
|---|---|---|
| `s` | Right after the N-th dynamic store to shared memory, one of the stored bits | Dataset for the network. It always hits live data, so it is not a model of upsets |
| `u` | SM x offset x time, with the CTA picked at random among those resident on the SM | The only one for AVF and coverage |

Both flip with `atomicXor` (`ATOMS`), which is atomic against the other threads of the CTA.

**Sampling of mode u.** An upset hits the SRAM of an SM, not a CTA, so each resident bit at
each time has the same weight. The count phase logs the SM, first and last instruction of each
CTA (`ctas` in the kernel event). A target keeps an SM, a time and a word of the window
(`sample_uniform`). At run time the first warp on that SM after that time picks a CTA at random
among those resident there, because a thread can only reach the window of its own CTA, and the
next warp of that CTA flips. Scheduling can place the CTAs differently from the profile, so
the choice cannot be made in advance.

The residents are kept in a mask per SM, updated when a CTA enters (first instruction of the
kernel) and leaves (all its threads at EXIT). So the choice takes 2 atomic reads. Two earlier
versions showed why this matters, both on the same pilot:

| Version | Median delay of the flip | Largest | Not injected |
|---|---:|---:|---:|
| Scan of a table with one entry per CTA | 154108 instructions | 582518 | 10 of 100 |
| Mask, choice published after the writes to host memory | 19130 | 345956 | 5 of 100 |
| Mask, choice published first (current) | 6471 (0.33% of the kernel) | 23268 (1.2%) | 0 of 100 |

**The SRAM as one structure** (`exposure-shared.json`, bit-seconds over all SMs and the whole run):

| Term | What | Treated as |
|---|---|---:|
| `windows` | application part of the windows of resident CTAs | reachable, injected |
| `reserved` | `cudaDevAttrReservedSharedMemoryPerBlock` (1024 bytes) per resident CTA | kept apart, not injected |
| `l1_data` | all the rest: `never_shared` (128 KiB - 102400 bytes) + `free_shared` | L1 data: not reachable, not Masked |

**AVF in two versions** (`run_mem_campaign.py avf`, from mode u only): on the injected part only,
and on the whole SRAM. The second is not one number: it is a known part (AVF x share of the windows)
plus the share of `reserved` and the share of `l1_data`, marked "not reachable, AVF to be estimated
with the beam".

**One more subclass:** `ambiguous` (`first_access` 3). A store was checked before the flip
and finished after it, so it cannot be told which came first. Reads after a store to the
byte are not counted in shared memory. This was added after one DUMMY=1 run of 25 showed 3
new reads with no flip, on a word that was then written twice: the data of a store read
back as the flipped bit is the likely cause, not a proven one. With the check, 0 of 200
DUMMY=1 runs show new reads. `ambiguous` occurs in about 1-2 runs of 100 in mode u.

**Limits of shared memory:**

- **The split of `l1_data` assumes** that the 28 KiB and the free shared space hold L1 data. How the
  driver splits the SRAM at run time is not documented, and 128 KiB comes from the chip specs.
- **The clock is the dynamic instruction count with the tool running.** Time inside a kernel
  and residence are proportional to it, which is an approximation. The windows of a run
  come from one timing run, as for global memory.
- **The flip comes slightly after the sampled time:** the next warp of the chosen CTA has to run.
  A CTA can only read its window through its own threads, so little can happen in between, except
  in its other warps.
- **The trace can miss a load** that passes the check just before the flip. Pilot: 1 SDC of 40
  in mode u with no new read and a store as first access. So a few `overwritten` runs could
  really be `read_masked`.
- **Not traced:** shared memory reached through a generic pointer, `GLOBAL_TO_SHARED` and
  `DISTRIBUTED_SHARED`. The record counts them in `unseen_shared_instrs`: runs with a non-zero
  value have an incomplete trace. shm_stencil has none.
- **Not exercised:** dynamic shared memory (the window size is taken as static + dynamic from
  `nvbit_get_func_config`, checked only with static), atomics of the application on shared
  memory, a crash from a flip in shared memory, a run not injected because the SM was idle in
  a pilot (only in a hand-built flip), and a CTA leaving before the choice: in this workload
  all CTAs start together, so every choice saw 6 or 7 residents with no thread gone.

## Conventions and traps

- No device-side printf. The record is in host-mapped memory, updated with
  atomics, and a plain-memory snapshot is taken after every event.
- **No fences or volatile loads in reader threads:** they would destroy the stale
  copies being measured. The trace uses plain loads and atomics, which the probe
  shows do not clear copies. Fences run only in the flipping thread, whose SM sees
  the new value anyway.
- **Flags changed during a kernel are read with atomics,** because a plain load can
  be stale too.
- `DUMMY=1` applies a zero mask through the same path. All runs must be Masked with
  `old_word == new_word`. `masked_subclass` means nothing in that mode: with a zero
  mask no load can show a flipped bit.

## Known limits

- **Addresses, not cells.** Software cannot pick an L2 slice, set or way, and cannot
  reproduce an upset on a clean L2 line that disappears on eviction. The class is
  "corrupted data in global memory" and slightly overestimates L2 upsets. Separating
  L2 from VRAM is left to the beam tests.
- **L1 data cells are not reached** by either mode, so they do not count as reachable
  in the coverage metric.
- **The trace can miss a load by a small margin in mode b.** A load that passes the
  check just before the flip may still return the new value. Pilot: 1 run of 16 with
  1 more wrong element than new reads. The classification of a run changes only if
  that load was the only one to see the flip.
- **Reads after a store do not count**, in global as in shared memory: the old/new
  comparison refers to the word before the flip. Before this rule, 117 false new reads appeared
  on cnn with DUMMY=1, all in `overwritten` runs. A store by another thread at the same
  moment can still slip past it. `first_access` marks these runs.
- **The trace read is an extra load.** It returns what the application sees, but it
  and the instrumentation change timing and L1 contents a little, so stale rates are
  not exactly those of the uninstrumented workload.
- Only cuMemAlloc-style allocations and instructions in the global or generic memory
  space are traced. Not covered: `cuMemCreate`, arrays, memory pools, surface and
  texture accesses, the CUDA runtime's own memory.
- Times are the mean of 10 timing runs: one alone varied between 2.2 and 6.3 ms in
  the pilot. The durations of the kernels come from `tests/kernel_timing`.
- Mode b is sampled with the upper bound of the time weights. `run_mem_campaign.py rates --floor-us F` gives
  the rates under both bounds, weighting each kernel. A kernel with no runs has no rate: its share is reported, not guessed.
- `config/hashes.jsonl` has the hashes of tool, workload and golden files, per launch.

## The first launch of a kernel

The first launch of a kernel takes far longer than the kernel, which inflated every time weight that
used the duration of a kernel. In the timing phase of the tool, `k_prepare` took 1879 us for 57344
instructions and `k_walk` 45 us for 1658880.

`CUDA_MODULE_LOADING=EAGER` does not fix it. Host timer, mean of 10 runs, launch 0 / mean of launches 2-5 (0-based), in us:

| Kernel | No tool, default | No tool, EAGER | Tool, default |
|---|---:|---:|---:|
| `k_prepare` | 496 / 6.5 | 837 / 6.2 | 2821 / 8.6 |
| `k_walk` | 27.5 / 20 | 26.7 / 19 | 48 / 22 |
| `shm_stencil` | 80 / 6.9 | 23 / 7.0 | 2079 / 10.6 |

nsys could not measure on the GPU side. Nsight Systems 2025.1.3 of CUDA 12.9 runs the applications but
collects no CUDA activity on this driver: it warns that driver CUDA version 13.0 is not supported and
CUPTI (NVIDIA's profiling interface) returns `CUPTI_ERROR_INVALID_DEVICE` and
`CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`. The cause was not established.

So `run_mem_campaign.py profile` measures the kernels on `tests/kernel_timing/repeat_<app>`, a generated
copy of the workload that repeats each launch. 10 clean runs, no tool, EAGER; the mean of launches 2-5
(0-based) becomes the duration of the kernel in `kernel-timing.json` and in the `duration` field of
`mem-profile.json`. The workloads still launch each kernel once. `t0`, `t1` and the gaps stay as
measured in a single timing run of the tool.

Limits of these durations:

- **They are host times with a floor.** An empty kernel takes 5.73 us (5.0-7.2, launches 2-7 of 5 runs, no
  tool, EAGER; `tests/kernel_timing/empty_kernel`), close to `k_prepare` (5.8 us) and `shm_stencil` (6.5
  us). So the weights keep the floor for now and the two cases below bound the result, until there are
  measurements on the GPU side.
- **Warm caches.** `k_walk` takes 27 us on its first launch without the tool and 19 us later. How much of
  this is cold caches is not known.
- `repeat_gemm` covers gemm; sfu_math launches through a macro and has no copy yet.

Bounds of the time weights (`scripts/mem_time_bounds.py`, no GPU):

| Quantity | Upper bound: as measured (used for the weights) | Lower bound: without the floor |
|---|---:|---:|
| `k_prepare` share of mode b weight, ptr_chase | 23.6% | 0.48% |
| Windows share of the SRAM, shm_stencil | 0.113% | 0.014% |
| Known part of the AVF on the whole SRAM (AVF 41% x windows) | 0.046% | 0.0057% |
| Reserved KiB share, not measured | 0.075% | 0.0092% |
| L1 data share, not reachable | 99.81% | 99.98% |

Neither is a measurement of the kernel. The lower bound subtracts two host times of similar size: for
`k_prepare` it leaves 0.06 us, within the noise of both, so its share is compatible with zero. The floor
comes from a 64 x 64 grid, while shm_stencil has 128 x 128, and from 5 runs. The AVF of 41% on the injected
part does not depend on any of this.

## Not exercised yet

The SIGKILL fallback of the timeout (the SIGINT handler has always been enough), and whether a
reused address is really left alone after a free: the trace stops at the free, but ptr_chase does not
check that `out` gets the address of `scale`.

## Validation

Offline: `python3 tests/test_mem_plan.py`.

Hand-built flips on gemm (`run_mem_campaign.py one`), all as predicted:

| Flip | Result |
|---|---|
| A[0] before `gemm_core` | 128 new reads on 1 SM, whole row wrong |
| B[5] before `gemm_core` | 128 new reads on 20 SMs, whole column wrong |
| C before `gemm_core` | `overwritten`, Masked |
| C before the copy out | host read, 1 wrong element |
| A after `gemm_core` | `no_read`, Masked |
| A_h[0] before `gemm_tc` | 8 tile reads, 128 wrong elements in a row |
| A[0] in `gemm_core`, early | 128 new reads, whole row wrong |
| A[0] in `gemm_core`, late | `no_read`, Masked |

Pilot on gemm, single bit (`logs/mem-pilot-2026-11-03-gemm/`, not a final campaign).
DUMMY=1, 25 per mode: all Masked, `old_word == new_word`. DUMMY=0, 100 per mode:

| Mode | SDC | Crash | Masked | `no_read` | `overwritten` | `stale_only` | `read_masked` |
|---|---:|---:|---:|---:|---:|---:|---:|
| a | 31 | 0 | 69 | 22 | 46 | 0 | 1 |
| b | 38 | 0 | 62 | 23 | 36 | 0 | 3 |

Only 4 of 131 Masked runs show tolerance. Stale reads occurred in 1 run of mode b
(4 old reads, 31 new, Masked). The sample is too small to compare the two modes.

Pilot on shm_stencil, single bit (not a final campaign). Mode s in `logs/mem-pilot-2026-11-04-shm/`,
mode u in `logs/mem-pilot-2026-11-04-shm-v2/`. DUMMY=1, 100 per mode: all Masked, `old_word == new_word`,
no new read. DUMMY=0, 100 per mode:

| Mode | SDC | Crash | Masked | `no_read` | `overwritten` | `stale_only` | `read_masked` | `ambiguous` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s | 97 | 0 | 3 | 0 | 0 | 0 | 3 | 0 |
| u | 41 | 0 | 59 | 5 | 49 | 0 | 3 | 2 |

No read returned the old bit in the 200 runs. All 138 SDCs change only the row of the logged
CTA, within 8 columns of the flipped word (8 stencil iterations). In mode u, the flipped CTA is
the chosen one and the SM is the target in 100 of 100 runs, and at the end of the kernel all CTAs
have left and no slot is in use.

AVF of shared memory in shm_stencil, from mode u: 41% on the injected part (Wilson 95%: 31.9-50.8).
With the corrected durations, on the whole SRAM: known part 0.046%, from windows 0.113% of bits x time;
reserved 0.075%, not measured; L1 data 99.81%, not reachable, AVF to be estimated with the beam. The first
pilot computed 6.1% for the windows, inflated by the first launch. These are the upper bounds: see the
table in "The first launch of a kernel". The 41% does not depend on it.

Pilot on ptr_chase, global memory, single bit (`logs/mem-pilot-2026-11-04-ptr/`, not a final campaign).
DUMMY=1, 100 per mode: all Masked, `old_word == new_word`. DUMMY=0, 100 per mode:

| Mode | SDC | Crash | Timeout | Masked | `no_read` | `overwritten` | `stale_only` | `read_masked` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a | 20 | 8 | 1 | 71 | 4 | 64 | 0 | 3 |
| b | 52 | 31 | 2 | 15 | 3 | 0 | 0 | 12 |

- All 72 SDCs change only the element of the list or index hit. All 39 crashes have the full record.
- Mode a: 63 runs flip `nodes` before its first copy, so the copy overwrites them.
- Mode b: 96 runs flip during `k_prepare`, which does not read `nodes`, and only 4 during `k_walk`, a
  result of the first launch below. So this pilot says little about stale L1 copies of pointers: see
  the second pilot of mode b.
- High word of `next` in mode b: 22 crashes of 22. Bits 4-9: 3 SDC, 2 timeouts.
- All 15 `read_masked` flip bits 0-3 of a `value` or `weight`.
- After a timeout the next run takes the usual 1.04 s.

Second pilot of mode b on ptr_chase, corrected durations (`logs/mem-pilot-2026-11-04-ptr-b-v2/`). DUMMY=1,
100 runs: all Masked, no new read. DUMMY=0, 100 runs:

| Flip during | Runs | SDC | Crash | Timeout | `no_read` | `overwritten` | `stale_only` | `read_masked` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `k_prepare` | 18 | 12 | 4 | 0 | 0 | 0 | 0 | 2 |
| `k_walk` | 82 | 15 | 14 | 0 | 50 | 1 | 1 | 1 |
| Total | 100 | 27 | 18 | 0 | 50 | 1 | 1 | 3 |

- `no_read` is high because a list is walked once: a flip behind the walking thread is never read.
- Stale L1 copies of pointers are rare here: of the 31 `k_walk` flips of `nodes` with a traced read, 2
  returned the old bit and 29 the new one, 28 of these from an SM other than the one that flipped. In
  `tests/l1_probe` the same word was reread, here each word is read once. Why so few was not investigated.
- Run 54 (`next` address bit 47): one old read, Masked, `stale_only`. A new value in that bit would be
  expected to crash, so the load probably did see the old one.
- Run 42 (`next` address bit 43): one old read, no new read, crash. Only one thread reads that word, so the
  load of the application probably saw the new value while the load of the tool saw the old one. So the
  old/new trace can disagree with the application; this is an inference from one run.
- All 27 SDCs change only the list hit. All 18 crashes have the full record. 0 runs not injected.

Hand-built flips on ptr_chase, as predicted except that bit 3 of `next` gave `illegal memory access`
(Xid 13 and 43), not `misaligned address`:

| Flip | Result |
|---|---|
| `value` of a node | 1 new read, only that list wrong |
| `next` high word; null `next` bit 4; `heads` high word | crash, Xid 31, full record |
| `next` bits 4-9 to an earlier node, mode a or b | timeout at 10 s, exit 255 from the SIGINT handler, full record (over 400000 new reads) |
| `next` bits 4-9 to a later node | nodes skipped, only that list wrong |
| `next` high word during `k_prepare` (mode b) | crash in the next kernel, record shows both |
| `scale` before `k_prepare` | only that element wrong |
| `scale` after `k_prepare`, on entry of its free | `no_read`, Masked |
| `scale` after its free | not injected |
| `out` before `k_walk`; `adj` on entry of its free | `overwritten`; `no_read` |

Hand-built flips on shm_stencil, as predicted except the last line:

| Flip | Result |
|---|---|
| `coef[60]`, mid kernel | 5 new reads, row of the CTA wrong in columns 56-64 |
| `coef` or `buf` word before the first store | `overwritten`, Masked |
| `buf` word, mid kernel, in its dead half | `overwritten`, Masked |
| last store of the run (mode s) | 1 new read, only the element of `tid` wrong |
| first store of the run (mode s) | 3 new reads, row wrong in columns 0-8 |
| store index beyond the last | not injected |
| `coef[60]` at instruction 1933000 of SM 0 (predicted `no_read`) | not injected: SM 0 had no CTA left |

Multi-cell pilot, single seed, not final (25 runs per mode and configuration). DUMMY=1, 200 runs: all Masked, no new
read, `old_word == new_word`, words flipped equal to requested. DUMMY=0, SDC/crash/Masked, both modes together:

| Workload | Adjacent bits k=3 | Words +4 | Words +8 | Words +512 |
|---|---:|---:|---:|---:|
| gemm (a, b) | 16/0/34 | 13/0/37 | 12/0/38 | 18/0/32 |
| ptr_chase (a, b) | 16/13/21 | 10/16/24 | 6/15/29 | 18/4/27, 1 timeout |
| shm_stencil (s, u) | 37/0/13 | 35/0/15 | | |

No run was left without injection. 181/181 SDCs are at the elements expected from the cells in the record
(`scripts/check_fault_locations.py`).

## Run

```bash
export NVBITFI_HOME=~/thesis/nvbit_release_x86_64/tools/nvbitfi
make -C $NVBITFI_HOME/mem_injector ARCH=sm_120 DUMMY=0
cd $NVBITFI_HOME/scripts
D=$NVBITFI_HOME/logs/mem-<date>-gemm
python3 run_mem_campaign.py profile gemm $D
python3 run_mem_campaign.py list gemm $D --mode b -n 100 --seed 1
python3 run_mem_campaign.py run gemm $D --mode b
# results: $D/results-modeb.jsonl, per-run dirs in $D/runs-modeb/
# shared memory: the same with shm_stencil and --mode s or --mode u
```
