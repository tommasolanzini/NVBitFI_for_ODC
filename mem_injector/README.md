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
- **Reads after an overwrite are not meaningful:** the old/new comparison refers to
  the word before the flip. `first_access` marks these runs.
- **The trace read is an extra load.** It returns what the application sees, but it
  and the instrumentation change timing and L1 contents a little, so stale rates are
  not exactly those of the uninstrumented workload.
- Only cuMemAlloc-style allocations and instructions in the global or generic memory
  space are traced. Not covered: `cuMemCreate`, arrays, memory pools, surface and
  texture accesses, the CUDA runtime's own memory.
- Times come from a single timing run. Total run time varied between 2.2 and 6.3 ms in
  the pilot, so the weights between kernels and between modes are noisy.

## The first launch of a kernel

The first launch of a kernel takes far longer than the kernel, and this inflates every time weight
that uses the duration of a kernel. In a run with the tool in its timing phase, the first kernel of
ptr_chase took 1879 us for 57344 instructions and the second 51 us for 1658880. Without the tool,
shm_stencil launched 5 times took 64-73 us the first time and 7-9 us after that. In the timing phase
its only kernel took about 1650 us. The cause is presumably the loading of the module, not checked
further. The instruction clock inside a kernel is not affected.

So in mode b the first kernel of a run gets too many flips, and the windows of shared memory get too
much weight. To fix before a final campaign.

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
On the whole SRAM, as computed: known part 2.5%, from windows 6.1% of bits x time; reserved 4.1%, not
measured; L1 data 89.9%, not reachable, AVF to be estimated with the beam. **These shares are too high
for the windows:** see "The first launch of a kernel" below. With the kernel at its real length they
would be about 0.1% and 0.05% (an estimate). The 41% does not depend on it.

Pilot on ptr_chase, global memory, single bit (`logs/mem-pilot-2026-11-04-ptr/`, not a final campaign).
DUMMY=1, 100 per mode: all Masked, `old_word == new_word`. DUMMY=0, 100 per mode:

| Mode | SDC | Crash | Timeout | Masked | `no_read` | `overwritten` | `stale_only` | `read_masked` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a | 20 | 8 | 1 | 71 | 4 | 64 | 0 | 3 |
| b | 52 | 31 | 2 | 15 | 3 | 0 | 0 | 12 |

- All 72 SDCs change only the element of the list or index hit. All 39 crashes have the full record.
- Mode a: 63 runs flip `nodes` before its first copy, so the copy overwrites them.
- Mode b: 96 runs flip during `k_prepare`, which does not read `nodes`, and only 4 during `k_walk`, a
  result of the first launch below. So this pilot says little about stale L1 copies of pointers.
- High word of `next` in mode b: 22 crashes of 22. Bits 4-9: 3 SDC, 2 timeouts.
- All 15 `read_masked` flip bits 0-3 of a `value` or `weight`.
- After a timeout the next run takes the usual 1.04 s.

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
