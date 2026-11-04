# Memory-data injector

Corrupts one bit of data where it is stored, not the copy in one thread's
register. Only global memory is implemented so far; shared memory is still to do.

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

## Not exercised yet

The crash path (gemm buffers hold no pointers, so no flip crashed), the timeout path,
and frees of the target buffer (gemm never frees).

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
```
