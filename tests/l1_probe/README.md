# L1 copies and atomics executed in L2

Question: when one thread flips a word in global memory with `atomicXor`, which
threads still read the old value? `mem_injector` flips memory this way, and its
Masked analysis depends on the answer. NVIDIA does not document this behaviour,
so it is measured here.

## Method

`l1_probe.cu` runs 80 CTAs of 32 threads on all 20 SMs. All threads meet at a
barrier made of atomics, read the word (fill), then wait on their own clock
without touching memory. One thread flips the word. All threads then reread it
8 times and count old and new values per SM, 20 rounds per case. Results are
stored only after the rereads, so a reader executes no atomic or fence in
between unless the case asks for it.

The loads are written in PTX to control their kind. `ld.global` compiles to
`LDG.E`, `ld.volatile.global` to `LDG.E.STRONG.SYS` (checked on SASS). A first
version of the probe used `volatile` for all reads and never saw a stale value,
which is explained by the second row below.

## Results (RTX PRO 1000 Blackwell, driver 580.178.04, two identical runs)

409600 rereads per case. Raw data: `logs/l1-probe-2026-11-03/l1_probe.json`.

| Case | Before the rereads | Old value | New value | SMs with old | SMs with new |
|---|---|---:|---:|---:|---:|
| `control_no_flip` | fill, no flip | 409600 | 0 | 20 | 0 |
| `no_fill` | flip, no fill | 0 | 409600 | 0 | 20 |
| `fill` | fill, flip | **389120** | 20480 | 19 | 1 |
| `fill_volatile` | fill, flip, volatile loads | 0 | 409600 | 0 | 20 |
| `fill_fence` | fill, flip, `__threadfence()` by the reader | 0 | 409600 | 0 | 20 |
| `fill_atomic_by_reader` | fill, flip, atomic on another word by the reader | **389120** | 20480 | 19 | 1 |
| `fill_evict` | fill, flip, 4 MiB read by the reader | 0 | 409600 | 0 | 20 |
| `fill_next_kernel` | fill, flip, reread in the next kernel | 0 | 409600 | 0 | 20 |

In `fill`, all 20480 new values come from the SM of the thread that flipped.

## Conclusions

- `atomicXor` executed in L2 does not update the L1 copies of the other SMs.
  They keep returning the old value, here for the whole wait.
- The SM that issues the atomic sees the new value in all its threads.
- Stale copies end with eviction, with a fence by the reader, and at kernel
  boundaries. So a flip between kernels (timing mode a) leaves no stale copy.
- A volatile load does not see stale values. An atomic on another word does not
  clear them.

Consequences for tools: code that measures staleness must not use fences or
volatile loads in the reader threads, or it destroys what it measures.

## Limits

- One word, one working set and one wait length. Eviction in a real workload
  depends on its memory traffic.
- It does not tell whether the issuing SM invalidates or updates its copy.
- The fence case does not tell whether the fence invalidates the whole L1.
- The cycle counts in the JSON are not per-load latencies: the compiler can
  move the check that forces the load to complete, and the first read shows
  33 cycles and the second 860. They are only a rough hint (volatile rereads are
  about 5 times slower than plain ones), and no conclusion above depends on them.

## Run

```bash
make && ./l1_probe 20
```
