# Shared memory probe

Measures what the shared memory modes of `mem_injector` rely on, on the RTX PRO
1000 Blackwell (sm_120, 20 SMs). Raw data: `logs/shm-probe-2026-11-04/` (git-ignored).

```bash
make -C tests/shm_probe
cd tests/shm_probe
./shm_probe atomic
./shm_probe occupancy
PROBE_MODE=addr LD_PRELOAD=$PWD/tool/probe_tool.so ./shm_probe known
PROBE_MODE=xor_atomic PROBE_OFFSET=1024 LD_PRELOAD=$PWD/tool/probe_tool.so ./shm_probe toolxor
```

Set `ACK_CTX_INIT_LIMITATION=1` and `NOBANNER=1` for the runs with the tool.

## Atomics on shared memory are atomic

`atomicXor` and `atomicAdd` on `__shared__` data compile to `ATOMS.XOR` and
`ATOMS.POPC.INC.32`.

| Test, 1023 threads x 1001 rounds | Result |
|---|---|
| Application `atomicXor`, 200 CTAs | 0 wrong |
| Application `atomicAdd`, 200 CTAs | 0 wrong |
| Application plain read-modify-write, 200 CTAs (control) | 200 wrong, 204263776 updates lost |
| Tool `atomicXor` through `__cvta_shared_to_generic`, 5 runs x 50 CTAs | 0 wrong of 250 |
| Tool plain read-modify-write, 5 runs x 50 CTAs (control) | 250 wrong of 250 |

So a flip with `atomicXor` cannot lose updates of other atomics or be lost by them.
A plain store from another thread is still ordered before or after it, as in any
program. That is the race the trace reports as `ambiguous` (see `mem_injector/README.md`).

## Addresses

- The address NVBit passes for LDS and STS (`nvbit_add_call_arg_mref_addr64`) is an
  offset inside the window of the CTA, and it is valid at `IPOINT_AFTER` too.
- **Application data start at offset 1024.** In `k_known`, `w[0]` is at 1024 and
  `w[64]` at 1280, in every CTA.
- `__cvta_shared_to_generic(offset)` from tool code reaches the window of the CTA of
  the calling thread. A flip at offset 1104 changed exactly word 20 of that CTA, and
  a flip at the address of a store changed exactly that word.
- Beyond the window: offset 4096 with a 272-byte window gave `Out Of Range Address`,
  Xid 13 and 43. A thread cannot reach other CTAs' data through its own window.

## Capacity of one SM

`cudaDeviceProp`: `sharedMemPerBlock` 49152, `sharedMemPerBlockOptin` 101376,
`sharedMemPerMultiprocessor` 102400, `maxBlocksPerMultiProcessor` 24. So 102400 bytes
(100 KiB) are available to shared memory, not the 128 KiB of L1 and shared storage in
the chip specs. The specs value is not measured by this probe.

Peak CTAs on one SM (the same on all 20 SMs), 3000 CTAs of 32 threads:

| Shared bytes per CTA | 0-2048 | 4096 | 8192 | 16384 | 32768 | 49152 | 65536 | 98304 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Peak | 24 | 20 | 11 | 5 | 3 | 2 | 1 | 1 |

`min(24, floor(102400 / (S + 1024)))` gives every value of the table. Together with
data starting at 1024 and `sharedMemPerBlockOptin = 102400 - 1024`, this says that
each CTA takes 1 KiB more than it declares. That is inferred: no document says it.

## The first KiB of the window

What it holds is unknown. Flips of bit 5 in all 256 words of `[0, 1024)`, in `k_known`,
after its stores: all 256 runs exit 0 with a correct output, and every word read 0
before the flip. This is one bit, one kernel and one time, so it does not prove the
area is unused in general. It is counted apart from free memory and not injected.

## Errors made while writing the probe

- The first `toolxor` read an address written during the kernel with a plain load, and
  got a stale L1 copy (`tests/l1_probe`). The offset now comes from the host.
- The compiler removed the shared word of the first `toolxor` kernel, which is why it
  crashed with offset 1024. The word is now accessed through a volatile pointer.
