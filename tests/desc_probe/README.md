# Memory descriptor experiment

Question: on `gemm_core` every injection into `LDCU.64 UR6/UR7, c[0x0][0x358]`
was Masked. These registers are the `desc[UR6]` of every `LDG.E`/`STG.E`. Is the
descriptor a cache policy, so that a flip changes the timing and not the output?

NVIDIA does not document its content. The CUDA Binary Utilities only name the
operand ("desc[URX][RY] for memory descriptors"), and an NVIDIA forum thread has
no field description.

## Method

- `desc_probe`: one warp (32 threads), one `LDCU.64` feeding all the accesses,
  1M loads and 1M stores per thread with a 64-byte stride. Times the second
  launch only, because the first one includes the NVBit instrumentation.
- `desc_tool.so`: a minimal NVBit tool that instruments only that `LDCU.64`, so
  no overhead is added to the timed loop, and XORs a mask into the destination.
- `sweep.py 5 out.json`: each of the 64 bits, plus all the bits at once, on 3
  working sets (64 KiB, 1 MiB, 256 MiB), 5 repetitions each, 40 per working set
  with no flip, random order. The checksum covers the loaded values and the whole
  buffer, so it also covers the stores.
- Positive control: the same mechanism on the access count (bit 21 of
  `c[0x0][0x38c]`) gives 158.5 ms instead of 51.3 ms and a different checksum.

Build with `make`. Results of 2026-11-03: `logs/desc-probe-2026-11-03/` (git-ignored),
RTX PRO 1000 Blackwell (sm_120), driver 580.178.04, CUDA 12.9.

## Results

The value of `c[0x0][0x358]` read by the descriptor injections in gemm and by the
probe is 0 in both words.

Times are medians of 5 runs, except for the baseline range.

| Flipped bit (of 64) | 64 KiB | 1 MiB | 256 MiB | Output |
|---|---|---|---|---|
| none (baseline) | 51.2-51.4 ms | 158.7 or 185.6 ms | 474-515 ms | reference |
| bit 61 (UR7 bit 29) | 157.9 ms | 158.6 ms (lower baseline level) | within baseline | unchanged |
| bit 59 (UR7 bit 27) | crash | crash | crash | none |
| the other 62 bits | within baseline | within baseline | within baseline | unchanged |
| all 64 bits | crash | crash | crash | none |

The crash is `an illegal instruction was encountered`, with Xid 13 "Illegal
Instruction Parameter" and Xid 43.

## Reading

- **Not the address.** The address is in `R.64`, and no single-bit flip changed
  the data read or written.
- **Compatible with a cache policy, for one bit.** Bit 61 slows the 64 KiB working
  set by 3 times, to the same time as the 1 MiB one, and does nothing on 256 MiB,
  which already misses. This looks like an L1 bypass, but it is an inference from
  timing: no hardware counter was read.
- **Not only a cache policy.** The hardware validates the descriptor: bit 59 alone
  always kills the kernel.
- **The other 62 bits are unexplained, not proven harmless.** They had no effect
  on this access pattern; they may act under conditions it does not exercise.

## Why the gemm pilot saw 37/37 Masked

Of those 37 injections, 15 hit UR7 and none hit bit 27 (bit 59), which has a 1 in
64 chance per injection. Bit 61 was not hit on `gemm_core` either; one hit on
`gemm_tc` (UR15 bit 29) was Masked. `runtime_s` cannot show a timing effect in a
campaign: one warp out of 512 is far below its resolution (median 1.092 s against
1.093 s for all the Masked runs).

**Consequence for the campaigns:** a Masked injection on these registers means
"output unchanged", not "no effect". The label does not capture the slowdown.
