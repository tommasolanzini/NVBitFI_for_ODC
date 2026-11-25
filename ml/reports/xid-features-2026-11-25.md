# Xid-detail features, 2026-11-25

Do the details of the Xid messages help stage 1 on crashes? Reduced version (no
allocation bases per run), dataset and network only, no GPU.

- Features: `ml/xid_features.py`, added to `ml/data/stage1_v0.parquet` by
  `ml/build_dataset.py` (prefixes `f_xid13_`, `f_xid31_`). The baseline script
  `ml/train_gbdt.py` excludes them, so the 2026-11-24 baseline is unchanged.
- Training: `ml/train_xid.py`, the folds, seed and parameters of the baseline
  (StratifiedGroupKFold by site, 5 folds, seed 0). Numbers in `xid-features-2026-11-25.json`.
- Run time 55 s.

## What the Xid messages contain (this GPU, driver 580)

The dmesg delta of every run is in the `dmesg` field of its JSONL record (8659/8659 runs
with Xids).

- **Xid 13 (SM exception).** Exception type and the (GPC, TPC, SM) that raised it. **No data
  address:** the ESR words that look like one (`0x1c81fb60`, `0x1174`) are the same in
  every run of all 8 binaries. Types: Out Of Range Address 2164, Misaligned Address 401,
  Invalid Address Space 281, Illegal Instruction Parameter 93.
- **Xid 31 (MMU fault).** Faulting address at **4 KiB page granularity** (bits 0-11 always 0,
  5786 addresses), access type read 4558 / write 1228 (no atomic seen), fault type
  `FAULT_PDE` 5785 of 5786 (not used).
- **Allocation bases.** `mem-profile.json` has no base for `alloc` events. Addresses appear
  only in host-device copies, so buffers that are never copied (cnn intermediates,
  ptr_chase `adj` and `out`) have no known address. Bases **change between processes**
  in bits ≥ 23 (gemm: 0x796f4b800000, 0x779dad800000, 0x77480d800000, 0x7526bd800000 in 4
  profiles). The low 23 bits (offset inside an 8 MiB region) are stable on 2-5
  independent profiles per workload. The golden profile is observable under beam too.

## Features (reduced version)

NaN where not applicable (no Xid 13 / no Xid 31 address / no usable profile).

| feature | meaning |
|---|---|
| `f_xid13_{oor,misaligned,invalid_space,illegal_param,other}` | Xid 13 exception types present (multi-hot; one type per run in practice) |
| `f_xid13_n_sm` | distinct (GPC, TPC, SM) in Xid 13 messages |
| `f_xid31_read`, `f_xid31_write` | access type |
| `f_xid31_low23_in_buf` (a') | the page of the fault address, low 23 bits, overlaps a known buffer |
| `f_xid31_bit_to_buf` (b') | if not (a'): one bit k in 12..22 moves it into a known buffer; `f_xid31_k` = smallest such k |
| `f_xid31_k_rel` (c') | k − log2(element size of that buffer) |
| `f_xid31_k_misaligned` | always NaN: page granularity hides k < 12 |
| `f_xid31_k_ptr_only`, `f_xid31_k_ambiguous` | k > 31 + log2(element) never happens for k ≤ 22: always 0 and 1 (constant, ignored by the trees) |
| `f_xid31_hamming_min` (d') | minimum Hamming distance, bits 12-22, from the pages of the known buffers (computed always: 0 if (a')) |
| `f_xid31_above_2_47` | fault address ≥ 2^47 (outside the user virtual space) |

- The number of Xid 13 lines and of Xid 31 messages are already features of the baseline
  (`f_xid_sm_exception_count`, `f_xid_mmu_fault_count`). They were not duplicated.
- No address test (NaN) on gemm_n1024, where the known buffers cover 2.5 × the 8 MiB region,
  and on gemm_n2048 (no profile).
- Element sizes come from the sources: gemm half buffers 2 B, ptr_chase `heads` 8 B and
  `nodes` 16 B, everything else 4 B.

**Coverage on training crashes** (crash + timeout runs, 4645):

| app | crashes | with Xid 31 address | with address test |
|---|---|---|---|
| cnn | 791 | 647 | 647 |
| cnn_large | 103 | 84 | 84 |
| gemm | 844 | 730 | 730 |
| gemm_n1024 | 111 | 92 | 0 |
| gemm_n2048 | 92 | 70 | 0 |
| ptr_chase | 1163 | 884 | 884 |
| sfu_math | 624 | 496 | 496 |
| shm_stencil | 917 | 41 | 41 |

Training rows only (the MCU test set is not used here). shm_stencil crashes
are almost all Xid 13 (shared memory, no MMU fault).

**Chance level of the address test** (uniform random page in the 8 MiB region):

| app | (a') by chance | (b') by chance, given not (a') |
|---|---|---|
| gemm | 0.04 | 0.19 |
| cnn | 0.02 | 0.12 |
| shm_stencil | 0.02 | 0.13 |
| sfu_math | 0.02 | 0.10 |
| cnn_large | 0.13 | 0.48 |
| **ptr_chase** | **0.51** | **1.00** |

On ptr_chase the test is close to uninformative: the known buffers cover half of the
region (nodes alone 4 MiB).

## Results (same folds as the baseline)

| variant | SDC macro-F1 / acc | crash macro-F1 / acc | crash w/o ptr_chase | all macro-F1 |
|---|---|---|---|---|
| A | 0.366 / 0.420 | 0.237 / 0.419 | 0.281 / 0.442 | 0.372 |
| **A+xid** | 0.371 / 0.422 | **0.276 / 0.445** | **0.314 / 0.489** | 0.380 |
| B2 | 0.364 / 0.417 | 0.133 / 0.240 | 0.187 / 0.304 | 0.343 |
| **B2+xid** | 0.363 / 0.416 | **0.195 / 0.367** | **0.277 / 0.445** | 0.358 |

A and B2 reproduce the baseline report exactly, so the folds are the same.

- **SDCs do not change in a meaningful way.** The new features are NaN on all 17572 SDC rows
  (checked). Macro-F1 moves by +0.005 (A) and −0.001 (B2), within the noise. Still, 817 (A)
  and 788 (B2) SDC predictions change, 4.6%. The model is trained on all rows at once, so
  the new features change the trees and the early-stopping point (iterations 26-32 → 20-29),
  and with them some SDC predictions. Not a leak.
- **Crashes improve:** +0.04 macro-F1 in A, +0.06 in B2. B2+xid without ptr_chase (0.277)
  comes close to A without ptr_chase (0.281). On crashes the Xid details recover most of
  what variant A got from knowing the workload and kernel.

### Crash confusion, row-normalised, before → after

Columns: predicted class. Diagonal in bold.

**A → A+xid**

| true \ pred | int | ld_const | ld_data | addr | move | mem_global | ur | pr | up | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| int | **.36 → .42** | .24 → .17 | . | .21 → .20 | .15 → .12 | . | .03 → .05 | . → .03 | .01 | 829 |
| ld_const | .09 → .11 | **.51 → .41** | .01 | .07 → .17 | .26 → .19 | . | .06 → .08 | . → .01 | . → .01 | 246 |
| ld_data | . | . | **. → .** | .80 → .62 | . | .20 → .38 | . | . | . | 149 |
| addr | .15 → .14 | .22 → .16 | . | **.49 → .45** | .10 → .09 | . → .01 | .01 → .12 | . | .03 → .02 | 1458 |
| move | .03 → .05 | .44 → .25 | . | .05 | **.36 → .46** | . → .01 | .10 → .18 | . | .03 → .01 | 101 |
| mem_global | . | . | .07 → .11 | .76 → .57 | .01 | **.15 → .32** | .01 → .00 | . | . | 361 |
| ur | .06 | .12 → .09 | . → .04 | .11 | .07 | .05 → .01 | **.51 → .56** | . | .09 → .05 | 1411 |
| pr | . → .66 | . | . | 1.00 → .34 | . | . | . | **. → .** | . | 77 |
| up | . | . | . | . → .08 | .31 | . | .46 → .08 | . | **.23 → .54** | 13 |

**B2 → B2+xid**

| true \ pred | int | ld_const | ld_data | addr | move | mem_global | ur | pr | up | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| int | **.06 → .26** | .85 → .25 | . | .03 → .23 | .04 → .12 | . | .02 → .08 | . | . → .06 | 829 |
| ld_const | .09 → .12 | **.85 → .24** | . | .02 → .33 | .02 → .12 | . | .03 → .17 | . | . → .02 | 246 |
| ld_data | . | .95 → .38 | **. → .** | .05 → .62 | . | . | . | . | . | 149 |
| addr | .06 → .07 | .82 → .28 | . | **.09 → .40** | .02 → .08 | . | .01 → .16 | . | . → .01 | 1458 |
| move | . → .03 | .75 → .16 | . | . → .21 | **.17 → .45** | . | .08 → .10 | . | . → .06 | 101 |
| mem_global | .08 | .82 → .42 | . | .07 → .48 | .02 | **.01 → .00** | . | . | . | 361 |
| ur | . | .48 → .14 | . | . → .19 | .02 → .06 | . | **.50 → .56** | . | . → .05 | 1411 |
| pr | . | 1.00 → . | . | . → .34 | . | . | . | **.** | . → .66 | 77 |
| up | . | .92 → . | . | . | . | . | .08 → .15 | . | **. → .85** | 13 |

Crash F1 per class:

| class | A | A+xid | B2 | B2+xid |
|---|---|---|---|---|
| addr | 0.48 | 0.47 | 0.16 | 0.41 |
| int | 0.41 | 0.45 | 0.10 | 0.36 |
| ld_const | 0.23 | 0.23 | 0.11 | 0.09 |
| ld_data | 0.00 | 0.00 | 0.00 | 0.00 |
| mem_global | 0.21 | **0.41** | 0.01 | 0.00 |
| move | 0.13 | 0.17 | 0.16 | 0.19 |
| pr | 0.00 | 0.00 | 0.00 | 0.00 |
| up | 0.03 | 0.11 | 0.00 | 0.10 |
| ur | 0.65 | 0.65 | 0.65 | 0.62 |

- In B2, without the Xid details, almost every crash collapsed into ld_const. With them,
  addr, int, move and up get their own predictions back.
- mem_global vs ld_data (A+xid): both live only in ptr_chase, where the address test is
  close to chance. They separate only through the workload marker (A: mem_global F1 0.41;
  B2+xid: 0.00).
- **pr and up crashes become one group** (B2+xid: pr → up 0.66). See k = 22 below.

### Most confused pairs on crashes, after

| A+xid pair | a→b | b→a | B2+xid pair | a→b | b→a |
|---|---|---|---|---|---|
| int ↔ pr | 0.03 | 0.66 | pr ↔ up | 0.66 | 0.00 |
| addr ↔ ld_data | 0.00 | 0.62 | addr ↔ ld_data | 0.00 | 0.62 |
| addr ↔ mem_global | 0.01 | 0.57 | addr ↔ ld_const | 0.28 | 0.33 |
| ld_data ↔ mem_global | 0.38 | 0.11 | addr ↔ mem_global | 0.00 | 0.48 |
| ld_const ↔ move | 0.19 | 0.25 | ld_const ↔ mem_global | 0.00 | 0.42 |
| addr ↔ pr | 0.00 | 0.34 | ld_const ↔ ld_data | 0.00 | 0.38 |
| addr ↔ int | 0.14 | 0.20 | int ↔ ld_const | 0.25 | 0.12 |
| addr ↔ ld_const | 0.16 | 0.17 | addr ↔ ur | 0.16 | 0.19 |
| move ↔ up | 0.01 | 0.31 | addr ↔ pr | 0.00 | 0.34 |
| int ↔ ld_const | 0.17 | 0.11 | ld_const ↔ ur | 0.17 | 0.14 |

### Crashes per workload (macro-F1 / accuracy)

| workload | crashes | A | A+xid | B2 | B2+xid |
|---|---|---|---|---|---|
| cnn | 894 | 0.150 / 0.17 | 0.266 / 0.33 | 0.189 / 0.20 | 0.284 / 0.34 |
| gemm | 1047 | 0.202 / 0.44 | 0.242 / 0.49 | 0.068 / 0.08 | 0.216 / 0.41 |
| sfu_math | 624 | 0.094 / 0.17 | 0.071 / 0.11 | 0.156 / 0.22 | 0.192 / 0.26 |
| shm_stencil | 917 | 0.438 / 0.90 | 0.444 / 0.90 | 0.334 / 0.72 | 0.324 / 0.72 |
| all except ptr_chase | 3482 | 0.281 / 0.44 | 0.314 / 0.49 | 0.187 / 0.30 | 0.277 / 0.45 |
| **ptr_chase** (buffers cover 51% of the region) | 1163 | 0.107 / 0.35 | 0.131 / 0.32 | 0.049 / 0.05 | 0.063 / 0.14 |

cnn and gemm gain the most; they have the most Xid 31 addresses with a usable test.
shm_stencil crashes are Xid 13 only (no address) and do not change. sfu_math does not gain
in A. ptr_chase gains almost nothing, as expected from the chance levels.

## SHAP on crashes (A+xid final model, crash rows)

Global top 10: `f_frac_wrong` 1.64, `f_log10_max_rel` 1.42, `f_xid_sm_exception_count` 1.08,
`f_frac_outputs_wrong` 1.06, `fw_wl_ptr_chase` 0.98, `f_hamming_mean` 0.92, `f_bbox_fill` 0.90,
`f_frac_cols` 0.74, `f_log10_wrong` 0.71, **`f_xid31_read` 0.64** (first new feature).

The output-signature features rank high on crashes because they are NaN there: they act as
"which kernel / which outputs exist" markers through the NaN pattern, with the
completed-kernel features. Among the new features, per class:

| class | top features (crash) | new features in top 6 |
|---|---|---|
| addr | f_bbox_fill, f_xid_sm_exception_count, fw_log10_size, f_hamming_mean, **f_xid31_low23_in_buf**, fw_wl_sfu_math | low23_in_buf |
| move | f_log10_max_rel, fw_wl_gemm, f_msb_max, **f_xid13_invalid_space**, f_pattern_scattered, **f_xid31_read** | invalid_space, read |
| ur | f_frac_cols, f_xid_sm_exception_count, f_log10_wrong, **f_xid31_read**, **f_xid31_hamming_min**, f_frac_kernels_completed | read, hamming_min |
| up | **f_xid31_k**, **f_xid31_read**, f_frac_kernels_completed, fw_wl_gemm, fw_log10_size, f_frac_wrong | k, read |
| ld_data | f_log10_max_rel, fw_wl_ptr_chase, f_hamming_mean, **f_xid31_read**, f_frac_outputs_wrong, f_rank_log2 | read |
| int, ld_const, mem_global, pr | output-signature NaN pattern and workload | none |

## Distribution of the new features per true class (training crashes)

Shares among the runs where the feature is defined. (a'), (b') and above 2^47 exclude
ptr_chase in the second table.

| class | crashes | with Xid 13 | oor | misaligned | invalid space | illegal param | n SM | with Xid 31 address | read | write |
|---|---|---|---|---|---|---|---|---|---|---|
| addr | 1458 | 360 | 0.54 | 0.36 | 0.06 | 0.04 | 1.0 | 1098 | 0.67 | 0.33 |
| int | 829 | 204 | 0.70 | 0.13 | 0.18 | 0 | 1.0 | 625 | 0.87 | 0.13 |
| ld_const | 246 | 57 | 0.81 | 0.11 | 0.09 | 0 | 1.0 | 189 | 0.70 | 0.30 |
| ld_data | 149 | 38 | 0.79 | 0.21 | 0 | 0 | 1.0 | 111 | 1.00 | 0 |
| mem_global | 361 | 90 | 0.71 | 0.19 | 0.08 | 0 | 0.98 | 256 | 1.00 | 0 |
| move | 101 | 26 | 0.31 | 0 | **0.69** | 0 | 1.0 | 75 | 0.81 | 0.19 |
| pr | 77 | 0 | - | - | - | - | - | 77 | 1.00 | 0 |
| up | 13 | 1 | 0 | 0 | 1.00 | 0 | 1.0 | 12 | 0.92 | 0.08 |
| ur | 1411 | **811** | 0.83 | 0.07 | 0.05 | 0.05 | 1.0 | 601 | 0.62 | 0.38 |

| class | (a') low23 in buffer, all | (a') w/o ptr_chase | (b') w/o ptr_chase | above 2^47 w/o ptr_chase | k (all runs with b') |
|---|---|---|---|---|---|
| addr | 0.78 | 0.75 (789) | 0.80 (194) | 0.08 | 14:48 16:13 17:13 18:55 19:51 22:18 |
| int | 0.68 | 0.67 (452) | 0.77 (148) | 0.23 | 18:18 19:20 20:16 22:50 |
| ld_const | 0.68 | 0.70 (174) | 0.79 (53) | 0.25 | 14:6 16:6 18:12 19:13 22:6 |
| ld_data | 1.00 | - (ptr_chase only) | - | - | - |
| mem_global | 1.00 | - (ptr_chase only) | - | - | - |
| move | 0.39 | 0.36 (70) | 0.60 (45) | 0.01 | spread 14-22 |
| pr | 0.34 | **0.00** (51) | 1.00 (51) | 0 | **22:51** |
| up | 0.08 | 0.08 (12) | 1.00 (11) | 0 | **22:11** |
| ur | 0.73 | 0.64 (450) | 0.81 (162) | 0.20 | 16:8 18:20 19:32 22:69 |

Chance levels outside ptr_chase: (a') 0.02-0.04, (b') 0.10-0.19.

- **The page offset of a faulting pointer is usually preserved.** For addr, int, ld_const
  and ur, (a') is 0.64-0.75 against 0.02-0.04 by chance. The fault address keeps the low 23
  bits of a valid buffer, i.e. the corruption moved bits ≥ 23 (or a base register), not
  the offset. When it does not, one bit in 12-22 explains it in 0.77-0.81 of the cases
  (chance 0.10-0.19).
- **move:** Xid 13 Invalid Address Space in 0.69 of its Xid 13 crashes (others ≤ 0.18),
  and (a') only 0.36. A moved or converted integer used as an address points to another
  address space or to nothing.
- **ur:** Xid 13 in 57% of its crashes (others 25%), mostly Out Of Range. A uniform value
  hits every lane of the warp.
- **Above 2^47:** 0.20-0.25 for int, ld_const and ur, 0.08 for addr, 0 for pr, up and move.
  A flip of a high bit (≥ 47) of a 64-bit value is visible here; it shows up more in
  integer and pointer values than in the address register at use.
- **ld_data and mem_global:** reads only, (a') 1.00, but both exist only in ptr_chase, where
  (a') is 0.51 by chance.
- **k = 22 is not a bit flip.** In 137 of the 211 runs with k = 22 the low 23 bits are exactly
  `0x400000`: the first page after a 4 MiB mapped chunk. A sequential access ran off the
  end of the mapped memory (a loop or an index running too far). That is the case for all
  pr crashes outside ptr_chase (51/51, the gemm loop-control predicates) and all up crashes
  (11/11). It also covers ur (55 of 69) and int (20 of 50); addr has 5 at `0x420000`. This feature
  carries control-flow information, not bit information. Proposal: replace k = 22 at
  `0x400000` with an explicit flag "fault at the end of the mapped chunk" and keep k only
  for the others.
- `f_xid31_k_rel` adds nothing beyond k (element sizes are 4 B almost everywhere). The
  flags of (c') are constant with page granularity: misaligned never visible, pointer-only
  never true, ambiguous always.

## Conclusions

1. The Xid details help on crashes: +0.04 macro-F1 (A) and +0.06 (B2). Without the
   workload markers (B2) they recover most of the crash information. They do not change
   SDCs.
2. Crash classes stay hard (best crash macro-F1 0.28). What separates:
   - ur (Xid 13 share);
   - move (Invalid Address Space);
   - addr/int/ld_const vs pr/up (page offset preserved vs fault at the end of a chunk).

   ld_data vs mem_global vs addr cannot be told apart here: they live in ptr_chase, where
   the address test is uninformative.
3. The bit-level questions of the original plan (misaligned → pointer only, k beyond a
   32-bit index → pointer only) cannot be answered on this GPU. Xid 13 has no address,
   and Xid 31 drops bits 0-11 and gives no per-run bases. This is a requirement for Orin
   below.

## Requirements for the Orin campaign (added to the list of the baseline report)

11. **Every workload logs base and size of every `cudaMalloc`** in each run (observable
    under beam). The address features then use the bases of that process, not the stable
    low bits of a golden profile.
12. **mem_injector writes the base address in the `alloc` events** of `mem-profile.json`
    (the tool has it at `mem_injector.cu:702`, it is not written).
13. **Check the address granularity of nvgpu fault messages on Orin** (full address or
    page only) and whether SM exceptions report an address.
14. Proposal: workloads whose buffers cover a small part of the region, or one region per
    buffer, so the address test stays informative (ptr_chase today: 51% coverage).

## Open points

- Flag "fault at the end of a 4 MiB chunk" in place of k = 22 (see above).
- The output-signature NaN pattern acts as a kernel marker on crashes in variant A. It is
  part of A by design; B2 removes the explicit markers but not this pattern.
