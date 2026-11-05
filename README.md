# NVBitFI_for_ODC

Fork of [NVBitFI](https://github.com/NVlabs/nvbitfi) used in an MSc thesis on radiation effects in GPUs for orbital data centers. The fault injection results are combined with proton-beam data to map which GPU subsystems need protection.

Main changes with respect to NVBitFI (tested with NVBit 1.5.5, Volta/Turing):

* **NVBit 1.8, CUDA 12, Blackwell (sm_120).** Shared build settings in `common/build.mk` (`NVBIT_PATH`, `ARCH`, `DUMMY`). All kernel launch APIs are intercepted, including `cuLaunchKernelEx`, which CUDA 12 runtimes use.
* **Opcode table for current GPUs.** `common/arch.h` holds one table (opcode → group) generated with an X-macro. It covers the Blackwell instruction set (CUDA Binary Utilities 12.9) and every opcode in the sm_120 SASS of cuBLAS/cuBLASLt 12.9, which has zero unknown opcodes. The original table missed 49 opcodes used on sm_120 (e.g. `LDCU`, `UIADD3`, `R2UR`, `QMMA`, `DMMA`, `LDGSTS`, `LDSM`, `F2FP`) and counted them as `FADD`. Opcodes still missing go to `G_UNKNOWN`: they are reported at run time and never injected.
* **Destination registers from NVBit's decoded operands** (`common/dests.h`) instead of string parsing. It separates GPRs, predicate registers, uniform registers and uniform predicates. It ignores discard targets (`RZ`, `URZ`, `PT`, `UPT`). It computes the right width of 64/128-bit, `IMAD.WIDE` and tensor-core results. Tested offline on 11 M sm_120 instructions (`tests/dests_test.cpp`).
* **Injection into every register class** with the NVBit 1.8 device API: GPRs, **predicate registers**, **uniform registers** and **uniform predicates**.
* **New instruction groups**: `tc` (tensor cores), `sfu` (MUFU), `fp16` (packed FP16/BF16/FP8), `uniform` (uniform datapath), `unknown`.
* **Two-phase injection.** The target is selected BEFORE the instruction, with the same rules and the same instrumentation point as the profiler. It is corrupted AFTER the instruction.
* **Hardware location in the log**: SM id, hardware warp slot, lane, CTA, register class and number.
* **One JSON record per injection** (`results-*.jsonl`), ready for the ML pipeline.
* **Bug fixes**: `SUST` was in two groups, `HSETP2` was in the wrong group (assert crash with `G_GP`), off-by-one in injection-site sampling, `__ballot` illegal on sm_70+, pf_injector deadlock past `INSTRUMENTATION_LIMIT` and injection into predicated-out instructions, Python 3 issues (`map` is lazy so SDC outputs were never saved, `dmesg` bytes/empty output, timeout check).

## Requirements

| | |
|---|---|
| OS | Linux x86_64 (NVBit does not run on Windows; WSL is not supported) |
| NVBit | 1.8 (`nvbit-Linux-x86_64-1.8.tar.bz2` from the NVBit releases) |
| CUDA toolkit | 12.0 or newer; 12.8 or newer for sm_120 |
| NVIDIA driver | **580.xx or older** (NVBit 1.8 requirement) |
| GCC | 8.5 or newer |

## Build

```console
wget https://github.com/NVlabs/NVBit/releases/download/v1.8/nvbit-Linux-x86_64-1.8.tar.bz2
tar xvfj nvbit-Linux-x86_64-1.8.tar.bz2
cd nvbit_release_x86_64/tools/
git clone https://github.com/tommasolanzini/NVBitFI_for_ODC nvbitfi
cd nvbitfi
make -C injector ARCH=sm_120     # ARCH=all (default) builds for every supported GPU
make -C profiler ARCH=sm_120
make -C pf_injector ARCH=sm_120
```

If the repository is somewhere else, pass `NVBIT_PATH=/path/to/nvbit_release_x86_64/core`.

## Instruction groups

Each opcode belongs to one exclusive group, given by the unit that produces its result. `gppr` and `gp` are aggregates defined by the destination operands.

| id | group | what | injectable destinations |
|---|---|---|---|
| 0 | fp64 | DADD, DFMA, DMUL, ... | all destinations (GPR + PR) |
| 1 | fp32 | FADD, FFMA, FMUL, FMNMX, ... | all destinations |
| 2 | ld | LDG, LDS, LDSM, LDC, LDL, ... | all destinations |
| 3 | pr | ISETP, FSETP, DSETP, PLOP3, R2P, ... | predicate registers |
| 4 | nodest | stores, branches, barriers, LDGSTS, REDG, ... | none |
| 5 | others | integer, conversion, move, shuffle, atomics, ... | all destinations |
| 6 | tc | HMMA, IMMA, DMMA, QMMA, OMMA | all destinations |
| 7 | sfu | MUFU | all destinations |
| 8 | fp16 | HADD2, HFMA2, HMUL2, QFMA4, ... | all destinations |
| 9 | uniform | UIADD3, ULEA, LDCU, S2UR, R2UR, VOTEU, ... | uniform registers + uniform predicates |
| 10 | unknown | opcodes missing from `arch.h` | none |
| 11 | gppr | any non-uniform instruction writing a GPR or PR | GPRs + predicate registers |
| 12 | gp | any non-uniform instruction writing a GPR | GPRs (original NVBitFI default) |
| 13 | addr | LD, LDG, LDS, ST, STG, STS, ATOM, ATOMG, ATOMS, RED with a register in the address | not a destination: see "Address at the moment of use" |

Profile counts per group are the **injectable** dynamic instances, i.e. those with at least one destination of the right class and the guard predicate true. Uniform-datapath instructions execute once per warp and are counted once per warp. For 1-bit destinations (predicates), `FLIP_SINGLE_BIT` and `FLIP_TWO_BITS` invert the predicate, `RANDOM_VALUE` writes a random 0/1 and `ZERO_VALUE` writes 0.

## Multi-cell upsets in registers

One particle can flip several neighbouring cells. The physical layout of the register files is not known, so these models say nothing about which cells are really neighbours: model and size are parameters, and every record keeps all the cells hit.

| id | model | what it flips | size k |
|---|---|---|---|
| 4 | `MCU_ADJ_BITS` | k adjacent bits of one register. `FLIP_TWO_BITS` is k = 2 | 2..8 |
| 5 | `MCU_ADJ_REGS` | same bit of Rn, Rn+1, ... of one thread (uniform registers: URn, URn+1, ...) | 2..8 |
| 6 | `MCU_ADJ_LANES` | same bit of the same register in k adjacent lanes of the warp | 2..8 |

Model and size go in one integer, in `inst_value_igid_bfm_map` and in the file names: `mcu_bfm(MCU_ADJ_BITS, 3)` = 43.

- **Predicate registers** (flags of one thread, P0-P6) and **uniform predicates** (flags of one warp, UP0-UP6): models 4 and 5 flip k adjacent bits around the destination, also non-destination flags. Model 6 flips the same predicate in k lanes.
- **Window.** `MCU_ADJ_LANES` draws a window of k lanes holding the selected one. `MCU_ADJ_BITS` draws its window inside the 32 bits.
- **What cannot be reached.** A lane that is not active does not run tool code, so it cannot be written. `MCU_ADJ_REGS` stops at the highest register named by the code of the function: a neighbour above it is not written, so it cannot make a run SDC, and a physically adjacent unused register would be Masked anyway. Reading above it can fail: on `gemm_core` `num_registers` says 40, the code uses up to R37, and DUMMY=1 runs that read R38 ended in `illegal instruction` (cause not established). `requested_cells` is what the model asked for, `cells` what was written, so the shortfall can be counted.
- **Uniform registers have no lanes:** model 6 on group `uniform` is refused at startup.
- `cells` in the JSONL has `reg_no`, `lane`, `tid`, `mask`, `before`, `after` per register. A predicate model with k bits is one cell with k bits in `mask`. The top-level fields stay those of the selected cell.

Default distribution of the sizes, for weighting afterwards and not for sampling: `scripts/mcu_sizes.py`, with its sources and how far they apply to this GPU.

## Address at the moment of use

A transient fault in the path of an address, not in the register: the instruction uses a wrong address, then the register is correct again. Group `addr` (13), model `FLIP_SINGLE_BIT`.

- **BEFORE** the instruction, one bit of a register of the address is flipped. **AFTER**, the register gets its value back, unless the instruction wrote that register itself (`LDG.64 R2, [R2.64]`). The check is per register: `LDG R2, [R2.64]` overwrites R2 but R3 comes back.
- **Which bit:** uniform over the significant bits of all the register parts:

  | Part | Registers | Significant bits |
  |---|---|---|
  | `ra`, global | Rn:Rn+1 | 49, virtual address: assumed, not measured |
  | `ra`, 32-bit | Rn | 32 |
  | `ra`, shared | Rn | 17: 102400 bytes per SM |
  | `ur`, in `[R+UR]` | URn, or URn:URn+1 | 32, or 49 |
  | `desc`, in `desc[UR]` | URn:URn+1 | 64: undocumented, see `tests/desc_probe/README.md` |

- **Immediate offsets are not reachable.** `[R5+0x1fc]`: 0x1fc is in the instruction word, not in a register.
- **Uniform parts hit the whole warp:** `ur` and `desc` are read by all the active lanes for that instruction. `ra` is one thread.
- **`desc` takes most of the flips where it exists** (64 bits against 49), and on its own it does not change the address. Compare the rates per `addr_part`.
- **Record:** `addr_orig`, `addr_new`, `addr_part`, `addr_bit`, `mem_access`, `addr_space`, `addr_valid`, `addr_class`, `store_wrong_location`, `addr_restored`, `addr_overwritten`, and in global memory buffer and offset of both addresses.
- **`addr_valid` means "inside a live allocation of the application"**, or inside the window of the CTA in shared memory. Memory of the tool or of the CUDA runtime is not tracked, so a load can be "invalid" and still not crash.
- **`addr_class`:** `load_valid`, `load_invalid`, `store_invalid`, `atomic_invalid`, `desc_*`, and apart `store_wrong_location` and `atomic_wrong_location`: a write in valid memory, which corrupts another location with no error, and leaves the right one stale.
- **Not generalised to all source operands.** The device code works on (register class, register, bit), so another source needs only a way to list its candidates, but only addresses are implemented.
- With DUMMY=1 `addr_new` is computed and not used.

`scripts/check_fault_locations.py` compares the wrong elements of each run with the record.

Pilot, single seed, not final (20-100 runs per configuration, on gemm, ptr_chase and shm_stencil). DUMMY=1: 240/240
Masked, counts equal to the profile, before == after in all cells. DUMMY=0, address model, SDC/crash/Masked: gemm
13/23/64, ptr_chase 15/27/58, of which the descriptor 0/2/64 and 0/0/58, so read the address register and the
descriptor apart. Location check: 475/486 register SDCs inside the expected elements. All 11 exceptions are
`MCU_ADJ_REGS`, where a neighbouring register can hold an index or a pointer, which displaces a store; one of them
(ptr_chase, `k_walk`, FFMA R0-R2, 24 wrong elements across the warp) is not explained yet. `MCU_ADJ_REGS` can write
fewer cells than asked (574 of 590) because the range stops at the highest register the function names.

## The stack pointer (R1)

The kernels of all the workloads start with `LDC R1, c[0x0][0x37c]` and then never read R1: it is the ABI stack pointer, and they use no stack. NVBit's
instrumentation uses it anyway. So a flip in R1 can change a run through the tool, where the same flip in the real kernel would do nothing.

Measured on `k_walk` of ptr_chase, R1 flips alone, before the fix (`logs/run39-2026-11-05/`):

| Bit flipped | Result |
|---|---|
| 0-2 | crash, `misaligned address` |
| 3-8, 10 | Masked |
| 9, lane 0 | 21-22 threads of the warp never store their result; no other lane gave this, in 40 runs |
| 11-31 | crash, `illegal memory access`, Xid 13 |

So the injector does not write R1 when no instruction of the kernel, or of what it calls, reads it (`reads_gpr`). The flip stays in the record, with
before == after and `sp_not_written`, and in the denominator, as it is a real instance that a real flip would leave masked. After the fix 40/40 flips were Masked.

If a kernel does read R1, the injector writes it and sets `sp_read_by_app`. Then an effect may come from the application or from the tool, and those runs must be
looked at apart. No workload here does. `pf_injector`, not wired into the scripts, does not have this guard. Why lane 0 and bit 9 lose the stores is not established: NVBit is closed.

## Register liveness

`liveness/` measures, for each value written to a GPR, predicate register, uniform register or uniform predicate, the time from the write to the last read
before the next write. Time runs on the thread (GPR, predicate registers) or on the warp (uniform registers and predicates).

```bash
make -C liveness ARCH=sm_120
python3 scripts/run_liveness.py gemm <dir>                                # also checks the golden output
python3 scripts/liveness_report.py occupancy <dir>/liveness.jsonl         # (a) live share per kernel
python3 scripts/liveness_report.py reweight <dir>/liveness.jsonl results-*.jsonl   # (b) rates per bit x time
```

- **Occupancy:** live bit x time over the register bit x time, per kernel and class, against used and allocated registers. Also the share of the whole file of an SM, from the
  resident warps, with the 256 KiB of the specs, not measured.
- **Reweighting:** a campaign draws writes, so a value that lives 1000 instructions counts as one that lives 1. Weight of a run: mean live time of its site x destinations of the
  instruction x bits. `reweight` writes the live time of each sampled instance in `.samples.jsonl`. `dead_effects` lists the runs that changed the outcome on a site never read: these must not exist.

Limits:

- **Cycles are almost all tool time:** about 24000 cycles per instruction on `k_walk`, far above a real instruction. Shares in cycles are therefore close to those in instructions (gemm_core GPR 0.684 and 0.670), and do not show real latency. Use instructions. The residents per SM and the share of the file are also measured under the tool.
- **Register level, not bit level:** reading a value makes all its bits live, also where the instruction ignores some.
- **Site level for instances:** the same `inst_id` does not hit the same thread twice, so an instance gets the mean of its site, not its own lifetime.
- **Time of the flip:** the injector flips right after the write, so all the reads see it. A flip later in the interval misses the earlier reads, so reweighted rates are an upper bound on that side.
- **Implicit sources:** only the operands named in the SASS. `reads_no_write` counts the reads of a register no instruction wrote: 0 on gemm, cnn and ptr_chase. `CALL`/`RET` (sfu_math) were not run here.

Validation: 40 offline source cases, `make -C tests check`. By hand on `k_walk`: `FFMA R0` 6 instructions (one loop), `LDC R1` and `HFMA2 R0` dead. Writes per instruction == profile on gemm and cnn, for `gp` and `uniform`. Output equal to the golden. On 6000 `final-v1` runs on gemm and 200 on cnn, the only runs with an effect on a dead site were the 6 on R1.

## Injection log and results

`nvbitfi-injection-log-temp.txt` now also contains `regType` (GPR/PR/UR/UP), `smID`, `warpID` (hardware warp slot), `laneID`, `ctaID` and `groupInstrs`. `groupInstrs` is the number of injectable instances of the group in the kernel; it must match the profile.

Next to every `results-*.txt` (same format as NVBitFI), `run_one_injection.py` writes `results-*.jsonl` with one JSON object per injection: app, group, model, kernel, PC, opcode, thread, CTA, SM, warp slot, lane, register class and number, mask, before/after value, outcome category, runtime and dmesg.

## First run on a new GPU (checklist)

1. `make -C tests` and `./tests/dests_test --check`, the offline destination analysis (no GPU needed).
2. Profile with `TOOL_VERBOSE=1`. Every static instruction is printed with its group and destinations. Check that there are no `opcode ... is not in common/arch.h` warnings.
3. Build the injector with `DUMMY=1` and run a short campaign. Every outcome must be *Masked* and `groupInstrs` must match the profile.
4. Run a few injections per group with `TOOL_VERBOSE=1`. The log must show `regType`/`regNo`, and no `Expected reg value doesn't match` (sanity check of the NVBit register-read path).
5. Uniform registers and predicates are written by a single lane, the warp leader. Check with a few `G_UNIFORM` and `G_PR` injections that the corrupted value is visible to the whole warp (UR/UP) or to the thread (PR).

---

# Original NVBitFI README

# NVBitFI: An Architecture-level Fault Injection Tool for GPU Application Resilience Evaluations

NVBitFI provides an automated framework to perform error injection campaigns for GPU application resilience evaluation.  NVBitFI builds on top of [**NV**IDIA **Bi**nary **I**nstrumentation **T**ool (NVBit)](https://github.com/NVlabs/NVbit), which is a research prototype of a dynamic binary instrumentation library for NVIDIA GPUs. NVBitFI offers functionality that is similar to a prior tool called [SASSIFI](https://github.com/NVlabs/sassifi).  

Please refer to our [NVBitFI paper](https://ieeexplore.ieee.org/abstract/document/9505068) for additional details about the tool and some experimental results.

# Summary of NVBitFI's capabilities 

NVBitFI injects errors into the destination register values of a dynamic thread-instruction by instrumenting instructions after they are executed.  A dynamic instruction is selected at random from all dynamic kernels of a program for error injection.  Only one error is injected per run.  This mode was referred to as IOV in SASSIFI.  As of now (4/1/2020), NVBitFI allows us to select the following instruction groups to study how errors in them can propagate to the application output.

 * Instructions that write to general purpose registers
 * Single precision floating point instructions
 * Double precision floating point instructions 
 * Load instructions 

NVBitFI can be extended to include custom instruction groups. See below for more details. 

For a selected destination register, following errors can be injected. 

 * Single bit-flip: one bit-flip in one register in one thread
 * Double bit-flip: bit-flips in two adjacent bits in one register in one thread
 * Random value: random value in one register in one thread
 * Zero value: zero out the value of one register in one thread

New bit-flip models can be added by modifying common/arch.h and injector/inject\_func.cu and scripts/params.py. 


# Prerequisites
 * [NVBit v1.5.5](https://github.com/NVlabs/NVBit/releases/tag/1.5.5) or newer
 * [System requirements](https://github.com/NVlabs/NVbit#requirements)

# Getting started on a Linux x86\_64 PC
The following commands are tested on an x86 system with Ubuntu 18.04 using CUDA-11.2 and NVBit version 1.5.5.
```console
# NVBit-v1.5.5
wget https://github.com/NVlabs/NVBit/releases/download/1.5.5/nvbit-Linux-x86_64-1.5.5.tar.bz2
tar xvfj nvbit-Linux-x86_64-1.5.5.tar.bz2
cd nvbit_release/tools/

# NVBitFI 
git clone https://github.com/NVlabs/nvbitfi
cd nvbitfi
find . -name "*.sh" | xargs chmod +x
./test.sh
```
On an ARM-based device (e.g., Jetson Nano)
```console
# NVBit-1.5.5
wget https://github.com/NVlabs/NVBit/releases/download/1.5.5/nvbit-Linux-aarch64-1.5.5.tar.bz2
tar xvfj nvbit-Linux-aarch64-1.5.5.tar.bz2
cd nvbit_release/tools/

# NVBitFI 
git clone https://github.com/NVlabs/nvbitfi
cd nvbitfi
find . -name "*.sh" | xargs chmod +x
./test.sh
```

If these commands complete without errors, you just completed your first error injection campaign using NVBitFI. The printed output should say where the results are stored. Summary of the campaign is stored in a tab-separated file, `results_*NVbitFI_details.tsv`. It can be opened using a spreadsheet program (e.g., Excel) for visualization and analysis. 

# Detailed steps 

There are three main steps to run NVBitFI. We provide a sample script (test.sh) that automates nearly all these steps.

## Step 0: Setup

 * One-time only: Copy NVBitFI package to tool directory in NVBit installation (see the above commands) 
 * Every time we run an injection campaign: Setup environment (see Step 0 (2) in test.sh)
 * One-time only: Build the injector and profiler tools (see Step 0 (3) in test.sh)
 * One-time only: Run and collect golden stdout and stderr files for each of the applications (see Step 0 (4) in test.sh). 
    * Record fault-free outputs: Record golden output file (as golden.txt), stdout (as golden\_stdout.txt), and stderr (as golden\_stderr.txt) in the workload directory (e.g., nvbitfi/test-apps/simple\_add).
    * Create application-specific scripts: Create run.sh and sdc\_check.sh scripts in workload directory. Instead of using absolute paths, please use environment variables for paths such as BIN\_DIR, APP\_DIR, and DATASET\_DIR. These variables are set in set\_env function in scripts/common\_functions.py. See the scripts in the nvbitfi/test-apps/simple\_add directory for examples.
    * Workloads will be run from logs/workload-name/run-name directory. It would be great if the workload can run from this directory. If the program requires input files to be in a specific location, either update the workload or provide soft links to the input files in appropriate locations. 
    * The program output should be deterministic. Please exclude non-deterministic values (e.g., runtimes) from the file if they are present in one of the output files (see test-apps/simple\_add/sdc\_check.sh for more details).

## Step 1: Profile and generate injection list

 * Profile the application: Run the program once by using profiler/profiler.so. We provide scripts/run\_profiler.py script for this step. A new file named nvbitfi-igprofile.txt will be generated in logs/workload-name directory. This file contains the instruction counts for all the instruction groups and opcodes defined in common/arch.h. One line is created per dynamic kernel invocation.
   Profiling is often slow as it instruments every instruction in every dynamic kernel. Using an approximate profile can speed it up by orders of magnitude. There are many ways to approximate a profile and trade-off accuracy for speed. In this release we implement a method that approximates the profiles of all dynamic invocations of a static kernel with the profile of the first invocation of the static kernel. It essentially profiles all static kernels just ones, which can make the profiling very fast if a program has few static kernels and many dynamic involutions per kernel. This approximation can be enabled by using the `SKIP_PROFILED_KERNELS` flag while building the profiler. 
 * Generate injection sites:
    * Ensure that the parameters are set correctly in scripts/params.py.  Following are some of the parameters that need user attention: 
		* Setting maximum number of error injections to perform per instruction group and bit-flip model combination. See NUM\_INJECTION and THRESHOLD\_JOBS in scripts/params.py. 
		* Selecting instruction groups and bit-flip models (more details in scripts/params.py). 
		* Listing the applications, benchmark suite name, application binary file name, and the expected runtime on the system where the injection job will be run. See the apps dictionary in scripts/params.py for an example. The expected runtime defined here is used later to determine when to timeout injection runs (based on the TIMEOUT\_THRESHOLD defined in scripts/params.py).
    * Run scripts/generate\_injection\_list.py to generate a file that contains a list of errors to be injected during the injection campaign. Instructions are selected randomly from the instructions of the selected instruction group. 

## Step 2: Run the error injection campaign

Run scripts/run\_injections.py to launch the error injection campaign. This script will run one injection run at a time in the standalone mode.  If you plan to run multiple injection runs in parallel, please take special care to ensure that the output file is not clobbered. As of now, we support running multiple jobs on a multi-GPU system. Please see scripts/run\_one\_injection.py for more details. 

Tip: Perform a few dummy injections before proceeding with full injection campaign (by setting DUMMY flag in injector/Makefile. Setting this flag will allow you to go through most of the SASSI handler code but skip the error injection. This is to ensure that you are not seeing crashes/SDCs that you should not see.

## Step 3: Parse the results

Use the scripts/parse\_results.py script to parse the results. This script generates three tab-separated values (tsv) files. The first file shows the fraction of executed instructions for different instruction groups and opcodes. The second file shows the outcomes of the error injections.  Refer to CAT\_STR in scripts/params.py for the list of error outcome categories. The third file shows the average runtime for the injection runs for different applications and selected error models. These files can be opened using a spreadsheet program (e.g., Excel) for plotting and analysis.



# NVBitFI vs. SASSIFI

NVBitFI benefits from the featured offered by NVBit. It can run on newer GPUs (e.g., Turing and Volta GPUs). It works with pre-compiled libraries also, unlike SASSIFI. NVBitFI is expected to be faster than SASSIFI as it instruments just a single chosen dynamic kernel (SASSIFI, as it was implemented, instrumented all dynamic kernels) for the injection runs.  As of now (April 14, 2020), NVBitFI implements a subset of the error injection models and we may be expanding this over time (users are more than welcome to contribute). 



# Contributing to NVBitFI

If you are interested in contributing to NVBitFI, please initialize a [Pull Request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) and complete the [Contributor License Agreement](https://www.apache.org/licenses/icla.pdf).
