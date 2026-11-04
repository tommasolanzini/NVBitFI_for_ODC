# Kernel timing

`make` builds `repeat_<app>`: a copy of a workload, generated at build time, where each kernel launch runs
6 times and prints `LAUNCH <kernel> <index> <us>` on stderr. It exists because the first launch of a kernel
is inflated and the workloads launch each kernel once, on purpose.

`scripts/run_mem_campaign.py profile` runs the copy 10 times, without the tool and with
`CUDA_MODULE_LOADING=EAGER`, and uses the mean of launches 2-5 (0-based) as the kernel duration. Results,
limits and the reason nsys is not used: "The first launch of a kernel" in `mem_injector/README.md`.

The times are from the host: launch plus `cudaDeviceSynchronize`. Do not use these binaries for injection.

`empty_kernel` launches an empty kernel 8 times and prints the host times: the floor used for the lower bound in `scripts/mem_time_bounds.py`.
