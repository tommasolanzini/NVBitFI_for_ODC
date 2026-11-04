// Times every launch of a kernel from the host, launch + cudaDeviceSynchronize.
#pragma once
#include <stdio.h>
#include <time.h>
#ifndef LAUNCHES
#define LAUNCHES 6
#endif

static inline double repeat_now_us()
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}

#define REPEAT(name, ...) do { \
	for (int repeat_i = 0; repeat_i < LAUNCHES; repeat_i++) { \
		const double repeat_t0 = repeat_now_us(); \
		__VA_ARGS__; \
		cudaDeviceSynchronize(); \
		fprintf(stderr, "LAUNCH %s %d %.1f\n", #name, repeat_i, repeat_now_us() - repeat_t0); \
	} \
} while (0)
