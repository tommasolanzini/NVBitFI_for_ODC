#ifndef PROBE_H
#define PROBE_H

#include <stdint.h>

#define PROBE_SAMPLES 8
#define PROBE_SLOTS 4 // 0 load before, 1 store before, 2 load after, 3 store after

typedef struct {
	unsigned int count[PROBE_SLOTS];
	unsigned long long addr[PROBE_SLOTS][PROBE_SAMPLES];
	unsigned long long trigger;
	unsigned long long offset;
	unsigned int mask;
	int use_offset;
	int fired;
	unsigned int old_word;
	unsigned int new_word;
	unsigned long long flip_addr;
	int flip_cta;
	int flip_tid;
} probe_state_t;

#endif
