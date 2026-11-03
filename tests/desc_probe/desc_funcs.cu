// Device side of desc_tool: XOR a mask into the uniform register(s) written by
// the instrumented LDCU. Uniform registers are shared by the warp, so only one
// lane does it (all of them would toggle it back).
#include <stdint.h>
#include <stdio.h>
#include "nvbit_reg_rw.h"
#include "utils/utils.h"

extern "C" __device__ __noinline__ void xor_ureg(int reg, int wide, uint32_t mask_lo, uint32_t mask_hi, uint64_t precord)
{
	if (get_laneid() != __ffs(__activemask()) - 1) return;
	volatile uint32_t *record = (volatile uint32_t *)precord;
	const uint32_t lo = nvbit_read_ureg((uint64_t)reg);
	const uint32_t hi = wide ? nvbit_read_ureg((uint64_t)reg + 1) : 0;
	nvbit_write_ureg((uint64_t)reg, lo ^ mask_lo);
	if (wide) nvbit_write_ureg((uint64_t)reg + 1, hi ^ mask_hi);
	record[0] = lo;
	record[1] = hi;
	record[2] = nvbit_read_ureg((uint64_t)reg); // read back: the write took effect
	record[3] = wide ? nvbit_read_ureg((uint64_t)reg + 1) : 0;
	record[4] = 1;
	__threadfence_system();
}
