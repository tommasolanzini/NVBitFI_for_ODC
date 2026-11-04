#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
__global__ void k_empty() {}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
int main(){for(int i=0;i<8;i++){double a=now();k_empty<<<64,64>>>();cudaDeviceSynchronize();printf("%.1f ",now()-a);}puts("");return 0;}
