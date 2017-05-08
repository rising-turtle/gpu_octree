#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include <stdio.h>

static inline void ___cudaSafeCall( cudaError err, const char *file, const int line )
{
  if( cudaSuccess != err) {
    printf( "%s(%i) : cudaSafeCall() Runtime API error : %s.\n", file, line, cudaGetErrorString( err) );
    exit(-1);
  }
}

#define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__)


#endif
