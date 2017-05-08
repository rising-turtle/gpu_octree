/**
 * test come cuda property 
 * 
 * 
 *
 */

#ifndef CUDA_WRAPPER_HPP
#define CUDA_WRAPPER_HPP

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <iostream>
#include "global.hpp"

class A
{
  public:
    struct KernelPolicy
    {
      enum{
        CTA_SIZE = 16 
      };
    };
  public:
    __host__ A(): n(KernelPolicy::CTA_SIZE){ init();}
    __device__ void assign(int i);
    __host__ void init();
    __host__ void download(thrust::host_vector<int>&);
    __host__ void download();

    void doKernel();
    int* pdata;
    thrust::device_vector<int> data;
    thrust::host_vector<int> h_data;
    int v;
    int n;
};

extern void doKernel(A&);

#endif
