#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "cuda_wrapper.hpp"

__device__ void A::assign(int i)
{
  pdata[i] = v;
  //*(pdata + i) = v;
  // data[i] = v;
  // thrust::device_ptr<int> pd = data.begin(); 
  // pd[i] = v;
}

__host__ void A::download(thrust::host_vector<int>& h)
{
  h = data;
}

__host__ void A::download()
{
  h_data = data;
}

__host__ void A::init()
{
  data = thrust::device_vector<int>(n);
  pdata = thrust::raw_pointer_cast(&data[0]);
}

__global__ void classKernel(A pa, int* tmpa)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x; 
  if(idx < 16)
  {
    pa.assign(idx);
    tmpa[idx] = pa.v;
  }
}

extern void doKernel(A& a)
{
  int block = A::KernelPolicy::CTA_SIZE;
  int grid = (1 + block - 1) / block;      
  a.v = 3;
  
  thrust::device_vector<int> tmpa = a.data;
  int* pta = thrust::raw_pointer_cast(&tmpa[0]);

  cudaSafeCall( cudaFuncSetCacheConfig(classKernel, cudaFuncCachePreferL1) );  
  classKernel <<<grid, block>>>(a, pta);
  
  cudaSafeCall( cudaGetLastError() );
  cudaSafeCall( cudaDeviceSynchronize() );

  std::cout<<"cuda_wrapper.cu: lets' see tmpa!"<<std::endl;
  thrust::copy(tmpa.begin(), tmpa.end(), std::ostream_iterator<int>(std::cout," "));
  a.download();
}

void A::doKernel()
{
  
}

__global__ void processKernel(int *numberArray, int N)
{
  // blockIdx.x is the unique number of the block, in which the thread is positioned
  // blockDim.x is holds the number of threads for each block 
  // threadIdx.x is the number of the thread in this block 
  int idx = blockIdx.x*blockDim.x + threadIdx.x; 
  if(idx < N)
    numberArray[idx] = numberArray[idx] + 1;
}

extern void cuda_doStuff (int *array_in, int *array_out, int N)
{
  int *numbers_d; 
  int numberOfBlocks = 1; 
  int threadsPerBlock = N; 
  int maxNumberOfThreads = N; 

  cudaMalloc((void **)&numbers_d, sizeof(int)*N); 
  cudaMemcpy(numbers_d, array_in, sizeof(int)*N, cudaMemcpyHostToDevice); 
  processKernel<<<numberOfBlocks, threadsPerBlock>>>(numbers_d, maxNumberOfThreads); 
  cudaDeviceSynchronize(); 
  cudaMemcpy(array_out, numbers_d, sizeof(int)*N, cudaMemcpyDeviceToHost); 
  cudaFree(numbers_d);

  return ;
}
