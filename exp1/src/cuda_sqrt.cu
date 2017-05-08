
#include "cuda_sqrt.hpp"
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

struct sqrt_impl
{
  int n_size;
  float* p_roots;
  float* p_err;
  float sq_v;
  float ret_v;
};

__global__ void sqrtKernel(sqrt_impl s)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x; 
  if(idx < s.n_size)
  {
    float pv = s.p_roots[idx]; 
    float dis = pv*pv - s.sq_v; 
    s.p_err[idx] = dis*dis;
  } 
}

void CSqrt::init()
{
  errs = thrust::device_vector<float>(Total, 0);
  roots = thrust::device_vector<float>(Total, 0);
  thrust::sequence(roots.begin(), roots.end());
  // thrust::for_each(roots.begin(), roots.end(), assign<MIN_V, MAX_V, Total>());
  thrust::transform(roots.begin(), roots.end(), roots.begin(), assign<MIN_V, MAX_V, Total>());
}

float CSqrt::g_sqrt(float sq_v)
{
  sqrt_impl sq_impl; 
  sq_impl.p_roots = thrust::raw_pointer_cast(&roots[0]); 
  sq_impl.p_err = thrust::raw_pointer_cast(&errs[0]);
  sq_impl.n_size = Total;
  sq_impl.sq_v  = sq_v; 
  
  int block = CSqrt::Threads;
  int grid = (Total + block - 1) / block;      
  
  // std::cout<<"before cuda run, roots: "<<std::endl;
  // thrust::copy(roots.begin(), roots.end(), std::ostream_iterator<float>(std::cout," "));
  // std::cout<<std::endl<<"errors: "<<std::endl;
  // thrust::copy(errs.begin(), errs.end(), std::ostream_iterator<float>(std::cout," "));
  // std::cout<<std::endl;
    
  cudaSafeCall( cudaFuncSetCacheConfig(sqrtKernel, cudaFuncCachePreferL1) );  
  sqrtKernel <<<grid, block>>>(sq_impl);
  
  cudaSafeCall( cudaGetLastError() );
  cudaSafeCall( cudaDeviceSynchronize() );
 
  // std::cout<<"after cuda run errors: "<<std::endl;
  // thrust::copy(errs.begin(), errs.end(), std::ostream_iterator<float>(std::cout," "));
  // std::cout<<std::endl;
 
  // find the minimum error 
  thrust::device_vector<float>::iterator iter = thrust::min_element(errs.begin(), errs.end()); 
  unsigned int position = iter - errs.begin();
  float min_error = *iter; 
  sq_impl.ret_v = roots[position];
  
  // std::cout<<"finally: position is: "<<position<<" min error is : "<<min_error<<" return value: "<<sq_impl.ret_v<<std::endl;
  std::cout<<" min error is : "<<min_error<<" return value: "<<sq_impl.ret_v<<std::endl;

  return sq_impl.ret_v;
}

