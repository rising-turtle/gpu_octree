/*
 * find the x^2 = y, input is y, output is x, using cuda structure
 * 
 * Apr. 30. 2015. 
 * 
 * */

#ifndef CUDA_SQRT_HPP
#define CUDA_SQRT_HPP

#include <cuda.h>
#include <thrust/device_vector.h>
#include "global.hpp"

template<int MIN, int MAX, int NUM>
class assign{
  public:
    assign()
    {
      // curr = 0; // member data does not kept for multi threads 
      scale = (float)(MAX-MIN)/(float)(NUM-1);
    }
    __host__ __device__ 
      float operator()(float x)
    {
      x = MIN + scale*x; 
      // if(++curr == NUM) curr = 0;
      return x;
    }
    float scale; 
    // int curr;
};

class CSqrt
{
  public:
    enum{
      MIN_V = 0,
      MAX_V = 10,
      Blocks = 128,
      Threads = 512,
      Total = Blocks*Threads
    };
    CSqrt(){init();}
    void init();
    float g_sqrt(float);
    thrust::device_vector<float> roots; 
    thrust::device_vector<float> errs;
};

#endif
