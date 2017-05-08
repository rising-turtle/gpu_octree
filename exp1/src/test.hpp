#ifndef TEST_HPP
#define TEST_HPP

struct do_stuff
{
  __device__ __host__ do_stuff():curr(0){ scale = 1.2;}
  __device__ __host__ 
    float operator()(float x)
    {
      x = scale*x+1;
      return x;
    }
  float scale;
  int curr; 
};

extern void test_thrust();

#endif
