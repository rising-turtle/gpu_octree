
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>

#include "test.hpp"

void test_thrust()
{
  thrust::device_vector<float> X(10); 
  thrust::sequence(X.begin(), X.end());
  thrust::transform(X.begin(), X.end(), X.begin(), do_stuff()); 
  // thrust::for_each(X.begin(), X.end(), do_stuff()); // this will not change the value of the vector
  thrust::copy(X.begin(), X.end(), std::ostream_iterator<float>(std::cout, " "));
}
