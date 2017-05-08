#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include "cuda_wrapper.hpp"
#include <cmath>
#include "cuda_sqrt.hpp"
#include "test.hpp"

using namespace std;
extern void cuda_doStuff(int * array_in, int * array_out, int N); 
 
void test1(); // cuda function call 

void test2(); // class data manipulation 

// void test3(); // sqrt implementation


int main(int argc, char* argv[])
{
 
  CSqrt cu_sqrt; 
  float sq_v = 5;
  for(int i=0; i<5; i++)
  {
    sq_v = i*15 + 5; 
    cout<<"function : sqrt("<<sq_v<<") = "<<sqrt(sq_v)<<endl; 
    cout<<"cuda_sqrt: = "<<cu_sqrt.g_sqrt(sq_v)<<endl;
  }
  
  // test_thrust(); 
  return 0; 
}


void test2()
{
  A a; 
  doKernel(a);
  cout<<"main.cpp: data in the host: "<<endl;
  thrust::copy(a.h_data.begin(), a.h_data.end(), std::ostream_iterator<int>(std::cout," "));
  cout<<endl;
  cout<<"main.cpp: data in the device: "<<endl;
  thrust::copy(a.data.begin(), a.data.end(), std::ostream_iterator<int>(std::cout," "));
  cout<<endl;
}

void test1()
{
  int numberOfNumbers = 100; 
  int *numbers1_h; 
  int *numbers2_h; 
  
  numbers1_h = (int*)malloc(sizeof(int)*numberOfNumbers); 
  numbers2_h = (int*)malloc(sizeof(int)*numberOfNumbers); 
  // memset(numbers1_h, 0, sizeof(int)*numberOfNumbers); 
  // memset(numbers2_h, 0, sizeof(int)*numberOfNumbers); 
  
  for(int i=0; i<numberOfNumbers; i++)
  {
    numbers1_h[i] = i;
  }
  
  // let the wrapper take care of the communication with the device 
  cuda_doStuff(numbers1_h, numbers2_h, numberOfNumbers); 
  
  int workedCorrectly = 1; 
  for(int i=0; i<numberOfNumbers; i++)
  {
    if(numbers1_h[i]+1 != numbers2_h[i])
    {
      workedCorrectly = 0; 
      printf("something not right! i = %d n1 = %d, n2 = %d \n", i, numbers1_h[i], numbers2_h[i]);
      // break;
    }
  }
  if(workedCorrectly)
    printf("everything is right!\n");
  free(numbers1_h); 
  free(numbers2_h);

}
