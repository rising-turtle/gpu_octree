
#include "trans_impl.hpp"
// #include "trans.h"

// #include "pcl/gpu/utils/timers_cuda.hpp"
// #include "pcl/gpu/utils/device/funcattrib.hpp"
// #include "pcl/gpu/utils/device/algorithm.hpp"
// #include "pcl/gpu/utils/device/static_check.hpp"
// #include "utils/scan_block.hpp"
// #include "utils/morton.hpp"

#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <iostream>

#include "transform_helper.hpp"

using namespace pcl::gpu;
using namespace thrust;

namespace pcl
{
    namespace device
    {
        __global__ void get_cc_kernel(int *data)
        {
            data[threadIdx.x + blockDim.x * blockIdx.x] = threadIdx.x;
        }
    }
}

namespace pcl
{
  namespace device
  {
    template<typename PointType>
      struct PointType_to_tuple
      {
        __device__ __forceinline__ thrust::tuple<float, float, float> operator()(const PointType& arg) const
        {
          thrust::tuple<float, float, float> res;
          res.get<0>() = arg.x;
          res.get<1>() = arg.y;
          res.get<2>() = arg.z;
          return res;
        }
      };
    
    class impl_set
    {
      public:
        __device__ __forceinline__ float sq(float x)
        {
          return (x*x);
        }
        __device__ __forceinline__ float brfc(float3 p)
        {
          float dis_min = minimum_distance; 
          for(int i=0; i<tar_n; i++)
          {
            float dis = sq(p.x - tar_x[i]) + sq(p.y - tar_y[i]) + sq(p.z - tar_z[i]);
            if(dis < dis_min) dis_min = dis;
          }
          return dis_min;
        }
      public:
        // transformation pointers
        float* tx;        float* ty;        float* tz; 
        float* r1;        float* r2;        float* r3;
        float* r4;        float* r5;        float* r6; 
        float* r7;        float* r8;        float* r9;

        // point cloud pointers 
        float* src_x;     float* src_y;     float* src_z;
        float* tar_x;     float* tar_y;     float* tar_z;
        int src_n;      // number of points in the source cloud 
        int tar_n;      // number of points in the target cloud

        // parameters for matched pair 
        float minimum_distance; // points distance threshold
        
        // error pointer
        float * err;
    };
    
    class test_impl{
      public:
        impl_set impl_cp; 
        float3 src_pt; 
        float3 tar_pt;
    };
    
    __global__ void testKernel(test_impl mt)
    {
      impl_set m = mt.impl_cp; 
      int idx = blockIdx.x*blockDim.x + threadIdx.x;  
      float3 xyz  = make_float3(m.tx[idx], m.ty[idx], m.tz[idx]); 
      float3 r123 = make_float3(m.r1[idx], m.r2[idx], m.r3[idx]); 
      float3 r456 = make_float3(m.r4[idx], m.r5[idx], m.r6[idx]); 
      float3 r789 = make_float3(m.r7[idx], m.r8[idx], m.r9[idx]);
      TransformHelper trans(r123, r456, r789, xyz); 
      float3 p1 = mt.src_pt; 
      float3 p2 = trans(p1); 
      float err = m.sq(p2.x - mt.tar_pt.x) + m.sq(p2.y - mt.tar_pt.y) + m.sq(p2.z - mt.tar_pt.z); 
      m.err[idx] = err;
      return ;
    }

    // distributing the search area 
    __global__ void transKernel(impl_set m)
    {
       int idx = blockIdx.x*blockDim.x + threadIdx.x;  
       float3 xyz  = make_float3(m.tx[idx], m.ty[idx], m.tz[idx]); 
       float3 r123 = make_float3(m.r1[idx], m.r2[idx], m.r3[idx]); 
       float3 r456 = make_float3(m.r4[idx], m.r5[idx], m.r6[idx]); 
       float3 r789 = make_float3(m.r7[idx], m.r8[idx], m.r9[idx]);
       TransformHelper trans(r123, r456, r789, xyz); 
       
       // TODO: using nested gpu parallem method 
       // find matched pair of points for each point, just brust force
       float min_dis = m.minimum_distance; 
       float dis;
       float3 p_src_t;
       float3 p_src;
       int sum_num = 0; 
       float sum_sq_err = 0;
       float eps = 1e-8;
       
       for(int i=0; i<m.src_n; i++)
       {
          p_src = make_float3(m.src_x[i], m.src_y[i], m.src_z[i]);
          p_src_t = trans(p_src);
          
          // sum_num = 0; 
          // sum_sq_err = 0;
          
          dis = m.brfc(p_src_t);
          // dis = min_dis - 0.001;

          if(dis < min_dis)
          {
            ++sum_num;
            sum_sq_err += dis;
          }
       }
       if(sum_num > 0)
       {
          if(sum_sq_err < eps) 
            m.err[idx] = FLT_MAX;
          else
            m.err[idx] = sum_num*1./sum_sq_err;
       }
       
    }
  }
}

void pcl::device::TransformImpl::findTrans(thrust::host_vector<float>& result_h)
{
  impl_set cuda_impl;
 
  // transformation 
  cuda_impl.tx = thrust::raw_pointer_cast(&tx[0]);
  cuda_impl.ty = thrust::raw_pointer_cast(&ty[0]); 
  cuda_impl.tz = thrust::raw_pointer_cast(&tz[0]); 
  cuda_impl.r1 = thrust::raw_pointer_cast(&r1[0]);
  cuda_impl.r2 = thrust::raw_pointer_cast(&r2[0]); 
  cuda_impl.r3 = thrust::raw_pointer_cast(&r3[0]); 
  cuda_impl.r4 = thrust::raw_pointer_cast(&r4[0]); 
  cuda_impl.r5 = thrust::raw_pointer_cast(&r5[0]);
  cuda_impl.r6 = thrust::raw_pointer_cast(&r6[0]); 
  cuda_impl.r7 = thrust::raw_pointer_cast(&r7[0]); 
  cuda_impl.r8 = thrust::raw_pointer_cast(&r8[0]); 
  cuda_impl.r9 = thrust::raw_pointer_cast(&r9[0]);
  
  // point cloud 
  cuda_impl.src_x = src_points_xyz.ptr(0); 
  cuda_impl.src_y = src_points_xyz.ptr(1); 
  cuda_impl.src_z = src_points_xyz.ptr(2);
  
  cuda_impl.tar_x = tar_points_xyz.ptr(0); 
  cuda_impl.tar_y = tar_points_xyz.ptr(1); 
  cuda_impl.tar_z = tar_points_xyz.ptr(2);
  
  // points size 
  cuda_impl.tar_n = tar_points_xyz.cols();
  cuda_impl.src_n = src_points_xyz.cols();

  // err initial set 
  cuda_impl.minimum_distance = MIN_DIS_THRE * 0.001;
  cuda_impl.minimum_distance = cuda_impl.minimum_distance * cuda_impl.minimum_distance; 
  float e = 1./(cuda_impl.minimum_distance); 
  //thrust::fill(se.begin(), se.end(), e); 
  thrust::find(se.begin(), se.end(), 0);
  cuda_impl.err = thrust::raw_pointer_cast(&se[0]);

  // main stuff 
  // cudaSafeCall( cudaFuncSetCacheConfig(pcl::device::transKernel, cudaFuncCachePreferL1) );  
  // cudaSafeCall( cudaFuncSetCacheConfig(pcl::device::testKernel, cudaFuncCachePreferL1) );  

  int block = BLOCK_N;
  int grid =  GRID_N; //(Total + block - 1) / block;      
  
  // test_impl cuda_test_impl; 
  // cuda_test_impl.impl_cp = cuda_impl; 
  // float3 src_pt = make_float3(1, 2.4, 1.0);
  // float3 tar_pt = make_float3(1.03, 2.35, 0.94); //(1.2, 2.0, 0.7); // make_float3();
  
  pcl::device::transKernel<<<grid, block>>>(cuda_impl);

  // cuda_test_impl.src_pt = src_pt; 
  // cuda_test_impl.tar_pt = tar_pt;
  // pcl::device::testKernel<<<grid, block>>>(cuda_test_impl);
  
  cudaSafeCall( cudaGetLastError() );
  cudaSafeCall( cudaDeviceSynchronize() );
  
  // find the best score 
  thrust::device_vector<float>::iterator iter = thrust::max_element(se.begin(), se.end()); 
  // thrust::device_vector<float>::iterator iter = thrust::min_element(se.begin(), se.end()); 
  // thrust::copy(se.begin(), se.end(), std::ostream_iterator<float>(std::cout, " "));
  std::cout<<std::endl<<" max score: "<<*iter<<" score threshold: "<<e<<std::endl;
  unsigned int position = iter - se.begin(); 
  result_h.resize(12); 
  result_h[0] = tx[position]; result_h[1] = ty[position]; result_h[2] = tz[position]; 
  result_h[3] = r1[position]; result_h[4] = r2[position]; result_h[5] = r3[position];
  result_h[6] = r4[position]; result_h[7] = r5[position]; result_h[8] = r6[position];
  result_h[9] = r7[position]; result_h[10] = r8[position]; result_h[11] = r9[position];
    
  return ;
}

void pcl::device::TransformImpl::init()
{
  int points_num = MAX_NUM_POINTS; // maximum storage
  const int transaction_size = 128 / sizeof(int);
  int cols = max<int>(points_num, transaction_size * 4);
  int rows = 6;
  host_data.downloaded_ = false;

  // memory allocate 
  storage.create(rows, cols); 
  // points_xyz = DeviceArray2D<float>(rows, points_num, storage.ptr(0), storage.step());
}

void pcl::device::TransformImpl::build()
{
  /*
  using namespace pcl::device; 
  int points_num = (int)src_points.size(); 

  // memory assignment 
  device_ptr<PointType> beg(src_points.ptr());
  device_ptr<PointType> end = beg + src_points.size();
  
  device_ptr<float> xs(src_points_xyz.ptr(0)); 
  device_ptr<float> ys(src_points_xyz.ptr(1)); 
  device_ptr<float> zs(src_points_xyz.ptr(2)); 
  
  thrust::transform(beg, end, make_zip_iterator(make_tuple(xs, ys, zs)), PointType_to_tuple<PointType>());
  */
}

void pcl::device::TransformImpl::setSourceCloud(const PointCloud& pc) 
{
  // src_points = pc;
  // host_data.downloaded_ = false;
  host_data.points_num = pc.size();

  int points_num = (int)pc.size(); 

  PointCloud src_points = pc;
  // memory assignment 
  device_ptr<PointType> beg(src_points.ptr());
  device_ptr<PointType> end = beg + src_points.size();
  
  src_points_xyz = DeviceArray2D<float>(3, points_num, storage.ptr(0), storage.step());
  
  device_ptr<float> xs(src_points_xyz.ptr(0)); 
  device_ptr<float> ys(src_points_xyz.ptr(1)); 
  device_ptr<float> zs(src_points_xyz.ptr(2)); 
  
  thrust::transform(beg, end, make_zip_iterator(make_tuple(xs, ys, zs)), PointType_to_tuple<PointType>());
}

void pcl::device::TransformImpl::setTargetCloud(const PointCloud& pc) 
{
  // tar_points = pc;
  int points_num = (int)pc.size(); 

  PointCloud tar_points = pc;
  // memory assignment 
  device_ptr<PointType> beg(tar_points.ptr());
  device_ptr<PointType> end = beg + tar_points.size();
  
  tar_points_xyz = DeviceArray2D<float>(3, points_num, storage.ptr(3), storage.step());
  
  device_ptr<float> xs(tar_points_xyz.ptr(0)); 
  device_ptr<float> ys(tar_points_xyz.ptr(1)); 
  device_ptr<float> zs(tar_points_xyz.ptr(2)); 
  
  thrust::transform(beg, end, make_zip_iterator(make_tuple(xs, ys, zs)), PointType_to_tuple<PointType>());
}

void pcl::device::TransformImpl::transform(float3 rpy123, float3 rpy456, float3 rpy789, float3 xyz)
{
  using namespace pcl::device; 
  unsigned int points_num = src_points_xyz.cols(); 
  device_ptr<float> xs(src_points_xyz.ptr(0)); 
  device_ptr<float> ys(src_points_xyz.ptr(1)); 
  device_ptr<float> zs(src_points_xyz.ptr(2)); 
  
  thrust::transform(make_zip_iterator(make_tuple(xs, ys, zs)), 
                    make_zip_iterator(make_tuple(xs + points_num, ys + points_num, zs + points_num)), 
                    make_zip_iterator(make_tuple(xs, ys, zs)), TransformHelper(rpy123, rpy456, rpy789, xyz));
}

void pcl::device::TransformImpl::internalDownload()
{
  src_points_xyz.download(host_data.points_xyz, host_data.points_step);
  host_data.downloaded_ = true;
}

void pcl::device::TransformImpl::get_gpu_arch_compiled_for(int& bin, int& ptx)
{
    cudaFuncAttributes attrs;
    cudaSafeCall( cudaFuncGetAttributes(&attrs, get_cc_kernel) );  
    bin = attrs.binaryVersion;
    ptx = attrs.ptxVersion;
}
