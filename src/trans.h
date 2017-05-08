/*
 * handle pcl::PointXYZ part, as a middleware to trans_impl, which is operated in GPU space 
 * Apr. 29 2015, David Z
 *
 * */

#ifndef TRANS_H
#define TRANS_H

#include <vector>
#include <iostream>

#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/octree/device_format.hpp>
#include <pcl/cuda/cutil_math.h>
#include <thrust/host_vector.h>

#define D2R(d) (((d)*(M_PI))/180.)
#define R2D(r) (((r)*180.)/(M_PI))
#define R_STEP ((M_PI)/180.)

#define SQ(x) ((x)*(x))

// extern std::ostream& operator<<(std::ostream& out, Eigen::Affine3f& m);

namespace pcl
{
  namespace gpu
  {
    class CTrans
    {
    public:
      CTrans();
      ~CTrans();
      typedef pcl::PointXYZ PointType;
      typedef DeviceArray<PointType> PointCloud;
      typedef DeviceArray<int> Indices;    
      
      typedef pcl::PointCloud<PointType> HostPointCloud;
      
      void initTrans();
      float err_match(HostPointCloud& tar_pc, HostPointCloud& src_pc, float3 xyz, float3 r123, float3 r456, float3 r789);
      void findTransformationHost(HostPointCloud& tar_pc, HostPointCloud& src_pc, Eigen::Affine3f& trans);
      void findTransformation(const PointCloud& tar_pc, const PointCloud& src_pc, Eigen::Affine3f& trans);
      void findTransformation(HostPointCloud& tar_pc, HostPointCloud& src_pc, Eigen::Affine3f& trans);
      inline void fromVector2Eigen(float* , Eigen::Affine3f& t); 

      void getCloud(HostPointCloud& cloud_arg);
      void setSourceCloud(const PointCloud& cloud_arg); 
      void setSourceCloud(HostPointCloud& cloud_arg);

      void setTargetCloud(const PointCloud& cloud_arg);
      void setTargetCloud(HostPointCloud& cloud_arg); 

      void build();
      void transform(float3 rpy, float3 xyz);
      void transformDegree(float3 rpy, float3 xyz);
    private:
      void* transImpl; 
    public:
      // transformation index 
      thrust::host_vector<float> tx_h;
      thrust::host_vector<float> ty_h; 
      thrust::host_vector<float> tz_h;
      thrust::host_vector<float> r1_h; 
      thrust::host_vector<float> r2_h; 
      thrust::host_vector<float> r3_h; 
      thrust::host_vector<float> r4_h; 
      thrust::host_vector<float> r5_h; 
      thrust::host_vector<float> r6_h; 
      thrust::host_vector<float> r7_h; 
      thrust::host_vector<float> r8_h; 
      thrust::host_vector<float> r9_h;
    };
  }
}


#endif
