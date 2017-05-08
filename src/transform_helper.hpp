/*
 * Function to transform a point in the device space 
 * Apr. 28 David Z 
 *
 * */

#ifndef TRANSFORM_HELPER_GPU
#define TRANSFORM_HELPER_GPU

#include <thrust/device_ptr.h>

namespace pcl
{
  namespace device
  {
    struct TransformHelper
    {
      float3 r123_;
      float3 r456_;
      float3 r789_;
      float3 xyz_;
      __device__ __host__ __forceinline__ TransformHelper(float3 r123, float3 r456, float3 r789, float3 xyz) : r123_(r123), r456_(r456), r789_(r789), xyz_(xyz){} 
  
      __device__ __host__ __forceinline__ TransformHelper() : r123_(), r456_(), r789_(), xyz_(){} 

      __device__ __host__ __forceinline__ float3 operator()(const float3 & p) const 
      {
        // float3 p_rt; 
        // p_rt.x = p.x + xyz_.x; p_rt.y = p.y + xyz_.y; p_rt.z = p.z + xyz_.z; 
        
        float r[9]; 
        r[0] = r123_.x; r[1] = r123_.y; r[2] = r123_.z; 
        r[3] = r456_.x; r[4] = r456_.y; r[5] = r456_.z; 
        r[6] = r789_.x; r[7] = r789_.y; r[8] = r789_.z; 
   
        float3 p_rt; 
        float x = p.x; float y = p.y; float z = p.z;
        p_rt.x = xyz_.x + x*r[0] + y*r[1] + z*r[2]; 
        p_rt.y = xyz_.y + x*r[3] + y*r[4] + z*r[5]; 
        p_rt.z = xyz_.z + x*r[6] + y*r[7] + z*r[8];
        
        return p_rt;
      }
      
      __device__ __host__ __forceinline__ float3 operator()(const float4& p) const 
      {
        return (*this)(make_float3(p.x, p.y, p.z));
      }

      __device__ __host__ __forceinline__ thrust::tuple<float, float, float> operator()(const thrust::tuple<float, float, float>& p) const
      {
        float x = p.get<0>(); float y = p.get<1>(); float z = p.get<2>(); 

        float3 p1 = make_float3(x,y,z); 
        float3 p2 = (*this)(p1);
        thrust::tuple<float, float, float> ret; 
        // ret.get<0>() = xyz_.x + x*r[0] + y*r[1] + z*r[2]; 
        // ret.get<1>() = xyz_.y + x*r[3] + y*r[4] + z*r[5]; 
        // ret.get<2>() = xyz_.z + x*r[6] + y*r[7] + z*r[8];
        ret.get<0>() = p2.x;
        ret.get<1>() = p2.y; 
        ret.get<2>() = p2.z;
          
        return ret; 
      }
    };
  
  }

}



#endif
