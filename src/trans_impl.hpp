/*
 * Interfaces that between host and device 
 * Apr. 28, 2015 David Z 
 *
 * */

#ifndef TRANS_IMPL_HPP
#define TRANS_IMPL_HPP

#include <vector>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/octree/device_format.hpp>
#include <pcl/gpu/utils/safe_call.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/host_vector.h>

namespace pcl
{
  namespace device
  {
      static const int DIS_NUM = 4; //8; //8; // 16; // number of dis steps for x,y,z
      static const int DIS_MAX = 8; // max distance for x,y,z, [-, ] cm
      // static const int DIS_STP = 2*DIS_MAX/(DIS_NUM-1); // distance step 
      static const int DIS_STP = 2*DIS_MAX/DIS_NUM; // distance step 
      static const int BLOCK_N = DIS_NUM*DIS_NUM*DIS_NUM; // initialize block num 

      static const int ANGLE_NUM = DIS_NUM; //8; //8 ;// 16; // number of angle steps for roll, pitch, yaw 
      static const int ANGLE_MAX = 8; // max angle for r,p,y, [-, ] degree 
      // static const int ANGLE_STP = 2*ANGLE_MAX/(ANGLE_NUM-1); 
      static const int ANGLE_STP = 2*ANGLE_MAX/ANGLE_NUM; 
      static const int GRID_N = ANGLE_NUM*ANGLE_NUM*ANGLE_NUM;
      
      // static const int TOTAL_NUM = DIS_NUM*DIS_NUM*DIS_NUM*ANGLE_NUM*ANGLE_NUM*ANGLE_NUM;
      static const int TOTAL_NUM = BLOCK_N*GRID_N;

    // using pcl::gpu::Octree::PointCloud;

    // find the transform between two point clouds 
    class TransformImpl 
    {
      public:
        enum
        {
          MAX_NUM_POINTS = 32768, //307200, // Kinect points 
          MIN_DIS_THRE = 100 //200 // 100  // Minimum distance mm,  * 0.001
        };
        typedef float4 PointType;
        typedef DeviceArray<PointType> PointArray;

        typedef PointArray PointCloud;
        typedef PointArray Queries;
      
        TransformImpl(){ init(); }
        ~TransformImpl(){} 
        
        static void get_gpu_arch_compiled_for(int& bin, int& ptx);

        void setSourceCloud(const PointCloud& input_points);   
        void setTargetCloud(const PointCloud& input_points);   

        void findTrans(thrust::host_vector<float>&);
        void init(); // init memory
        void build();
        void transform(float3, float3, float3, float3); 
        void internalDownload();
      
        // storage for the transformation
        thrust::device_vector<float> tx;
        thrust::device_vector<float> ty; 
        thrust::device_vector<float> tz;
        thrust::device_vector<float> r1; 
        thrust::device_vector<float> r2; 
        thrust::device_vector<float> r3; 
        thrust::device_vector<float> r4; 
        thrust::device_vector<float> r5; 
        thrust::device_vector<float> r6; 
        thrust::device_vector<float> r7; 
        thrust::device_vector<float> r8; 
        thrust::device_vector<float> r9;
        
        thrust::device_vector<float> se; // square error
        // thrust::device_vector<float> result; // result of trans

        //storage
        DeviceArray2D<int> storage;         

        // PointCloud src_points;
        // PointCloud tar_points;
        DeviceArray2D<float> src_points_xyz;
        DeviceArray2D<float> tar_points_xyz;

        struct Host_storage
        {
          std::vector<float> points_xyz; 
          int points_step; // points number
          bool downloaded_; // whether the data has been downloaded_;
          unsigned int points_num; 
        }host_data;    
        
    };
  }
}
#endif
