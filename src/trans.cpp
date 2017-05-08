/*
 * handle pcl::PointXYZ part, as a middleware to trans_impl, which is operated in GPU space 
 * Apr. 29 2015, David Z
 *
 * */

#include "trans.h"

#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/utils/timers_cuda.hpp>
#include <pcl/gpu/utils/safe_call.hpp>
#include <pcl/common/transforms.h>
#include <limits>
#include "trans_impl.hpp"
#include "transform_helper.hpp"

#include <iostream>
#include <fstream>

using namespace pcl::device; 
using namespace std;

namespace pcl{
namespace gpu{

CTrans::CTrans() : transImpl(0)
  {
    int device;
    cudaSafeCall( cudaGetDevice( &device ) );

    cudaDeviceProp prop;
    cudaSafeCall( cudaGetDeviceProperties( &prop, device) );

    if (prop.major < 2)
      pcl::gpu::error("This code requires devices with compute capability >= 2.0", __FILE__, __LINE__);

    int bin, ptx;
    TransformImpl::get_gpu_arch_compiled_for(bin, ptx);

    if (bin < 20 && ptx < 20)
      pcl::gpu::error("This must be compiled for compute capability >= 2.0", __FILE__, __LINE__);    

  transImpl = new TransformImpl; 
  initTrans();
}

CTrans::~CTrans()
{
 if(transImpl)  
   delete static_cast<TransformImpl*>(transImpl);
}

void CTrans::initTrans()
{
  /*
  thrust::host_vector<float> tx_h(TOTAL_NUM);
  thrust::host_vector<float> ty_h(TOTAL_NUM); 
  thrust::host_vector<float> tz_h(TOTAL_NUM);
  thrust::host_vector<float> r1_h(TOTAL_NUM); 
  thrust::host_vector<float> r2_h(TOTAL_NUM); 
  thrust::host_vector<float> r3_h(TOTAL_NUM); 
  thrust::host_vector<float> r4_h(TOTAL_NUM); 
  thrust::host_vector<float> r5_h(TOTAL_NUM); 
  thrust::host_vector<float> r6_h(TOTAL_NUM); 
  thrust::host_vector<float> r7_h(TOTAL_NUM); 
  thrust::host_vector<float> r8_h(TOTAL_NUM); 
  thrust::host_vector<float> r9_h(TOTAL_NUM);
  thrust::host_vector<float> err(TOTAL_NUM, 0);  
  */

  tx_h = thrust::host_vector<float>(TOTAL_NUM);
  ty_h = thrust::host_vector<float>(TOTAL_NUM); 
  tz_h = thrust::host_vector<float>(TOTAL_NUM);
  r1_h = thrust::host_vector<float>(TOTAL_NUM); 
  r2_h = thrust::host_vector<float>(TOTAL_NUM); 
  r3_h = thrust::host_vector<float>(TOTAL_NUM); 
  r4_h = thrust::host_vector<float>(TOTAL_NUM); 
  r5_h = thrust::host_vector<float>(TOTAL_NUM); 
  r6_h = thrust::host_vector<float>(TOTAL_NUM); 
  r7_h = thrust::host_vector<float>(TOTAL_NUM); 
  r8_h = thrust::host_vector<float>(TOTAL_NUM); 
  r9_h = thrust::host_vector<float>(TOTAL_NUM);
  thrust::host_vector<float> err(TOTAL_NUM, 0);  

  // distribute x,y,z, roll, pitch, yaw
  float x, y, z, r, p, yaw;
  Eigen::Affine3f trans;
  float cm2m = 0.01;
  unsigned int i=0;
  for(int ix=0; ix<DIS_NUM; ix++)
  {
    x = (-DIS_MAX + (ix+1)*DIS_STP)*cm2m;
    for(int iy=0; iy<DIS_NUM; iy++)
    {
      y = (-DIS_MAX + (iy+1)*DIS_STP)*cm2m; 
      for(int iz=0; iz<DIS_NUM; iz++)
      {
        z = (-DIS_MAX + (iz+1)*DIS_STP)*cm2m; 
        for(int iroll=0; iroll<ANGLE_NUM; iroll++)
        {
          r = D2R(-ANGLE_MAX + iroll*ANGLE_STP);  
          for(int ipitch=0; ipitch<ANGLE_NUM; ipitch++)
          {
            p = D2R(-ANGLE_MAX + (ipitch+1)*ANGLE_STP); 
            for(int iyaw=0; iyaw<ANGLE_NUM; iyaw++)
            {
              yaw = D2R(-ANGLE_MAX + (iyaw+1)*ANGLE_STP);
              
              // transform rpy to rotation matrix 
              pcl::getTransformation(0,0,0,r,p,yaw,trans); 
              
              // assignment 
              tx_h[i] = x;  ty_h[i] = y;  tz_h[i] = z; 
              r1_h[i] = trans(0,0); r2_h[i] = trans(0,1); r3_h[i] = trans(0,2);
              r4_h[i] = trans(1,0); r5_h[i] = trans(1,1); r6_h[i] = trans(1,2);
              r7_h[i] = trans(2,0); r8_h[i] = trans(2,1); r9_h[i] = trans(2,2); 
              // index 
              ++i;
            }
          }
        }
      }
    }
  }
  
  // copy from host to device 
  static_cast<TransformImpl*>(transImpl)->tx = tx_h ;
  static_cast<TransformImpl*>(transImpl)->ty = ty_h ;
  static_cast<TransformImpl*>(transImpl)->tz = tz_h ;
  static_cast<TransformImpl*>(transImpl)->r1 = r1_h ;
  static_cast<TransformImpl*>(transImpl)->r2 = r2_h ;
  static_cast<TransformImpl*>(transImpl)->r3 = r3_h ;
  static_cast<TransformImpl*>(transImpl)->r4 = r4_h ;
  static_cast<TransformImpl*>(transImpl)->r5 = r5_h ;
  static_cast<TransformImpl*>(transImpl)->r6 = r6_h ;
  static_cast<TransformImpl*>(transImpl)->r7 = r7_h ;
  static_cast<TransformImpl*>(transImpl)->r8 = r8_h ;
  static_cast<TransformImpl*>(transImpl)->r9 = r9_h ;
  static_cast<TransformImpl*>(transImpl)->se = err ;
}

float CTrans::err_match(HostPointCloud& tar_pc, HostPointCloud& src_pc, float3 xyz, float3 r123, float3 r456, float3 r789)
{
  TransformHelper trans_h(r123, r456, r789, xyz); 
  float3 p1, p2; 
  float dis_thre = SQ(pcl::device::TransformImpl::MIN_DIS_THRE * 0.001);
  int sum_num = 0; 
  float sum_err = 0; 
  for(int i=0; i<src_pc.points.size(); i++)
  {
    PointType& pt_src = src_pc.points[i]; 
    p1 = make_float3(pt_src.x, pt_src.y, pt_src.z); 
    p2 = trans_h(p1); 
    
    // find the minimum 
    float dis_min = std::numeric_limits<float>::max(); // dis_thre;
    float dis = 0;
    for(int j =0; j<tar_pc.points.size(); j++)
    {
      PointType& pt_tar = tar_pc.points[j]; 
      dis = SQ(p2.x - pt_tar.x) + SQ(p2.y - pt_tar.y) + SQ(p2.z - pt_tar.z); 
      if(dis < dis_min) dis_min = dis;
    }

    // if this is a pair 
    if(dis_min < dis_thre)
    {
      ++sum_num; 
      sum_err += dis_min;
    }
  }
  if(sum_num > 0)
  {
    if(sum_err < 1e-5)
      return std::numeric_limits<float>::max(); 
    else
      return (sum_num*1./sum_err);
  }
  return 0;
}

void CTrans::findTransformationHost(HostPointCloud& tar_pc, HostPointCloud& src_pc, Eigen::Affine3f& trans)
{
    int N = tx_h.size();
    thrust::host_vector<float> score(N, 0); 
    float3 xyz, r123, r456, r789; 
    for(int i=0; i<N ;i++)
    {
      xyz = make_float3(tx_h[i], ty_h[i], tz_h[i]); 
      r123 = make_float3(r1_h[i], r2_h[i], r3_h[i]); 
      r456 = make_float3(r4_h[i], r5_h[i], r6_h[i]); 
      r789 = make_float3(r7_h[i], r8_h[i], r9_h[i]); 
      
      score[i] = err_match(tar_pc, src_pc, xyz, r123, r456, r789); 
    } 

    // find the max score 
    thrust::host_vector<float>::iterator iter = thrust::max_element(score.begin(), score.end()); 
    unsigned int position = iter - score.begin(); 
    thrust::host_vector<float> result_h(12); 
    result_h[0] = tx_h[position]; result_h[1] = ty_h[position]; result_h[2] = tz_h[position]; 
    result_h[3] = r1_h[position]; result_h[4] = r2_h[position]; result_h[5] = r3_h[position];
    result_h[6] = r4_h[position]; result_h[7] = r5_h[position]; result_h[8] = r6_h[position];
    result_h[9] = r7_h[position]; result_h[10] = r8_h[position]; result_h[11] = r9_h[position];
    
    // 
    fromVector2Eigen(result_h.data(), trans);
    
    // for debug
    cout<<"trans.cpp: max score is: "<<*iter<<endl;
    // thrust::copy(score.begin(), score.end(), std::ostream_iterator<float>(std::cout, " ")); 
    // cout<<endl;
}

inline void CTrans::fromVector2Eigen(float* p, Eigen::Affine3f& t)
{
  float * pt = p; 
  t(0,3) = *pt;     t(1,3) = *(pt+1); t(2,3) = *(pt+2);  // xyz
  t(0,0) = *(pt+3); t(0,1) = *(pt+4); t(0,2) = *(pt+5);  // r123
  t(1,0) = *(pt+6); t(1,1) = *(pt+7); t(1,2) = *(pt+8);  // r456
  t(2,0) = *(pt+9); t(2,1) = *(pt+10); t(2,2) = *(pt+11); // r789
  return;
}

void CTrans::findTransformation(const PointCloud& tar_pc, const PointCloud& src_pc, Eigen::Affine3f& trans)
{
  setTargetCloud(tar_pc); 
  setSourceCloud(src_pc); 
  thrust::host_vector<float> tr(12);
  static_cast<TransformImpl*>(transImpl)->findTrans(tr);
  
  // from tr to Eigen::Affine3 
  fromVector2Eigen(tr.data(), trans); 
}


void CTrans::findTransformation(HostPointCloud& tar_pc, HostPointCloud& src_pc, Eigen::Affine3f& trans)
{
  setTargetCloud(tar_pc); 
  setSourceCloud(src_pc); 
  thrust::host_vector<float> tr(12); 
  static_cast<TransformImpl*>(transImpl)->findTrans(tr);

  // from tr to Eigen::Affine3 
  fromVector2Eigen(tr.data(), trans); 
}

void CTrans::setTargetCloud(const PointCloud& cloud_arg)
{
  const pcl::device::TransformImpl::PointCloud& cloud = (const pcl::device::TransformImpl::PointCloud&)(cloud_arg); 
  static_cast<TransformImpl*>(transImpl)->setTargetCloud(cloud);
}

void CTrans::setTargetCloud(HostPointCloud& cloud_arg)
{
  PointCloud cloud_device; 
  cloud_device.upload(cloud_arg.points);
  setTargetCloud(cloud_device);
}

void CTrans::setSourceCloud(const PointCloud& cloud_arg)
{
  const pcl::device::TransformImpl::PointCloud& cloud = (const pcl::device::TransformImpl::PointCloud&)(cloud_arg); 
  static_cast<TransformImpl*>(transImpl)->setSourceCloud(cloud);
}

void CTrans::setSourceCloud(HostPointCloud& cloud_arg)
{
  PointCloud cloud_device; 
  cloud_device.upload(cloud_arg.points);
  setSourceCloud(cloud_device);
}

void CTrans::build()
{
  static_cast<TransformImpl*>(transImpl)->build();    
}

void CTrans::transformDegree(float3 rpy, float3 xyz)
{
  float3 rpy_r = make_float3(D2R(rpy.x), D2R(rpy.y), D2R(rpy.z)); 
  transform(rpy_r, xyz);
}

void CTrans::transform(float3 rpy, float3 xyz)
{
  Eigen::Affine3f r;
  getTransformation(0,0,0, rpy.x, rpy.y, rpy.z, r);
  cout<<"trans.cpp: device trans: "<<endl<<r.matrix()<<endl;
  float3 r123 = make_float3(r(0,0), r(0,1), r(0,2)); 
  float3 r456 = make_float3(r(1,0), r(1,1), r(1,2));
  float3 r789 = make_float3(r(2,0), r(2,1), r(2,2));

  // cout<<r(0,0)<<" "<<r(0,1)<<" "<<r(0,2)<<" "<<r(1,1)<<" "<<r(1,2)<<" "<<r(2,2)<<endl;

  static_cast<TransformImpl*>(transImpl)->transform(r123, r456, r789, xyz);    
}

void CTrans::getCloud(HostPointCloud& cloud_arg)
{
  TransformImpl::Host_storage& host_data = static_cast<TransformImpl*>(transImpl)->host_data;
  if(!host_data.downloaded_)
     static_cast<TransformImpl*>(transImpl)->internalDownload(); 

  if(cloud_arg.points.size() != host_data.points_num) 
  {
    cloud_arg.points.resize(host_data.points_num); 
    cloud_arg.width = host_data.points_num; 
    cloud_arg.height = 1;
  } 
  for(int i=0; i<host_data.points_num; i++)
  {
      PointType& pt = cloud_arg.points[i]; 
      pt.x = host_data.points_xyz[i]; 
      pt.y = host_data.points_xyz[i + host_data.points_step]; 
      pt.z = host_data.points_xyz[i + host_data.points_step*2];
  }
}

}
}

/*
std::ostream& operator<<(std::ostream& out, Eigen::Affine3f& m)
{
  using std::endl;
  out<<endl
    <<m(0,0)<<" "<<m(0,1)<<" "<<m(0,2)<<endl
    <<m(1,0)<<" "<<m(1,1)<<" "<<m(1,2)<<endl
    <<m(2,0)<<" "<<m(2,1)<<" "<<m(2,2)<<endl;
  return out;
}
*/
