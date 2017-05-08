/*
 * Try to manipulate point cloud in device space 
 * David Z, Apr. 28
 *
 * */

#include <iostream>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree.h>

#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/common/time.h>
#include <pcl/cuda/cutil_math.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include "pcl/visualization/cloud_viewer.h"

#include "data_source.hpp"
#include "trans.h"

using namespace pcl::gpu;
using namespace std;

// test transformation function in the device space , 
// and return distance using octree search
void test1();

// generate point cloud randomly
void generatePC(int n, pcl::gpu::Octree::PointCloud&, pcl::PointCloud<pcl::PointXYZ>::Ptr&);

// find transformation
void test2();

// test example 
void test3();

int main(int argc, char* argv[])
{
    // test1();
    // test2();
    test3();
    return 0;
}

void print_f3(float3 rpy, float3 xyz)
{
  printf("rpy: %f, %f, %f, xyz: %f, %f, %f\n", R2D(rpy.x), R2D(rpy.y), R2D(rpy.z), xyz.x, xyz.y, xyz.z);
}

void showCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& pc)
{
  // visualization 
  pcl::visualization::CloudViewer viewer("MesaSR PCL Viewer"); 
  while(!viewer.wasStopped())
  {
    viewer.showCloud(pc);
    usleep(30000); // sleep 30 ms 
  }
}

void showCloud2(string name1, string name2, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc1, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc2)
{
  // visualization 
  // pcl::visualization::CloudViewer viewer("MesaSR PCL Viewer"); 
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));
  
  // showCloud(show_PC); PCLVisualizer
  viewer->addPointCloud<pcl::PointXYZ>(pc1, name1.c_str());
  viewer->addPointCloud<pcl::PointXYZ>(pc2, name2.c_str());
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
}



// test example 
void test3()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_PC(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tar_PC(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tar2_PC(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr save_PC(new pcl::PointCloud<pcl::PointXYZ>);

  CTrans gpu_trans; 
  Eigen::Affine3f t;
  Eigen::Affine3f t_mat; 

  // 1 bunny sample
  pcl::io::loadPCDFile("bunny.pcd", *src_PC); 
  float3 rpy = make_float3(8, 0, 0); 
  float3 xyz = make_float3(0.07, 0.08, 0);
  pcl::getTransformation( xyz.x, xyz.y, xyz.z, D2R(rpy.x), D2R(rpy.y), D2R(rpy.z), t_mat); 
  cout<<"host mat: "<<endl<<t_mat.matrix()<<endl;
  pcl::transformPointCloud(*src_PC, *tar_PC, t_mat);
  
  /*
  // *save_PC = *src_PC + *tar_PC; 
  pcl::io::savePCDFile("show/bunny_before_align_src.pcd", *src_PC);
  pcl::io::savePCDFile("show/bunny_before_align_tar.pcd", *tar_PC);

  // showCloud2("src_bunny", "tar_bunny", src_PC, tar_PC);
  // 
  {
    pcl::ScopeTime time("series time cost for bunny.pcd ");
    gpu_trans.findTransformationHost(*tar_PC, *src_PC, t); 
  }
  
  // pcl::transformPointCloud(*src_PC, *tar2_PC, t);
  // showCloud2("src_bunny", "tar_bunny", tar2_PC, tar_PC); 
  {
    pcl::ScopeTime time("gpu time cost for bunny.pcd"); 
    gpu_trans.findTransformation(*tar_PC, *src_PC, t);
  }

  pcl::transformPointCloud(*src_PC, *tar2_PC, t);
  // *save_PC = *tar2_PC + *tar_PC;
  pcl::io::savePCDFile("show/bunny_after_align_tar.pcd", *tar2_PC);
  // showCloud2("src_bunny", "tar_bunny", tar2_PC, tar_PC); 
  
  // 2. curve3d 
  pcl::io::loadPCDFile("curve3d.pcd", *src_PC); 
  rpy = make_float3(0, 8, 8); 
  xyz = make_float3(0.07, 0.08, 0.08);
  pcl::getTransformation( xyz.x, xyz.y, xyz.z, D2R(rpy.x), D2R(rpy.y), D2R(rpy.z), t_mat); 
  cout<<"host mat: "<<endl<<t_mat.matrix()<<endl;
  pcl::transformPointCloud(*src_PC, *tar_PC, t_mat);  
  // *save_PC = *src_PC + *tar_PC; 
  pcl::io::savePCDFile("show/curve3d_before_align_src.pcd", *src_PC);
  pcl::io::savePCDFile("show/curve3d_before_align_tar.pcd", *tar_PC);
  {
    pcl::ScopeTime time("series time cost for curve3d.pcd ");
    gpu_trans.findTransformationHost(*tar_PC, *src_PC, t); 
  } 
  {
    pcl::ScopeTime time("gpu time cost for curve3d.pcd"); 
    gpu_trans.findTransformation(*tar_PC, *src_PC, t);
  }
  pcl::transformPointCloud(*src_PC, *tar2_PC, t);
  // *save_PC = *tar2_PC + *tar_PC;
  pcl::io::savePCDFile("show/curve3d_after_align_tar.pcd", *tar2_PC);
*/
  // 3. wolf
  pcl::io::loadPCDFile("wolf.pcd", *src_PC); 
  for(int i=0; i<src_PC->points.size(); i++)
  {
    pcl::PointXYZ& pt = src_PC->points[i]; 
    pt.x*=0.01; pt.y*=0.01; pt.z*=0.01;
  }
  rpy = make_float3(0.1, 0, 0); 
  xyz = make_float3(0.042, 0.01, 0.08);
  pcl::getTransformation( xyz.x, xyz.y, xyz.z, D2R(rpy.x), D2R(rpy.y), D2R(rpy.z), t_mat); 
  cout<<"host mat: "<<endl<<t_mat.matrix()<<endl;
  pcl::transformPointCloud(*src_PC, *tar_PC, t_mat);  
  // *save_PC = *src_PC + *tar_PC; 
  pcl::io::savePCDFile("show/wolf_before_align_src.pcd", *src_PC);
  pcl::io::savePCDFile("show/wolf_before_align_tar.pcd", *tar_PC);
  // showCloud2("src", "tar", src_PC, tar_PC);
  {
    // pcl::ScopeTime time("series time cost for wolf.pcd ");
    // gpu_trans.findTransformationHost(*tar_PC, *src_PC, t); 
  } 
  {
    pcl::ScopeTime time("gpu time cost for wolf.pcd"); 
    gpu_trans.findTransformation(*tar_PC, *src_PC, t);
  }
  cout<<"gpu mat: "<<endl<<t.matrix()<<endl;
  pcl::getTranslationAndEulerAngles (t, xyz.x, xyz.y, xyz.z, rpy.x, rpy.y, rpy.z);
  print_f3(rpy, xyz);
  pcl::transformPointCloud(*src_PC, *tar2_PC, t);
  // *save_PC = *tar2_PC + *tar_PC;
  pcl::io::savePCDFile("show/wolf_after_align_tar.pcd", *tar2_PC);
  
}

// find transformation
void test2()
{
  // randomly generate a point cloud
  pcl::gpu::Octree::PointCloud cloud_device; 
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_PC(new pcl::PointCloud<pcl::PointXYZ>);
  generatePC(3, cloud_device, src_PC);
  
  // transform the point cloud using the pcl::transformPointCloud
  // tar_PC = R*src_PC + T;
  pcl::PointCloud<pcl::PointXYZ>::Ptr tar_PC(new pcl::PointCloud<pcl::PointXYZ>); 
  Eigen::Affine3f t_mat; 
  float3 rpy = make_float3(0, 0, 0); 
  float3 xyz = make_float3(0.07, 0.08, 0);
  print_f3(rpy, xyz);

  pcl::getTransformation( xyz.x, xyz.y, xyz.z, D2R(rpy.x), D2R(rpy.y), D2R(rpy.z), t_mat); 
  cout<<"host mat: "<<endl<<t_mat.matrix()<<endl;
  pcl::transformPointCloud(*src_PC, *tar_PC, t_mat);
  
  // for debug 
  // pcl::io::savePCDFile("src.pcd", *src_PC); 
  // pcl::io::savePCDFile("tar.pcd", *tar_PC);
  
  // compute trans from the gpu 
  CTrans gpu_trans; 
  Eigen::Affine3f t;
  {
    pcl::ScopeTime time("series time cost: ");
    gpu_trans.findTransformationHost(*tar_PC, *src_PC, t); 
  }
  cout<<"gpu host mat: "<<endl<<t.matrix()<<endl; 

  /*
  // from mat to xyz rpy 
  pcl::getTranslationAndEulerAngles (t, xyz.x, xyz.y, xyz.z, rpy.x, rpy.y, rpy.z);
  print_f3(rpy, xyz);
  
  {
    pcl::ScopeTime time("gpu time cost: ");
    gpu_trans.findTransformation(*tar_PC, *src_PC, t); 
  }
  cout<<"gpu mat: "<<endl<<t.matrix()<<endl; 
    
  pcl::getTranslationAndEulerAngles (t, xyz.x, xyz.y, xyz.z, rpy.x, rpy.y, rpy.z);
  print_f3(rpy, xyz);
*/
  return; 
}

// test transformation 
void test1()
{
  // randomly generate a point cloud
  pcl::gpu::Octree::PointCloud cloud_device; 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_host(new pcl::PointCloud<pcl::PointXYZ>);
  generatePC(100, cloud_device, cloud_host);

  // transform the point cloud using the pcl::transformPointCloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans1(new pcl::PointCloud<pcl::PointXYZ>); 
  Eigen::Affine3f t_mat; 
  float3 rpy = make_float3(10, 20, 30); 
  float3 xyz = make_float3(1, 2, 3);
  pcl::getTransformation( xyz.x, xyz.y, xyz.z, D2R(rpy.x), D2R(rpy.y), D2R(rpy.z), t_mat); 
  cout<<"host mat: "<<endl<<t_mat.matrix()<<endl;
  pcl::transformPointCloud(*cloud_host, *cloud_trans1, t_mat);

  // cout<<"test_gpu_point.cpp: host_cloud1 has points: "<<cloud_host->points.size()<<endl;
  // transform the point cloud 
  // pcl::device::TransformImpl trans_impl; 
  pcl::gpu::CTrans trans;
  // trans.setCloud(cloud_device);
  // trans.build(); 
  // cout<<"test_gpu_point.cpp: after build, before transform"<<endl;
  trans.transformDegree(rpy, xyz);   
  // cout<<"test_gpu_point.cpp: after transform before getCloud"<<endl;

  // get the point cloud after transformation 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_host2(new pcl::PointCloud<pcl::PointXYZ>);	
  trans.getCloud(*cloud_host2); 
  // cout<<"test_gpu_point.cpp: after getCloud host_cloud2 has: "<<cloud_host2->points.size()<<endl;

  if(cloud_trans1->points.size() != cloud_host2->points.size())
  {
    cout<<"test_gpu_point.cpp: error size not equal, first: "<<cloud_trans1->points.size()<<" "
      <<"second: "<<cloud_host2->points.size()<<endl;
    return ;
  }

  // check the result 
  double small_thre = 1e-3;
  for(int i=0; i<cloud_host->points.size(); i++)
  {
    pcl::PointXYZ& pt1 = cloud_trans1->points[i]; 
    pcl::PointXYZ& pt2 = cloud_host2->points[i]; 
    if(fabs(pt2.x - pt1.x)> small_thre || fabs(pt2.y - pt1.y) > small_thre || \
        fabs(pt2.z - pt1.z) > small_thre)
    {
      cout<<"test_gpu_point.cpp: error displacement at point : "<<i<<endl;
      cout<<"p1: "<<pt1.x<<" "<<pt1.y<<" "<<pt1.z<<" p2: "<<pt2.x<<" "<<pt2.y<<" "<<pt2.z<<endl;
    }else
    {
      cout<<"test_gpu_point.cpp: succeed dis: pt1: "<<pt1.x<<" "<<pt1.y<<" "<<pt1.z<<" pt2: "<<pt2.x<<" "<<pt2.y<<" "<<pt2.z<<endl;
    }
  }
}


void generatePC( int n, pcl::gpu::Octree::PointCloud& cloud_device, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_host)
{
  DataGenerator data;
  data.data_size = n;// 871000;
  // data.tests_num = 10000;    
  data.cube_size = 256.f; //1024.f;
  // data.max_radius    = data.cube_size/30.f;
  // data.shared_radius = data.cube_size/30.f;
  // data.printParams();

  // generate
  data();

  //prepare device cloud
  // pcl::gpu::Octree::PointCloud cloud_device;
  cloud_device.upload(data.points);

  //prepare host cloud
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_host(new pcl::PointCloud<pcl::PointXYZ>);	
  cloud_host->width = data.points.size();
  cloud_host->height = 1;
  cloud_host->points.resize (cloud_host->width * cloud_host->height);    
  std::transform(data.points.begin(), data.points.end(), cloud_host->points.begin(), DataGenerator::ConvPoint<pcl::PointXYZ>());
}

