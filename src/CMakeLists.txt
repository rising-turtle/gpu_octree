
project(Octree_GPU)

  if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
    endif(COMMAND cmake_policy)


## Find pcl ## 
set(PCL_DIR "/home/davidz/work/3rdlibs/pcl-git/build")
find_package(PCL 1.7 REQUIRED) #COMPONENTS common io gpu_octree cuda_common cuda_io gpu_containers gpu_utils search registration )
message("STATUS PCL_INCLUDE_DIRS : ${PCL_INCLUDE_DIRS}")
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
set(PCL_LIBS pcl_octree pcl_gpu_octree pcl_gpu_containers)

## Find CUDA ## 
find_package(CUDA QUIET REQUIRED)
# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20)

## other config ## 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

## build target ## 
set(SRCS test_knn_search.cpp )
add_executable(test_gpu_knn ${SRCS}) 
target_link_libraries(test_gpu_knn ${PCL_LIBS})

set(CUDA_SRCS transform_helper.hpp trans_impl.hpp trans_impl.cu test_gpu_point.cpp trans.cpp)
cuda_add_executable(cuda_transform ${CUDA_SRCS})
# target_link_libraries(cuda_transform ${PCL_LIBS})
target_link_libraries(cuda_transform ${PCL_LIBRARIES})

 ##
 # set(the_test_target test_gpu_octree)
 # FILE(GLOB test_src *.cpp *.hpp)
 # PCL_ADD_TEST(a_gpu_octree_test ${the_test_target} FILES ${test_src} LINK_WITH pcl_octree pcl_gpu_containers pcl_gpu_octree)    
 # add_dependencies(${the_test_target} pcl_octree pcl_gpu_containes pcl_gpu_octree)
	

