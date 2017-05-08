## Find CUDA ## 
find_package(CUDA QUIET REQUIRED)
# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20)

## other config ## 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CUDA_SRCS cuda_wrapper.cu cuda_wrapper.hpp cuda_sqrt.hpp cuda_sqrt.cu test.hpp test.cu main.cpp )
cuda_add_executable(main ${CUDA_SRCS})
# target_link_libraries(main ${PCL_LIBS})
