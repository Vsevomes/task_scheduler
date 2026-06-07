cmake_minimum_required(VERSION 3.18)
project(task_scheduler LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(STARPU starpu-1.4)
if(NOT STARPU_FOUND)
  pkg_check_modules(STARPU REQUIRED starpu-1.3)
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/common
)

set(COMMON_SOURCES
  src/common/metrics.cpp
  src/common/starpu_runtime.cpp
)

set(COMMON_LIBS
  ${STARPU_LIBRARIES}
  CUDA::cudart
)

function(add_benchmark target)
  add_executable(${target} ${ARGN} ${COMMON_SOURCES})
  target_include_directories(${target} PRIVATE ${STARPU_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/common)
  target_link_directories(${target} PRIVATE ${STARPU_LIBRARY_DIRS})
  target_link_libraries(${target} PRIVATE ${COMMON_LIBS})
  target_compile_options(${target} PRIVATE ${STARPU_CFLAGS})
  target_compile_definitions(${target} PRIVATE STARPU_USE_CUDA)
endfunction()

add_benchmark(bench_matmul
  src/matmul/bench_matmul.cpp
  src/matmul/matmul_kernels.cu
)

add_benchmark(bench_independent
  src/independent/bench_independent.cpp
  src/independent/independent_kernels.cu
)

add_benchmark(bench_heterogeneous
  src/heterogeneous/bench_heterogeneous.cpp
  src/heterogeneous/heterogeneous_kernels.cu
)

add_benchmark(bench_image
  src/image/bench_image.cpp
  src/image/image_kernels.cu
)

add_benchmark(bench_overhead
  src/overhead/bench_overhead.cpp
  src/overhead/overhead_kernels.cu
)
