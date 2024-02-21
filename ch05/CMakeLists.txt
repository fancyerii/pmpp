cmake_minimum_required(VERSION 3.20.0)
project(ch05 VERSION 0.1.0 LANGUAGES  CUDA CXX C)
find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 11)

 

add_executable(ch05 main.cu)
target_link_libraries(ch05 PRIVATE CUDA::cudart)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ch05 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(ch05 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

option(CHECK_EQUAL "check two algos equals" OFF)

if(CHECK_EQUAL)
  add_definitions(-DCHECK_EQUAL)
endif()