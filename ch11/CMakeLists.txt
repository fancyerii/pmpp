cmake_minimum_required(VERSION 3.20.0)
project(ch11 VERSION 0.1.0 LANGUAGES  CUDA CXX C)
find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 11)

 

add_executable(ch11 main.cu)
target_link_libraries(ch11 PRIVATE CUDA::cudart)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ch11 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(ch11 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

