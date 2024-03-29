cmake_minimum_required(VERSION 3.20.0)
project(ch07 VERSION 0.1.0 LANGUAGES  CUDA CXX C)
find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 11)

 

add_executable(ch07 main.cu)
target_link_libraries(ch07 PRIVATE CUDA::cudart)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ch07 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(ch07 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

target_link_libraries(ch07 PRIVATE nlohmann_json::nlohmann_json)

FILE(COPY conv.json DESTINATION "${CMAKE_BINARY_DIR}")