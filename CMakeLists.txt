cmake_minimum_required(VERSION 3.8)

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-14" CACHE FILEPATH "Host compiler for NVCC" FORCE)
endif()

project(denoiser LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

option(BUILD_DEBUG "Build with debug settings" OFF)
set(BUILD_DEBUG ${BUILD_DEBUG} CACHE BOOL "Build debug" FORCE)

if(BUILD_DEBUG)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall -Wextra -Wpedantic")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")
endif()

set(APPLICATION_SOURCE
    src/denoiser.cu
    src/exr.cpp
)

find_package(OpenEXR REQUIRED)

add_executable(denoiser ${APPLICATION_SOURCE})
target_include_directories(denoiser PUBLIC include)
target_link_libraries(denoiser
    OpenEXR::OpenEXR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_custom_command(TARGET denoiser POST_BUILD
  BYPRODUCTS
    ${CMAKE_CURRENT_BINARY_DIR}/denoiser.cubin
    ${CMAKE_CURRENT_BINARY_DIR}/denoiser.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/denoiser.sass
  COMMAND ${CMAKE_COMMAND} -E env PATH=$ENV{PATH}
    cuobjdump --extract-elf all $<TARGET_FILE:denoiser>
  COMMAND ${CMAKE_COMMAND}
    -DCUBIN_PREFIX=denoiser
    -DCUBIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DCUBIN_OUTPUT=$<TARGET_FILE_DIR:denoiser>/denoiser.cubin
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/process_cubin.cmake
  COMMAND ${CMAKE_COMMAND} -E env PATH=$ENV{PATH}
    cuobjdump --dump-ptx $<TARGET_FILE:denoiser> > $<TARGET_FILE_DIR:denoiser>/denoiser.ptx
  COMMAND ${CMAKE_COMMAND} -E env PATH=$ENV{PATH}
    nvdisasm --print-line-info -hex $<TARGET_FILE_DIR:denoiser>/denoiser.cubin > $<TARGET_FILE_DIR:denoiser>/denoiser.sass
  COMMENT "Generating cubin, PTX and SASS dumps for denoiser")
