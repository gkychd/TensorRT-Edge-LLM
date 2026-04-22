# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# CMake 最低版本要求
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# 项目名称、版本以及支持的语言
# LANGUAGES CXX 表示使用 C++，CUDA 表示使用 CUDA
project(
  tensorrt_edgellm_sdk
  VERSION 0.6.1
  LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)

# C++ 编译标志：关闭deprecated警告、启用所有警告、将警告视为错误、忽略未使用参数警告
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wall -Werror -Wno-error=unused-parameter"
)
# CUDA 编译标志：同上，额外通过 -Xcompiler 传递标志给主机编译器
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations -Xcompiler=-Werror,-Wno-error=unused-parameter"
)
# 链接器标志：禁止未定义符号（需要在链接时解决）
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# 代码覆盖率选项（用于 SonarQube 分析）
# 可通过 -DENABLE_COVERAGE=ON 启用
option(ENABLE_COVERAGE "Enable gcov code coverage instrumentation" OFF)
include(${CMAKE_SOURCE_DIR}/cmake/CodeCoverage.cmake)

# 宏：如果变量未定义，则设置为默认值
# 作用：提供CMake的默认值，同时允许用户通过-D参数覆盖
macro(set_ifndef var val)
  if(NOT DEFINED ${var})
    set(${var} ${val})
  endif()
  message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# 宏：添加交叉编译时的链接选项
# 作用：为ARM64交叉编译添加CUDA库的搜索路径
macro(add_cross_build_link_options target)
  if(DEFINED AARCH64_BUILD)
    target_link_options(${target} PRIVATE "-L${CUDA_DIR}/lib")
    target_link_options(${target} PRIVATE "-L${CUDA_DIR}/lib/stubs")
  endif()
endmacro(add_cross_build_link_options)

# ========== CUDA 配置 ==========
# 警告：CUDA_VERSION 变量可能与CUDA宏产生歧义，请使用 -DCUDA_CTK_VERSION 指定版本
if(DEFINED CUDA_VERSION)
  message(
    FATAL_ERROR
      "CUDA_VERSION can cause ambiguity with CUDA Macros. Please use -DCUDA_CTK_VERSION to specify the CUDA Toolkit version."
  )
endif()
# 设置默认CUDA版本为12.8，默认安装路径为 /usr/local/cuda-${CUDA_CTK_VERSION}
set_ifndef(CUDA_CTK_VERSION 12.8)
set_ifndef(CUDA_DIR /usr/local/cuda-${CUDA_CTK_VERSION})

# 设置CUDA架构（非交叉编译时）
# 默认：Ampere(80), Ada(89), Hopper(90)
# CUDA >= 12.8 时额外添加 Lovelace(100) 和 Blackwell(120)
if(NOT DEFINED AARCH64_BUILD)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89)
  if(CUDA_CTK_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 100 120)
  endif()
endif()

# 下面写法的好处
# 灵活性：用户可以编译时通过 -DCUDA_DIR=... 指定CUDA安装路径
# 跨平台/跨环境：不同平台环境CUDA的安装位置不同，这种查找方式能自动适应
# 健壮性：PATH_SUFFIX 模式，无论CUDA安装在哪，只要结构标准，都能定位到include目录
# 查找 CUDA 运行时 API 头文件
find_path(
  CUDA_RUNTIME_API_INCLUDE_DIR cuda_runtime_api.h  #目标变量  查找目标
  HINTS ${CUDA_DIR}       #搜索提示路径，HINTS优先级很高，CMake优先在这里搜索
  PATH_SUFFIXES include)  #基础路径下需要追加的子路径

# 查找 CUDA 运行时库 (cudart)
find_library(
  CUDART_LIB cudart
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES lib lib64)

# 查找 CUDA 驱动库 (cuda)
find_library(
  CUDA_DRIVER_LIB cuda
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES lib lib64 lib/stubs)

# 查找 cuRAND kernel 头文件（CUDA随机数生成库）
find_path(
  CURAND_KERNEL_INCLUDE_DIR
  NAMES curand_kernel.h
  HINTS ${CUDA_DIR} ${CUDA_TARGET_DIR}
  PATH_SUFFIXES include)

# 合并 CUDA 头文件目录
set(CUDA_INCLUDE_DIR ${CUDA_RUNTIME_API_INCLUDE_DIR}
                     ${CURAND_KERNEL_INCLUDE_DIR})

# 禁用静态CUDA运行时（使用动态链接）
# 静态链接ON：将静态库libcudart_static.a链接进可执行文件中，找不到.a文件会报错
# 动态链接OFF：可执行文件只会包含对libcudart.so的引用，找不到.so文件不会报错，运行时加载即可
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# ========== NVTX 性能分析配置 ==========
# NVTX (NVIDIA Tools Extensions) 用于CUDA程序的性能分析和可视化
# 默认关闭，可通过 -DENABLE_NVTX_PROFILING=ON 启用
option(ENABLE_NVTX_PROFILING "Enable NVTX profiling support" OFF)

if(ENABLE_NVTX_PROFILING)
  # 使用捆绑的 NVTX3 headers (v3.4.0-c-cpp) 以确保跨平台一致性
  # 支持所有目标平台 (GPU/Thor/Orin) 和 CUDA 11/12/13
  set(NVTX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdParty/NVTX/include/nvtx3")

  if(EXISTS "${NVTX_INCLUDE_DIR}/nvtx3.hpp")
    message(STATUS "NVTX profiling ENABLED: ${NVTX_INCLUDE_DIR}")
    # 定义预处理器宏，用于条件编译
    add_compile_definitions(ENABLE_NVTX_PROFILING)
  else()
    message(
      FATAL_ERROR
        "NVTX3 headers not found at ${NVTX_INCLUDE_DIR}. "
        "Please ensure 3rdParty/NVTX is initialized: git submodule update --init 3rdParty/NVTX"
    )
  endif()
else()
  message(
    STATUS "NVTX profiling DISABLED (use -DENABLE_NVTX_PROFILING=ON to enable)")
  set(NVTX_INCLUDE_DIR "")
endif()

# ========== TensorRT 配置 ==========
# 必须指定 TRT_PACKAGE_DIR（通过 -DTRT_PACKAGE_DIR=<path>）
if(NOT DEFINED TRT_PACKAGE_DIR)
  message(
    FATAL_ERROR "Please specify the -DTRT_PACKAGE_DIR when invoking CMake.")
endif()

# 查找 TensorRT 主头文件
find_path(
  TRT_INCLUDE_DIR NvInfer.h
  HINTS ${TRT_PACKAGE_DIR}
  PATH_SUFFIXES include ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)

# 查找 ONNX Parser 头文件
find_path(ONNX_PARSER_INCLUDE_DIR NvOnnxParser.h HINTS ${TRT_INCLUDE_DIR})
if(NOT ONNX_PARSER_INCLUDE_DIR)
  message(
    FATAL_ERROR
      "NvOnnxParser.h not found in TensorRT headers, please specify the -DONNX_PARSER_INCLUDE_DIR when invoking CMake"
  )
endif()

# 查找 TensorRT 库文件
find_library(
  NVINFER_LIB nvinfer
  HINTS ${TRT_PACKAGE_DIR}
  PATH_SUFFIXES lib lib64 ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)

# 查找 ONNX Parser 库文件
find_library(
  NV_ONNX_PARSER_LIB nvonnxparser
  HINTS ${TRT_PACKAGE_DIR}
  PATH_SUFFIXES lib lib64 x86_64-linux-gnu)

# 将 TRT 和 CUDA 头文件作为系统包含项处理
# 作用：抑制来自第三方库的头文件警告（如 NvInferRuntime.h 中的 -Wunused-parameter）
# 普通包含 include_directories(...)
# 系统包含 include_directories(SYSTEM ...)
# CUDA 和 TensorRT 的官方头文件有时包含一些老旧的 C++ 写法，或者在开启高警告级别（如 -Wall -Wextra）时会触发大量无关紧要的警告。
# 使用 SYSTEM 可以让你的编译输出非常干净，只显示你自己代码的错误。
include_directories(SYSTEM ${CUDA_INCLUDE_DIR} ${TRT_INCLUDE_DIR})

# 公共包含目录列表
set(COMMON_INCLUDE_DIRS
    ${NVTX_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp
    ${CMAKE_SOURCE_DIR}/examples/multimodal
    ${CMAKE_SOURCE_DIR}/3rdParty/nlohmannJson/include
    ${CMAKE_SOURCE_DIR}/3rdParty/stb)

# 创建公共接口库，链接 TensorRT 和 CUDA 运行时
add_library(commonLibraryExt INTERFACE)
target_link_libraries(commonLibraryExt INTERFACE ${NVINFER_LIB} ${CUDART_LIB})

# ARM64 交叉编译的特殊处理：
# libnvinfer.so 依赖于 libnvdla_compilers，但该库在目标板上运行时才存在
# 使用 --unresolved-symbols=ignore-in-shared-libs 让链接器忽略未解析的符号
target_link_options(commonLibraryExt INTERFACE
                    "-Wl,--unresolved-symbols=ignore-in-shared-libs")

# ========== 构建子目录 ==========

# 构建核心库：attentionPlugin 动态库 + edgellmCore/edgellmTokenizer/edgellmBuilder 静态库
add_subdirectory(cpp)

# 构建 C++ 示例程序
add_subdirectory(examples)

# ========== 单元测试配置 ==========
# 可通过 -DBUILD_UNIT_TESTS=ON 启用
option(BUILD_UNIT_TESTS "Enable building unit tests" OFF)

if(BUILD_UNIT_TESTS)
  # 禁用 Google Mock（只使用 Google Test）
  set(BUILD_GMOCK
      OFF
      CACHE BOOL "Disable Google Mock")

  # 添加 Google Test 子目录
  add_subdirectory(${CMAKE_SOURCE_DIR}/3rdParty/googletest)

  # 查找所有单元测试源文件 (.cpp 和 .cu)
  file(GLOB_RECURSE UNIT_TESTS_SRCS ${CMAKE_SOURCE_DIR}/unittests/*.cpp
       ${CMAKE_SOURCE_DIR}/unittests/*.cu)

  # 创建测试可执行文件
  add_executable(unitTest ${UNIT_TESTS_SRCS})

  # 设置包含目录
  target_include_directories(unitTest PRIVATE ${COMMON_INCLUDE_DIRS}
                                              ${CMAKE_SOURCE_DIR})

  # 定义项目根路径宏，供测试访问资源文件
  target_compile_definitions(unitTest
                             PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

  # 链接测试所需的库
  target_link_libraries(unitTest PRIVATE gtest gtest_main edgellmCore
                                         ${CUDART_LIB} ${CUDA_DRIVER_LIB})

  # 添加交叉编译链接选项
  add_cross_build_link_options(unitTest)

  # 可选：链接预编译的 CuTe DSL FMHA 内核
  # 可通过 -DENABLE_CUTE_DSL_FMHA=ON 启用
  if(ENABLE_CUTE_DSL_FMHA)
    include(${CMAKE_SOURCE_DIR}/cmake/CuteDslFMHA.cmake)
    cute_dsl_fmha_setup(LINK_TARGETS unitTest)
  endif()
endif()
