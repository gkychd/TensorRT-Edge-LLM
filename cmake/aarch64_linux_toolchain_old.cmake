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

# ========== AArch64 (ARM64) 交叉编译工具链配置 ==========
# 用途：在 x86_64 主机上交叉编译适用于 ARM64 设备的程序

# 设置目标系统为 Linux
set(CMAKE_SYSTEM_NAME Linux)
# 设置目标处理器架构为 aarch64 (ARM64)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 指定交叉编译器路径
# aarch64-linux-gnu-gcc/g++ 是用于 ARM64 目标的 GCC 交叉编译器
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

# 设置编译器目标为 aarch64-linux-gnu
# 这告诉编译器生成 ARM64 架构的代码
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

# ========== CUDA 交叉编译配置 ==========

# 指定 CUDA 编译器 (nvcc) 路径
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# 设置 nvcc 的主机编译器为前面定义的交叉编译器
set(CMAKE_CUDA_HOST_COMPILER
    ${CMAKE_CXX_COMPILER}
    CACHE STRING "" FORCE)
# 强制使用指定的 CUDA 编译器
set(CMAKE_CUDA_COMPILER_FORCED TRUE)
# CUDA 编译标志：生成位置无关代码 (-fPIC)
set(CMAKE_CUDA_FLAGS
    " -Xcompiler=\"-fPIC \""
    CACHE STRING "" FORCE)

# ========== 嵌入式目标平台配置 ==========
# 根据 EMBEDDED_TARGET 设置不同的 CUDA 版本和 GPU 架构

# 宏：如果变量未定义，则设置为默认值
# 作用：提供CMake的默认值，同时允许用户通过-D参数覆盖
macro(set_ifndef var val)
  if(NOT DEFINED ${var})
    set(${var} ${val})
  endif()
  message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# 警告：避免 CUDA_VERSION 变量与 CUDA 宏产生歧义
if(DEFINED CUDA_VERSION)
  message(
    FATAL_ERROR
      "CUDA_VERSION can cause ambiguity with CUDA Macros. Please use -DCUDA_CTK_VERSION to specify the CUDA Toolkit version."
  )
endif()

# ========== 根据目标设备配置 ==========

# auto-thor: 通用的 Thor 设备（服务器级 ARM）
# - CUDA >= 13.0: 使用 SM 110 (Blackwell)
# - CUDA < 13.0: 使用 SM 101 (Hopper)
if("${EMBEDDED_TARGET}" STREQUAL "auto-thor")
  set_ifndef(CUDA_CTK_VERSION 13.2)
  if(CUDA_CTK_VERSION VERSION_LESS 13.0)
    set(CMAKE_CUDA_ARCHITECTURES 101)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 110)
  endif()
  set(CUDA_DIR
      /usr/local/cuda/targets/aarch64-linux
      CACHE STRING "CUDA toolkit dir")
  set(CUDA_TARGET_DIR
      /usr/local/cuda/thor/targets/aarch64-linux
      CACHE STRING "CUDA toolkit target dir")
  message(STATUS "Using CUDA toolkit dir: ${CUDA_DIR}")

# jetson-thor: Jetson Thor 设备（SBSA 架构）
# - 固定使用 SM 110 (Blackwell)
# - CUDA 路径使用 sbsa-linux 目标
elseif("${EMBEDDED_TARGET}" STREQUAL "jetson-thor")
  set_ifndef(CUDA_CTK_VERSION 13.0)
  set(CMAKE_CUDA_ARCHITECTURES 110)
  set(CUDA_DIR
      /usr/local/cuda/targets/sbsa-linux
      CACHE STRING "CUDA toolkit dir")

# jetson-orin: Jetson Orin 设备
# - 使用 SM 87
# - CUDA 路径使用 aarch64-linux 目标
elseif("${EMBEDDED_TARGET}" STREQUAL "jetson-orin")
  set_ifndef(CUDA_CTK_VERSION 12.6)
  set(CMAKE_CUDA_ARCHITECTURES 87)
  set(CUDA_DIR
      /usr/local/cuda/targets/aarch64-linux
      CACHE STRING "CUDA toolkit dir")

# gb10: GB10 设备
# - 使用 SM 121
# - 支持多 CUDA 目标路径
elseif("${EMBEDDED_TARGET}" STREQUAL "gb10")
  set_ifndef(CUDA_CTK_VERSION 13.0)
  set(CMAKE_CUDA_ARCHITECTURES 121)
  set(CUDA_DIR
      /usr/local/cuda/targets/aarch64-linux
      CACHE STRING "CUDA toolkit dir")
  set(CUDA_TARGET_DIR
      /usr/local/cuda/n1/targets/aarch64-linux
      CACHE STRING "CUDA toolkit target dir")
  message(STATUS "Using CUDA toolkit dir: ${CUDA_DIR}")
endif()

# ========== CMake 搜索路径配置 ==========
# 告诉 CMake 在交叉编译时如何搜索程序、库和头文件

# NEVER: 不在目标系统上搜索程序（使用主机上的程序）
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# ONLY: 只在交叉编译工具链指定的目录中搜索库
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# ONLY: 只在交叉编译工具链指定的目录中搜索头文件
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# 设置变量，标记正在进行 AArch64 交叉编译
# 供主 CMakeLists.txt 中的 add_cross_build_link_options 宏使用
set(AARCH64_BUILD TRUE)
