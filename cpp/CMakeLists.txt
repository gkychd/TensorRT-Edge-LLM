# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

# ========== C++ 编译标志 ==========
# -Wfatal-errors: 遇到第一个错误就停止编译
# -Wall -Wextra: 启用所有常见和额外警告
# -Wno-unknown-pragmas: 忽略未知pragma警告
# -Wno-deprecated: 忽略deprecated声明警告
# -fPIC: 生成位置无关代码（用于共享库）
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -Wall -Wextra -Wno-unknown-pragmas -Wno-deprecated -fPIC"
)

# ========== CUDA 编译标志 ==========
# --expt-relaxed-constexpr: 允许在constexpr函数中使用非常量表达式
# -t 0: 使用所有可用CPU线程进行编译
# -fPIC: 生成位置无关代码
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -t 0 -fPIC")

# PTXAS 警告标志（PTX assembler）
# -warn-lmem-usage: 警告本地内存使用情况
# -warn-double-usage: 警告双精度使用
# -warn-spills: 警告寄存器溢出
set(CUDA_PTXAS_FLAGS "-warn-lmem-usage -warn-double-usage -warn-spills")

# Release 模式的 CUDA 优化标志
# -lineinfo: 生成行信息（用于调试和profiling）
# -res-usage: 显示资源使用情况
# -use_fast_math: 使用快速数学运算（牺牲精度换性能）
# -Xptxas: 传递标志给 PTX assembler
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -res-usage --use_fast_math -Xptxas='${CUDA_PTXAS_FLAGS}'"
)

# Debug 模式不添加额外优化标志
set(CMAKE_CUDA_FLAGS_DEBUG "")

# ------------------------------------------------------------------------------
# FMHA (Flash Multi-Head Attention) Cubin 条件编译
# 作用：根据目标架构排除不需要的 SM 架构，减少二进制体积
# 只有目标架构的 cubin 会被编译
# ------------------------------------------------------------------------------

# 所有支持的 SM 架构版本
# 80=Ampere, 86=Ampere, 87=Orin, 89=Ada, 100=Lovelace, 101=Hopper, 110=Blackwell, 121=GB10
set(FMHA_ALL_SM_VERSIONS
    80
    86
    87
    89
    100
    101
    120
    121)

# 需要排除的 SM 架构定义
set(FMHA_EXCLUDE_DEFINITIONS "")

# 根据目标架构构建所需的 FMHA SM 版本列表
# 运行时某些 SM 版本会复用其他版本的 cubin
set(REQUIRED_FMHA_SM_VERSIONS ${CMAKE_CUDA_ARCHITECTURES})

# 特殊处理：SM 110 (Blackwell) 运行时使用 SM 101 (Hopper) 的 FMHA cubin
if(110 IN_LIST CMAKE_CUDA_ARCHITECTURES AND NOT 101 IN_LIST
                                            CMAKE_CUDA_ARCHITECTURES)
  list(APPEND REQUIRED_FMHA_SM_VERSIONS 101)
endif()

# 排除不需要的 SM 版本，生成编译定义（选出排除掉的SM版本，加入到 FMHA_EXCLUDE_DEFINITIONS 列表中）
foreach(SM_VER ${FMHA_ALL_SM_VERSIONS})
  if(NOT SM_VER IN_LIST REQUIRED_FMHA_SM_VERSIONS)
    # 例如：EXCLUDE_SM_80, EXCLUDE_SM_86 等
    list(APPEND FMHA_EXCLUDE_DEFINITIONS "EXCLUDE_SM_${SM_VER}")
  endif()
endforeach()

if(FMHA_EXCLUDE_DEFINITIONS)
  message(
    STATUS
      "FMHA Kernels: Excluding SM architectures: ${FMHA_EXCLUDE_DEFINITIONS}")
else()
  message(STATUS "FMHA Kernels: Compiling cubins for all SM architectures")
endif()

# ========== 源文件收集 ==========

# 通用 C++ 源文件
file(GLOB_RECURSE COMMON_CPP_SRCS "common/*.cpp")
# CUDA kernel 源文件 (.cpp)
file(GLOB_RECURSE KERNELS_CPP_SRCS "kernels/*.cpp")
# CUDA kernel 源文件 (.cu)
file(GLOB_RECURSE KERNELS_CU_SRCS "kernels/*.cu")

# 排除 MoE Marlin ops/sm80 - 仅通过 moeMarlin.cu 的 include 编译
list(FILTER KERNELS_CU_SRCS EXCLUDE REGEX
     "marlin_moe_wna16/(ops|sm80_kernel_float16_u4_float16)\\.cu$")

# Sampler 源文件
file(GLOB_RECURSE SAMPLER_CU_SRCS "sampler/*.cu")
# 多模态源文件
file(GLOB_RECURSE MULTIMODAL_CPP_SRCS "multimodal/*.cpp")
# 运行时源文件
file(GLOB_RECURSE RUNTIME_CPP_SRCS "runtime/*.cpp")
# Profiler 源文件
file(GLOB_RECURSE PROFILER_CPP_SRCS "profiling/*.cpp")

# MoE Marlin kernel 源文件（启用可重定位设备代码）
file(GLOB_RECURSE MOE_MARLIN_CU_SRCS "kernels/moe/moe_marlin/*.cu")
set_source_files_properties(${MOE_MARLIN_CU_SRCS} PROPERTIES COMPILE_FLAGS
                                                             "-rdc=true")

# ========== 构建 edgellmKernels 静态库 ==========
# 包含：通用模块 + CUDA kernel 模块
add_library(edgellmKernels STATIC ${KERNELS_CPP_SRCS} ${KERNELS_CU_SRCS}
                                  ${COMMON_CPP_SRCS})
# 设置头文件搜索路径，PRIVATE表示这些头文件仅对当前模块有效，如果有其他库链接了edgellmKernels，并不会继承这些头文件路径
# 内部实现细节被隐藏了，外部调用不需要知道这些头文件在哪里
target_include_directories(edgellmKernels PRIVATE ${COMMON_INCLUDE_DIRS})
# 向编译器传递宏定义（相当于命令行里的 -D 参数），相当于剪支，不需要的架构就不编译
# 应用 FMHA SM 排除定义
target_compile_definitions(edgellmKernels PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
# 配置目标库编译的属性
# 启用 CUDA 可分离编译（用于 MoE Marlin kernel 模板）
# 不同的cu文件分开编译，确保库中的多个CUDA源文件能够相互调用，实现函数的模块化组织
set_target_properties(edgellmKernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ========== 构建 edgellmCore 静态库 ==========
# 包含：所有核心模块（Common + Kernels + Sampler + Multimodal + Runtime + Profiler）
add_library(
  edgellmCore STATIC
  ${COMMON_CPP_SRCS}
  ${COMMON_CU_SRCS}
  ${KERNELS_CPP_SRCS}
  ${KERNELS_CU_SRCS}
  ${SAMPLER_CU_SRCS}
  ${MULTIMODAL_CPP_SRCS}
  ${RUNTIME_CPP_SRCS}
  ${PROFILER_CPP_SRCS})
target_include_directories(edgellmCore PRIVATE ${COMMON_INCLUDE_DIRS})
# 链接动态链接库 (-ldl)
target_link_libraries(edgellmCore PRIVATE dl)
# 应用 FMHA SM 排除定义
target_compile_definitions(edgellmCore PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
# 启用 CUDA 可分离编译
set_target_properties(edgellmCore PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ========== 构建 edgellmTokenizer 静态库 ==========
file(GLOB_RECURSE TOKENIZER_CPP_SRCS "tokenizer/*.cpp")
add_library(edgellmTokenizer STATIC ${TOKENIZER_CPP_SRCS})
target_include_directories(edgellmTokenizer PRIVATE ${COMMON_INCLUDE_DIRS})

# ========== 构建 edgellmBuilder 静态库 ==========
file(GLOB_RECURSE BUILDER_CPP_SRCS "builder/*.cpp")
add_library(edgellmBuilder STATIC ${BUILDER_CPP_SRCS} ${COMMON_CPP_SRCS})
target_include_directories(edgellmBuilder PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                                  ${COMMON_INCLUDE_DIRS})
target_link_libraries(edgellmBuilder PRIVATE ${NV_ONNX_PARSER_LIB} dl)

# ========== 构建 TensorRT Plugin 动态库 ==========
# 生成名为 libNvInfer_edgellm_plugin.so 的共享库
file(GLOB_RECURSE PLUGIN_CPP_SRCS "plugins/*.cpp")
add_library(NvInfer_edgellm_plugin SHARED ${PLUGIN_CPP_SRCS})
# 设置版本：VERSION=1.0（实际版本）, SOVERSION=1（接口版本）
set_target_properties(NvInfer_edgellm_plugin PROPERTIES VERSION 1.0 SOVERSION 1)
# 输出到 build 目录
set_target_properties(NvInfer_edgellm_plugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                        ${CMAKE_BINARY_DIR})
target_include_directories(NvInfer_edgellm_plugin
                           PRIVATE ${COMMON_INCLUDE_DIRS})
# 应用 FMHA SM 排除定义
target_compile_definitions(NvInfer_edgellm_plugin
                           PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
# 链接所需库
target_link_libraries(
  NvInfer_edgellm_plugin PRIVATE commonLibraryExt edgellmKernels
                                 ${CUDA_DRIVER_LIB} ${CUDART_LIB})
# 交叉编译链接选项
# 具体实现为指定库的路径
add_cross_build_link_options(NvInfer_edgellm_plugin)

# ========== CuTe DSL FMHA (Blackwell SM100+) ==========
# 预编译的 CuTe DSL FMHA 内核库（用于 Blackwell 架构 SM100+）
# 离线生成：python kernelSrcs/fmha_cutedsl_blackwell/build_static_lib.py
# 启用方式：-DENABLE_CUTE_DSL_FMHA=ON（需要预编译产物）
option(ENABLE_CUTE_DSL_FMHA "Link prebuilt CuTe DSL FMHA kernels" OFF)
if(ENABLE_CUTE_DSL_FMHA)
  include(${CMAKE_SOURCE_DIR}/cmake/CuteDslFMHA.cmake)
  cute_dsl_fmha_setup(TARGETS edgellmKernels edgellmCore LINK_TARGETS
                      NvInfer_edgellm_plugin)
endif()
