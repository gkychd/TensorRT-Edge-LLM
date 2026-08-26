/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright (c) 2023 MIT HAN Lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * reference: https://github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization_new/gemv/gemv_cuda.cu
 */

#include "dequantize.cuh"
#include <cuda_fp16.h>
#include <stdexcept>

// 一个 int32 中 pack 了 8 个 int4 值
#define PACK_FACTOR 8
#define WARP_SIZE 32
// 每次内存访问 128 bits = 16 bytes
#define MEM_ACCESS_SIZE 128

namespace trt_edgellm
{
namespace kernel
{

// ============================================================================
// Warp 内归约：将每个线程的部分和汇总到 shared memory
//
// 背景：由于权重的 interleave=4 packing 格式，一个 warp 内的线程按特定模式
// 分工处理 K 维度的不同区间。归约需要把分散在不同线程中的部分和累加起来。
//
// 线程分工模式（以 kInterleave=4 为例）：
//   每 kThreadsNumPerTile=2 个线程处理一个 kStride=64 的 tile
//   每 kInterleave=4 组 tile 对应同一个 K 区间
//   所以每 8 个线程（2×4）处理同一个 K 区间的 4 个 interleave 行
//
// 需要归约的线程对：
//   lane 0,1 处理同一 K 区间 → xor 1 归约
//   lane 0,8 处理相邻 K 区间 → xor 8 归约
//   lane 0,16 处理更远 K 区间 → xor 16 归约
//
// 归约后，lane 0,2,4,6 各持有一个 interleave 行的完整结果
// ============================================================================
template <int Num, int WarpSize>
__device__ __forceinline__ static void warp_reduce(half* psum, float (*out_smem)[Num * 4])
{
    // 先转 float 避免 half 精度累加误差
    float fpsum[Num];
#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
        fpsum[i] = static_cast<float>(psum[i]);
    }

#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
        // xor 16: 归约 lane 0↔16, 1↔17, ... （跨半 warp）
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 16);
        // xor 8:  归约 lane 0↔8, 1↔9, ...
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 8);
        // xor 1:  归约 lane 0↔1, 2↔3, ...（同一 tile 内的 2 个线程）
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 1);
    }
    __syncthreads();

    int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
    // 归约完成后，lane 0,2,4,6 分别持有 interleave 行 0,1,2,3 的结果
    // 写入 shared memory 供最终输出
    if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
    {
#pragma unroll
        for (int i = 0; i < Num; ++i)
        {
            out_smem[warp][i * 4 + lane / 2] = fpsum[i];
        }
    }
    __syncthreads();
};

__device__ __forceinline__ int make_divisible(int c, int divisor)
{
    return (c + divisor - 1) / divisor;
}

// ============================================================================
// INT4 Groupwise GEMV Kernel
//
// 功能：计算 output[M, N] = input[M, K] × W_dequant[K, N]
//       其中 W 是 INT4 量化 + group-wise scale 的权重
//
// 模板参数：
//   NPerBlock: 每个 block 处理的"逻辑"输出通道数（实际处理 NPerBlock × kInterleave 个）
//   Batch:     batch size (M)，编译时确定
//   BlockSize: 每个 block 的线程数（256）
//   GroupSize: 量化分组大小（128）
//
// 权重内存布局（经过 pack_intweights 重排后）：
//   - N 维度按 kInterleave=4 交错：每 4 行为一组，按 kStride=64 分块交替存储
//   - K 维度内部经过 Phase1/Phase2 shuffle（适配 ldmatrix/MMA 访存模式）
//   - 最终 4 个 4-bit 值 pack 到一个 int16（存为 int8 对）
//
// 线程分工（以 BlockSize=256, kStride=64, kElemsPerThread=32 为例）：
//   - 每 2 个线程（kThreadsNumPerTile=2）覆盖一个 kStride=64 的 K tile
//   - 每 4 组 tile（kInterleave=4）对应 4 个交错行
//   - 每 8 个线程处理一个 (K_tile, 4_interleave_rows) 单元
//   - 256 个线程 = 32 个这样的单元，覆盖 32×64 = 2048 个 K 位置（一次迭代）
//
// 以 m=1, n=11008, k=2048 为例：
//   grid:  n / NPerBlock / kInterleave = 11008 / 2 / 4 = 1376 个 block
//   block: 256 个线程, 8 个 warp
//   每个 block 负责 NPerBlock × kInterleave = 2×4 = 8 个输出通道
//   主循环迭代次数: IC × kInterleave / (BlockSize × kElemsPerThread) = 2048×4 / (256×32) = 1
// ============================================================================
template <int NPerBlock, int Batch, int BlockSize, int GroupSize>
__global__ void gemv_kernel(
    half const* inputs, uint32_t const* weight, half const* scales, half* outputs, int const IC, int const OC)
{
    // ---- 常量定义 ----
    int const kStride = 64;                              // 每个 K tile 的宽度
    int const kElemsPerThread = MEM_ACCESS_SIZE / 4;     // 128/4 = 32，每线程处理 32 个 int4 值
    int const kThreadsNumPerTile = kStride / kElemsPerThread; // 64/32 = 2，每 tile 需要 2 个线程

    // 反 shuffle 参数（对应 pack 时的 Phase1/Phase2 重排）
    static constexpr int kShuffleBasicTile = 2;   // 最小 shuffle 单元：2 个连续元素
    static constexpr int kShuffleContinous = 4;   // 连续方向的 tile 数
    static constexpr int kShuffleStrided = 4;     // 跨步方向的 tile 数

    constexpr int Num = NPerBlock * Batch;        // 每线程需要维护的部分和数量
    constexpr int kInterleave = 4;                // N 维度交错因子

    // ---- 寄存器分配 ----
    half local_inputs[kElemsPerThread];                    // 32 个 activation 值
    uint32_t local_qweights[MEM_ACCESS_SIZE / 32];         // 128bit / 32bit = 4 个 uint32（packed int4）
    half half_weight_buffer[kElemsPerThread];               // 反量化后的 32 个 FP16 权重（中间缓冲）
    half dequantized_weight[kElemsPerThread * NPerBlock];   // scale 后的权重，按输出通道排列
    half local_scale[NPerBlock];                            // 当前 group 的 scale 值

    // 部分和累加器
    half psum[Num];
    for (int i = 0; i < Num; ++i)
        psum[i] = static_cast<half>(0.f);

    // Shared memory：每个 warp 一行，用于 warp 间归约
    // 维度 [warp数×2][Num × kInterleave]
    // ×2 是因为 warp_reduce 中 lane 0,2,4,6 各写一个 interleave 行
    __shared__ float out_smem[BlockSize / WARP_SIZE * 2][Num * kInterleave];

    // ---- 计算当前线程的数据偏移 ----

    // 当前 block 负责的起始输出通道（N 维度偏移）
    int const blk_row_offset = blockIdx.x * NPerBlock * kInterleave;

    // 当前线程在 kInterleave=4 中负责哪一行（0~3）
    int const thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;

    // 当前线程负责的 K 维度起始偏移
    // 逻辑：先按 (kThreadsNumPerTile × kInterleave) = 8 个线程为一组，
    //        组号 × kStride 得到大偏移，组内位置 × kElemsPerThread 得到小偏移
    int const act_k_offset = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride
        + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;

    // 当前 K 偏移对应的 group 索引
    int const group_offset = act_k_offset / GroupSize;

    // 权重指针：指向当前 block 负责的 N 区间的起始位置
    // 除以 PACK_FACTOR=8 是因为每个 uint32 存了 8 个 int4
    uint32_t const* blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;

    // Scale 指针：scales 的布局是 [K/GroupSize, N]
    // blk_row_offset + thd_row_offset 定位到具体的 interleave 行
    // group_offset * OC 跳到对应的 group
    half const* scale_ptr = scales + blk_row_offset + thd_row_offset + group_offset * OC;

    // Activation 指针：直接按 K 偏移
    half const* inputs_ptr = inputs + act_k_offset;

    // 每次主循环迭代，所有线程共同前进的 K 步长
    int const act_forward_step = BlockSize * kElemsPerThread / kInterleave;
    // 对应的 scale 前进步长（每前进 GroupSize 个 K 位置，scale 换一行）
    int const scale_forward_step = act_forward_step / GroupSize * OC;

    // ============================================================================
    // 主循环：遍历 K 维度
    // 循环条件中 IC × kInterleave 是因为权重按 interleave 格式存储，
    // 实际 K 维度被展开了 kInterleave 倍（4 行交错）
    // ============================================================================
    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread)
    {
        // ---- 加载并反量化权重 ----
#pragma unroll
        for (int idx = 0; idx < NPerBlock; ++idx)
        {
            // 128-bit load：一次读取 4 个 uint32 = 32 个 int4 值
            // 权重地址计算：idx * kInterleave * IC 跳到下一个 NPerBlock 组
            //               kk 是当前 K 位置（已含 interleave 展开）
            *((float4*) (local_qweights)) = *((float4*) (blk_weight_ptr + (idx * kInterleave * IC + kk) / PACK_FACTOR));

            // 加载当前 group 的 scale
            local_scale[idx] = *(scale_ptr + idx * kInterleave);

            // INT4 → FP16 反量化
            // 每个 uint32 包含 8 个 int4，用 PTX lop3 + magic number 技巧转为 8 个 FP16
#pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i)
            {
                dequantize_s4_to_fp16x2(*reinterpret_cast<half2*>(local_qweights + i),
                    reinterpret_cast<uint4*>(half_weight_buffer + i * PACK_FACTOR));
            }

            // 反 shuffle + 乘 scale
            // pack 时做了 Phase1(stride-8 交错) + Phase2(pair-swap)，这里要还原
            // kShuffleContinous=4, kShuffleStrided=4, kShuffleBasicTile=2
            // 遍历顺序：连续方向 i, 跨步方向 j, 每组 2 个元素
            // 还原后的元素按 [K_position × NPerBlock + output_channel] 排列
#pragma unroll
            for (int i = 0; i < kShuffleContinous; ++i)
            {
#pragma unroll
                for (int j = 0; j < kShuffleStrided; ++j)
                {
                    // 从 shuffle 后的位置读取 2 个 FP16 值
                    half2 w = *reinterpret_cast<half2*>(
                        half_weight_buffer + (i + j * kShuffleContinous) * kShuffleBasicTile);
                    // 乘以 group scale
                    w = __hmul2(w, __half2half2(local_scale[idx]));
                    // 存入 dequantized_weight，按 [K_pos, N_channel] 交错排列
                    // 这样后续 MAC 循环可以连续访问同一 K 位置的所有输出通道
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 0) * NPerBlock + idx] = w.x;
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 1) * NPerBlock + idx] = w.y;
                }
            }
        }

        // ---- 加载 activation 并执行 MAC ----
#pragma unroll
        for (int batch_idx = 0; batch_idx < Batch; ++batch_idx)
        {
            half const* local_inputs_ptr = inputs_ptr + batch_idx * IC;
            // 128-bit load activation：每次 8 个 half = 128 bits
#pragma unroll
            for (int idx = 0; idx < kElemsPerThread / 8; ++idx)
            {
                *((float4*) (local_inputs + idx * 8)) = *((float4*) (local_inputs_ptr + idx * 8));
            }

            // FMA 累加：对每个输出通道，遍历所有 K 位置做乘加
            // 使用 half2 FMA 一次处理 2 个输出通道（NPerBlock/2 次）
#pragma unroll
            for (int x = 0; x < NPerBlock / 2; ++x)
            {
#pragma unroll
                for (int y = 0; y < kElemsPerThread; ++y)
                {
                    // __hfma2(a, b, c) = a * b + c
                    // a: 2 个输出通道在 K=y 位置的权重
                    // b: activation[y] 广播为 half2
                    // c: 累加器
                    *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2) = __hfma2(
                        *reinterpret_cast<half2*>(dequantized_weight + y * NPerBlock + x * 2),
                        __half2half2(local_inputs[y]), *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2));
                }
            }
        }
        // 前进到下一个 K 区间
        inputs_ptr += act_forward_step;
        scale_ptr += scale_forward_step;
    }

    // ---- Warp 内归约 ----
    // 将分散在不同线程中的部分和累加到 shared memory
    warp_reduce<Num, WARP_SIZE>(psum, out_smem);

    // ---- 写回全局内存 ----
    // 每个 block 输出 Num × kInterleave = Batch × NPerBlock × 4 个值
    // 用所有线程协作写回
    for (int i = threadIdx.x; i < Num * kInterleave; i += BlockSize)
    {
        int batch_idx = i / (NPerBlock * kInterleave);
        int oc_idx = i % (NPerBlock * kInterleave);
        // 跨 warp 归约：累加所有 warp 的贡献
        float acc = 0.f;
        for (int j = 0; j < BlockSize / WARP_SIZE; ++j)
        {
            acc += out_smem[j][i];
        }
        outputs[batch_idx * OC + blk_row_offset + oc_idx] = static_cast<half>(acc);
    }
}

// ============================================================================
// GEMV 入口函数
//
// 参数：
//   in_feats:        输入 activation [M, K]，FP16
//   weights_device:  packed INT4 权重 [N/8, K]，int32 格式（每个 int32 存 8 个 int4）
//   scaling_factors: group-wise scale [K/group_size, N]，FP16
//   out_feats:       输出 [M, N]，FP16
//   m: batch size (1~6)
//   n: 输出维度 N
//   k: 输入维度 K
//   group_size: 量化分组大小（必须为 128）
//
// Grid/Block 配置：
//   grid:  n / N_PER_BLOCK / K_INTERLEAVE = n/8 个 block
//   block: BLOCK_SIZE=256 个线程（8 个 warp）
//   每个 block 负责 N_PER_BLOCK × K_INTERLEAVE = 2×4 = 8 个输出通道的完整计算
// ============================================================================
void gemv_forward_cuda_new(half const* in_feats, int8_t const* weights_device, half const* scaling_factors,
    half* out_feats, int m, int n, int k, int group_size, cudaStream_t stream)
{
    // 将 int8 指针重解释为 uint32（每个 uint32 = 8 个 int4）
    auto kernel = reinterpret_cast<uint32_t const*>(weights_device);
    static constexpr int N_PER_BLOCK = 2;    // 每 block 处理 2 个"逻辑"输出通道组
    static constexpr int K_INTERLEAVE = 4;   // N 维度交错因子
    static constexpr int BLOCK_SIZE = 256;   // 每 block 256 线程

    dim3 num_blocks(n / N_PER_BLOCK / K_INTERLEAVE);  // 总 block 数 = n/8
    dim3 num_threads(BLOCK_SIZE);

    if (group_size != 128)
    {
        throw std::runtime_error("Unsupported group size for gemv kernel.\n");
    }
    // Batch size 在编译时确定（模板参数），避免运行时分支
    switch (m)
    {
    case 1:
        gemv_kernel<N_PER_BLOCK, 1, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    case 2:
        gemv_kernel<N_PER_BLOCK, 2, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    case 3:
        gemv_kernel<N_PER_BLOCK, 3, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    case 4:
        gemv_kernel<N_PER_BLOCK, 4, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    case 5:
        gemv_kernel<N_PER_BLOCK, 5, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    case 6:
        gemv_kernel<N_PER_BLOCK, 6, BLOCK_SIZE, 128>
            <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, k, n);
        break;
    default: throw std::runtime_error("Unsupported batch size for gemv kernel.\n");
    }
}

} // namespace kernel
} // namespace trt_edgellm