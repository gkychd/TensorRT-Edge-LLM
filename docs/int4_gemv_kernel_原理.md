# INT4 Groupwise GEMV Kernel 原理文档

## 1. 问题定义

计算矩阵向量乘法：

```
output[M, N] = input[M, K] × W[K, N]
```

其中：
- `input` 是 FP16 activation，shape `[M, K]`，M=1~6（小 batch）
- `W` 是 INT4 量化权重，原始 shape `[K, N]`，每 `group_size=128` 个 K 位置共享一个 FP16 scale
- `output` 是 FP16，shape `[M, N]`

**典型参数**：M=1, K=2048, N=11008（LLaMA FFN gate/up projection）

---

## 2. 权重 Packing 格式（Python 端预处理）

原始 INT4 权重 `[K, N]` 经过三步重排后 pack 为 `[N//8, K]` 的 int32 数组。

### 2.1 为什么需要重排？

GPU kernel 的访存模式和计算模式有特定要求：
1. **128-bit 对齐 load**：每个线程一次读 128 bits（4 个 uint32 = 32 个 int4）
2. **kInterleave=4 交错**：每个 thread block 同时处理 4 个输出通道，需要 4 行数据在内存中连续
3. **dequantize_s4_to_fp16x2 的输入格式**：PTX 反量化指令期望特定的 bit 排列

### 2.2 三步重排详解

以一行 K=128 的数据为例，先转置为 `[N, K]` 再处理：

#### Phase 1：K 维度 stride-8 交错

```python
reshape(N, K//32, 4, 4, 2).transpose(0, 1, 3, 2, 4).reshape(N, K//32, 32)
```

把每 32 个元素按 `(4, 4, 2)` 解读，交换中间两维：

```
原始:    0  1 | 2  3 | 4  5 | 6  7
         8  9 |10 11 |12 13 |14 15
        16 17 |18 19 |20 21 |22 23
        24 25 |26 27 |28 29 |30 31

交换后:  0  1 | 8  9 |16 17 |24 25
         2  3 |10 11 |18 19 |26 27
         4  5 |12 13 |20 21 |28 29
         6  7 |14 15 |22 23 |30 31
```

**目的**：匹配 `dequantize_s4_to_fp16x2` 的输入期望。该函数从一个 uint32 中提取 8 个 int4，但期望它们按 stride-8 排列（对应 FP16 magic number 的 bit 操作顺序）。

#### Phase 2：K 维度 pair-swap

```python
reshape(N, K//32, 4, 4, 2).transpose(0, 1, 2, 4, 3).reshape(N, K)
```

在 Phase 1 结果上，把每组 `(4, 2)` 转置为 `(2, 4)`：

```
Phase 1 的 [0,1,8,9,16,17,24,25] 按 (4,2):
  [0,  1]
  [8,  9]
  [16, 17]
  [24, 25]

转置为 (2,4):
  [0, 8, 16, 24]
  [1, 9, 17, 25]
```

**目的**：匹配 kernel 中反 shuffle 循环的访问模式。kernel 用 `kShuffleContinous=4, kShuffleStrided=4, kShuffleBasicTile=2` 的嵌套循环还原数据，Phase 2 保证还原后得到正确的原始顺序。

#### Phase 3：N 维度 interleave

```python
reshape(N//4, 4, K//64, 64).transpose(0, 2, 1, 3).reshape(N//4, K//64, 64, 4)
```

把每 4 行为一组，按 kStride=64 分块后交错存储：

```
原始内存顺序（行优先）:
  row0 全部 K → row1 全部 K → row2 全部 K → row3 全部 K

交错后:
  [row0_chunk0][row1_chunk0][row2_chunk0][row3_chunk0] | [row0_chunk1][row1_chunk1]...
  |←────────── kStride=64 ──────────→|
```

**目的**：一个 thread block 处理 4 个输出通道时，一次 128-bit load 就能拿到 4 行在同一 K 区间的数据，最大化内存带宽利用率。

#### 最终 bit-packing

```python
int16 = val[..., 0] | (val[..., 1] << 4) | (val[..., 2] << 8) | (val[..., 3] << 12)
```

4 个 4-bit 值压入一个 int16，再 view 为 int32 得到最终 `[N//8, K]` 的 packed 权重。

---

## 3. Kernel 整体架构

### 3.1 Grid/Block 配置

```
Grid:  n / (NPerBlock × kInterleave) = n/8 个 block
Block: 256 个线程 = 8 个 warp
```

每个 block 负责 `NPerBlock × kInterleave = 2×4 = 8` 个输出通道的**完整** K 维度归约。

### 3.2 线程分工

256 个线程如何瓜分 K 维度：

```
kStride = 64          每个 K tile 的宽度
kElemsPerThread = 32  每线程处理 32 个 int4
kThreadsNumPerTile = 2  每 tile 需要 2 个线程
kInterleave = 4       4 行交错

每 8 个线程 (2 × 4) 为一组：
  - 2 个线程覆盖一个 kStride=64 的 tile
  - 4 组对应 4 个 interleave 行

256 / 8 = 32 组，每组覆盖 64 个 K 位置
一次迭代覆盖: 32 × 64 = 2048 个 K 位置
```

对于 K=2048，主循环只需 1 次迭代。

### 3.3 线程 ID 到数据偏移的映射

```c
threadIdx.x = 0~255

thd_row_offset = (threadIdx.x / 2) % 4     // 当前线程负责哪个 interleave 行 (0~3)
act_k_offset   = (threadIdx.x / 8) * 64    // K 维度大偏移（哪个 tile）
               + (threadIdx.x % 2) * 32    // K 维度小偏移（tile 内前半/后半）
```

示例：
| threadIdx.x | thd_row_offset | act_k_offset | 含义 |
|:-----------:|:--------------:|:------------:|:----:|
| 0 | 0 | 0 | tile0 前半, row0 |
| 1 | 0 | 32 | tile0 后半, row0 |
| 2 | 1 | 0 | tile0 前半, row1 |
| 3 | 1 | 32 | tile0 后半, row1 |
| 4 | 2 | 0 | tile0 前半, row2 |
| 5 | 2 | 32 | tile0 后半, row2 |
| 6 | 3 | 0 | tile0 前半, row3 |
| 7 | 3 | 32 | tile0 后半, row3 |
| 8 | 0 | 64 | tile1 前半, row0 |
| ... | ... | ... | ... |

---

## 4. 主循环详解

### 4.1 权重加载

```c
*((float4*)(local_qweights)) = *((float4*)(blk_weight_ptr + offset));
```

一次 128-bit load 读取 4 个 uint32 = 32 个 int4 值。

由于 Phase 3 的 interleave 格式，这 32 个 int4 全部属于**同一个 interleave 行**在当前 K tile 内的数据。

### 4.2 INT4 → FP16 反量化

```c
dequantize_s4_to_fp16x2(*reinterpret_cast<half2*>(local_qweights + i),
                         reinterpret_cast<uint4*>(half_weight_buffer + i * 8));
```

每次处理 1 个 uint32（8 个 int4）→ 8 个 FP16。

**PTX 实现原理**：
1. 用 `lop3.b32` 指令提取 4-bit 字段并拼接 FP16 的 exponent bits（magic number `0x64006400`）
2. 得到的中间值是 `1024 + original_value`（FP16 格式）
3. 用 `sub.f16x2` 减去 1032 得到 `[-8, 7]` 范围的 FP16 值

### 4.3 反 Shuffle + 乘 Scale

```c
for (i = 0; i < 4; ++i)        // kShuffleContinous
    for (j = 0; j < 4; ++j)    // kShuffleStrided
        half2 w = half_weight_buffer[(i + j*4) * 2 : (i + j*4) * 2 + 2]
        w *= scale
        dequantized_weight[(i*4+j)*2 * NPerBlock + idx] = w.x
        dequantized_weight[(i*4+j)*2 * NPerBlock + idx + 1] = w.y
```

这个双重循环做两件事：
1. **反 shuffle**：从 Phase1+Phase2 重排后的位置读取数据，按 `(i + j×4) × 2` 的模式访问，还原为原始 K 顺序
2. **乘 scale**：应用 group-wise 量化的缩放因子
3. **转置存储**：输出按 `[K_position, N_channel]` 排列（K 在外层），方便后续 MAC 循环连续访问同一 K 位置的所有输出通道

### 4.4 MAC 累加

```c
for (x = 0; x < NPerBlock/2; ++x)       // 遍历输出通道对
    for (y = 0; y < kElemsPerThread; ++y) // 遍历 K 位置
        psum[x*2 : x*2+2] += dequantized_weight[y*NPerBlock + x*2 : ...] * input[y]
```

使用 `__hfma2`（half2 FMA）一次处理 2 个输出通道：
- `a = dequantized_weight`：2 个通道在 K=y 位置的权重
- `b = __half2half2(input[y])`：activation 广播为 half2
- `c = psum`：累加器

---

## 5. Warp 归约

### 5.1 为什么需要归约？

同一个输出通道的 K 维度被分散到多个线程处理。以 K=2048 为例：
- 32 组线程，每组覆盖 64 个 K 位置
- 但同一 interleave 行只有 32/4=8 组线程在处理
- 这 8 组分布在 warp 内的 lane 0,1,8,9,16,17,24,25（以 row0 为例）

### 5.2 归约步骤

```
Step 1: xor 16  →  lane 0 += lane 16,  lane 1 += lane 17, ...
Step 2: xor 8   →  lane 0 += lane 8,   lane 1 += lane 9, ...
Step 3: xor 1   →  lane 0 += lane 1,   lane 2 += lane 3, ...
```

三步后：
- lane 0 持有 interleave row 0 的完整结果
- lane 2 持有 interleave row 1 的完整结果
- lane 4 持有 interleave row 2 的完整结果
- lane 6 持有 interleave row 3 的完整结果

### 5.3 跨 Warp 归约

每个 warp 的 lane 0,2,4,6 将结果写入 shared memory，最后由少量线程累加所有 warp 的贡献并写回全局内存。

---

## 6. 数据流总结

```
┌─────────────────────────────────────────────────────────────┐
│ Global Memory                                               │
│                                                             │
│  weights[N/8, K] (int32, packed)                            │
│  scales[K/128, N] (fp16)                                    │
│  input[M, K] (fp16)                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ 128-bit load (float4)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Registers: local_qweights[4] (uint32)                       │
│            = 32 个 int4 值                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ dequantize_s4_to_fp16x2 (PTX)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Registers: half_weight_buffer[32] (fp16)                    │
│            = 32 个反量化后的 FP16 权重                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ 反 shuffle + × scale
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Registers: dequantized_weight[32 × NPerBlock] (fp16)        │
│            按 [K_pos, N_channel] 排列                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ __hfma2 (MAC)
                           │ × input (128-bit load)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Registers: psum[Num] (fp16)                                 │
│            每线程的部分和                                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ __shfl_xor_sync (warp reduce)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Shared Memory: out_smem[warps][channels] (float)            │
│                每 warp 的归约结果                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ 跨 warp 累加 + 写回
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Global Memory: output[M, N] (fp16)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 性能关键点

| 优化手段 | 说明 |
|:---------|:-----|
| 128-bit vectorized load | 每次访存读 16 bytes，充分利用内存带宽 |
| 寄存器内计算 | 权重反量化、scale 乘法、MAC 全在寄存器完成，无 shared memory bank conflict |
| half2 FMA | 一条指令处理 2 个 FP16 乘加，吞吐翻倍 |
| Interleave=4 packing | 4 行数据连续存储，一次 load 服务 4 个输出通道 |
| 编译时 batch 特化 | Batch 作为模板参数，循环完全展开，无运行时分支 |
| Warp shuffle 归约 | 无 shared memory 开销的快速归约 |

---

## 8. 与 GEMM Kernel 的关系

| | GEMV (本文件) | GEMM (int4WoqGemmCuda.cu) |
|:--|:--|:--|
| 适用场景 | M ≤ 6 | M > 6 |
| 计算方式 | 标量/half2 FMA | Tensor Core MMA (m16n8k16) |
| 数据搬运 | 直接 global → register | global → shared memory → register |
| 权重处理 | 寄存器内反量化 | shared memory 内反量化 |
| 流水线 | 无 | 多级 async pipeline (cp.async) |

Plugin 的 `enqueue` 函数根据 M 值自动选择：
```cpp
if (M <= 6)
    gemv_bf16_forward_cuda_new(...);  // 本 kernel
else
    gemm_bf16_forward_cuda_new(...);  // Tensor Core kernel
```
