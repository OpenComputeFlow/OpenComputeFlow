# OpenComputeFlow 架构设计文档

> **An Experimental AI Compute Architecture Framework**
>
> *Exploring the abstraction from DNN APIs to hardware execution.*

---

## 1. 项目定位

### 1.1 我们不是又一个 MLIR Compiler

开源社区不缺"把 A 转成 B"的编译器项目：

```
TensorFlow → MLIR → LLVM → Assembly
PyTorch → MLIR → LLVM → Assembly
cuDNN → MLIR → LLVM → Assembly
```

它们的共同问题：**只回答了"怎么做"，没有回答"为什么这样做"**。

OpenComputeFlow 的目标不同：

> **研究 DNN API 到 AI 芯片执行模型之间的抽象路径。**

我们关心的是：

- 一个 `cudnnConvolutionForward` 应该如何被逐步分解为硬件指令？
- 每一层的抽象边界在哪里？为什么画在那里？
- 不同的硬件（向量机、脉动阵列、SIMT）如何统一到同一条 lowering 路径下？
- 如何用 Cost Model 驱动 lowering 决策，而不是硬编码规则？

**核心价值在 IR 抽象层，不在 LLVM。** LLVM 只是最后一步的编码工具。

### 1.2 项目的长期目标

成为 AI 芯片架构研究的实验平台——你可以在这个框架上：

- 定义新的计算 IR 层，验证抽象边界的合理性
- 插入 Cost Model，研究自动调优策略
- 描述一种新硬件，看编译器能否自动生成合理代码
- 可视化 lowering 全过程，理解计算如何一步步映射到硬件

---

## 2. 核心理念：Progressive Lowering

### 2.1 每一层回答一个问题

一个 Project 如果只是把 cuDNN API 调用的结果转成 LLVM IR，那它只是一段转换代码。但如果你把整个过程分解为清晰的抽象层次，每一层回答一个独立的问题，它就变成了**架构**。

```
┌────────────────────────────────────────────────────────────┐
│                      回答什么问题？                         │
├────────────────────────────────────────────────────────────┤
│  Tensor IR      │  算法是什么？                              │
│                 │  数据依赖图、算子语义、数据布局             │
├─────────────────┼──────────────────────────────────────────┤
│  Loop IR        │  如何计算？                                │
│                 │  Affine 循环嵌套、访存模式、数据依赖距离    │
├─────────────────┼──────────────────────────────────────────┤
│  Schedule IR    │  如何组织计算？                             │
│                 │  Tile 划分、Pipeline、并行度、循环重排      │
├─────────────────┼──────────────────────────────────────────┤
│  Hardware IR    │  如何使用硬件？                             │
│                 │  向量指令、SIMT 线程束、脉动数据流          │
├─────────────────┼──────────────────────────────────────────┤
│  LLVM IR        │  如何编码？                                │
│                 │  寄存器分配、指令编码、目标三元组            │
└─────────────────┴──────────────────────────────────────────┘
```

注意：每一层**只降一个维度的抽象**。Tensor IR 不知道什么是 tile，Schedule IR 不知道什么是 vfmacc，Hardware IR 不知道什么是卷积。

### 2.2 为什么不是 3 层或 7 层

**为什么 Tensor 和 Loop 要分开？**

Tensor IR 表达的是算子间的关系（conv 的输出是 bias 的输入），是**数据依赖图**。Loop IR 表达的是单个算子的计算方式（用几重循环实现），是**执行方式**。这是两种不同的信息，混在一起会丢失结构。

**为什么 Schedule 要从 Loop 中独立出来？**

Tiling 策略（tile 大小、循环重排、pipeline 深度）是**性能最敏感的决策**。把它独立出来，Cost Model 就有了一个清晰的介入点。如果 Schedule 和 Loop 混在一起，调优逻辑会散布在整个 lowering 流程中，无法系统化。

**为什么不再加更多层？**

抽象层的价值在于"语义差异足够大"。如果两层之间只是换个名字，那就不该分开。Tensor→Loop→Schedule→Hardware→LLVM 这五层，每一层之间都有质的差异，不存在冗余。

### 2.3 统一抽象路径

同一个 lowering 管道，可以处理所有 DNN 算子：

```
Conv2D ─┐
Matmul ─┼──→ Tensor IR ──→ Loop IR ──→ Schedule IR ──→ Hardware IR ──→ LLVM IR
Attn ───┤
MoE ────┘
Reduce ─┘
```

算子到了 Loop IR 以下，就全部变成同一套原语：循环、访存、运算。Tensor IR 之上的差异被统一了。

---

## 3. 总体架构

```
                         cuDNN Frontend / PyTorch FX / ONNX / ...
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        COST MODEL（贯穿全程）                        │
│         ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│         │ Generate │ →  │ Estimate │ →  │  Choose  │               │
│         │ 生成候选  │    │ 预估代价  │    │ 选择最优  │               │
│         └──────────┘    └──────────┘    └──────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 1: TENSOR IR                    "算法是什么？"                 │
│                                                                      │
│  Op: conv2d, matmul, attention, reduce, elemwise, reshape, ...      │
│  Graph: 数据依赖边，生产者-消费者关系                                   │
│  Attr: 数据布局 (NCHW/NHWC), 数据类型 (f32/f16/i8), 问题维度          │
│                                                                      │
│  本层不做: tiling, 循环生成, 指令选择, 任何与硬件相关的决策              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  --tensor-lower-to-loop
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 2: LOOP IR                       "如何计算？"                  │
│                                                                      │
│  结构: affine.for 循环嵌套, affine.load/store, affine.if             │
│  表达: 访存模式 (连续/跨步/间接), 数据依赖距离, 循环边界               │
│                                                                      │
│  关键变换:                                                           │
│  • conv2d → im2col + affine 循环 (算法选择在此步完成)                  │
│  • matmul → 三重 affine 循环 (M,N,K)                                  │
│  • attention → 多重循环 + softmax 展开                                │
│                                                                      │
│  本层不做: tile 划分, 循环重排, 并行标注                               │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  --loop-lower-to-schedule
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 3: SCHEDULE IR                   "如何组织计算？"              │
│                                                                      │
│  这是 Cost Model 发挥核心作用的层。                                    │
│                                                                      │
│  变换:                                                               │
│  • tile(M, N, K)       — 将循环切分为 tile 和外层迭代                 │
│  • reorder(i, j, k)    — 重排循环顺序以优化数据局部性                  │
│  • pipeline(n)         — 插入软件流水线，隐藏访存延迟                  │
│  • parallel(dim)       — 标注可并行的循环维度                          │
│  • unroll(dim, factor) — 循环展开                                     │
│  • vectorize(dim)      — 标注可向量化的循环                            │
│  • memspace(tensor)    — 分配内存层级 (global/shared/register)        │
│                                                                      │
│  Cost Model 驱动决策：                                                │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│  │ 生成候选 │ →  │ 估算延迟  │ →  │ 比较选择  │ →  │ 应用变换  │        │
│  │ Tile=64 │    │ Lat=120  │    │          │    │ Tile=128 │        │
│  │ Tile=128│    │ Lat=90   │    │   ✓      │    │          │        │
│  └─────────┘    └──────────┘    └──────────┘    └──────────┘        │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  --schedule-lower-to-hardware
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 4: HARDWARE IR                   "如何使用硬件？"              │
│                                                                      │
│  从 Hardware Description (YAML) 自动推导映射规则:                      │
│                                                                      │
│  Hardware Description              Hardware IR                       │
│  ═══════════════════              ═══════════════                     │
│  arch: rvv                        rvv.vle, rvv.vfmacc,              │
│    vlen: 256                        rvv.vfredusum, ...               │
│    sew: [8,16,32,64]                                                  │
│    lmul: [1,2,4,8]                 ───→ 向量指令序列                   │
│                                                                      │
│  arch: tensor_core                 tile_load, tile_mma,              │
│    mma: [16,16,16]                   tile_store, ...                 │
│    sram: 128KB                                                       │
│                                     ───→ 张量核指令序列                │
│                                                                      │
│  arch: simt                         threadIdx, blockIdx,             │
│    warp_size: 32                      shfl_sync, ...                 │
│    smem: 48KB                                                        │
│                                     ───→ SIMT 指令序列                │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  --hardware-lower-to-llvm
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 5: LLVM IR                       "如何编码？"                  │
│                                                                      │
│  • @llvm.riscv.vle.nxv8f32, @llvm.riscv.vfmacc.nxv8f32 (RVV)       │
│  • @llvm.nvvm.load, @llvm.nvvm.mma.sync (CUDA)                      │
│  • <8 x float> fadd, fmul (x86 AVX)                                 │
│                                                                      │
│  → llc -mtriple=<target> → Machine Code                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Cost Model（一等公民）

### 4.1 不是"Lower 之后再调优"，而是"调优后再 Lower"

传统编译器的做法：

```
IR → Lower → 目标代码 → Benchmark → 不满意？ → 回退重来
```

OpenComputeFlow 的做法：

```
IR → Generate candidates → Estimate each → Choose best → Lower
```

Cost Model 不是一个事后优化 pass，而是**贯穿 lowering 全程的决策引擎**。

### 4.2 Cost Model 的输入与输出

**输入**：
- 问题维度（M, N, K 等）
- 当前 IR 片段（Loop IR 或 Schedule IR）
- Hardware Description（YAML）

**输出**：
- 预估延迟 / 吞吐 / 功耗
- 用于在多个候选方案中选择最优

### 4.3 Cost Model 在各层的应用

| IR 层 | Cost Model 决策 | 候选搜索空间 |
|---|---|---|
| Tensor IR | 算子融合方案（conv+bias+act 是否融合） | 融合 vs 不融合，2 种 |
| Loop IR | 算法选择（im2col+GEMM vs Winograd vs 直接） | 3-5 种算法 |
| Schedule IR | Tile 大小、循环重排、Pipeline 深度 | 指数级搜索空间，核心调优层 |
| Hardware IR | 指令序列选择（vfmacc vs vfmul+vfadd） | 2-5 种模式 |

### 4.4 Cost Model 的架构

```
┌──────────────────────────────────────────┐
│              Cost Model API              │
│  estimate(ir, problem_dims, hw_desc)     │
│      → {latency, throughput, energy}     │
└────────────────┬─────────────────────────┘
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│  Micro-  │ │  Memory │ │  Compute│
│  Bench   │ │  Model  │ │  Model  │
│ (实测)   │ │ (分析)   │ │ (分析)   │
└─────────┘ └─────────┘ └─────────┘
```

- **Micro-Benchmark**：测量真实硬件上的基础操作延迟（如单次 vfmacc、单次 shared memory load）
- **Memory Model**：分析访存模式，估算 cache miss、bank conflict、带宽利用率
- **Compute Model**：分析计算密度，估算计算单元利用率、流水线气泡

早期可以用分析模型（Roofline + 简单线性模型），后期可接入真实的 micro-benchmark 数据。

---

## 5. Hardware Description（YAML 驱动）

### 5.1 不绑定任何硬件

传统做法：为每种硬件写单独的 lowering pass。

```
RVV:  CIR → RVV Dialect → LLVM RVV Intrinsics
ARM:  CIR → NEON Dialect → LLVM NEON Intrinsics
x86:  CIR → AVX Dialect → LLVM x86 Intrinsics
...
```

问题：每增加一种硬件，需要新写一整套 dialect + lowering pass。N 种硬件 = N 套代码。

OpenComputeFlow 的做法：**用 YAML 描述硬件能力，编译器根据描述自动适配。**

### 5.2 Hardware Description 格式

```yaml
# hardware/riscv_v_vector.yaml
name: "RISC-V Vector 1.0"
arch: rvv
version: "1.0"

pe:
  vector_register_count: 32
  vlen: 256              # bit (128/256/512/1024)
  sew: [8, 16, 32, 64]   # supported element widths
  lmul: [0.125, 0.25, 0.5, 1, 2, 4, 8]

compute:
  fma_latency: 5          # cycles
  fma_throughput: 1       # ops/cycle/PE
  mac_per_cycle: 8        # f32 MAC per cycle (VLEN=256, SEW=32 → 8 elements)
  transcendental: false   # no hardware sigmoid/tanh/exp

memory:
  levels:
    - name: global
      type: ddr
      size: 16GB
      bandwidth: 50        # GB/s
      latency: 200         # cycles
    - name: shared
      type: sram
      size: 128KB
      bandwidth: 512
      latency: 5
    - name: register
      type: vector_register
      size: 8KB            # 32 regs × 256-bit
      bandwidth: 4096
      latency: 1

instructions:
  load:
    unit_stride:    {op: "vle.v",   throughput: 1, latency: 5}
    strided:        {op: "vlse.v",  throughput: 1, latency: 6}
  store:
    unit_stride:    {op: "vse.v",   throughput: 1, latency: 4}
    strided:        {op: "vsse.v",  throughput: 1, latency: 5}
  compute:
    fadd:           {op: "vfadd.vv", throughput: 1, latency: 3}
    fmul:           {op: "vfmul.vv", throughput: 1, latency: 3}
    fmacc:          {op: "vfmacc.vv", throughput: 1, latency: 5}
    fredsum:        {op: "vfredusum.vs", throughput: 1, latency: 8}
  compare:
    mflt:           {op: "vmflt.vv", throughput: 1, latency: 2}
  merge:
    vmerge:         {op: "vmerge.vvm", throughput: 1, latency: 2}
```

### 5.3 支持的硬件类型

| 硬件类型 | 示例 | 核心特征 |
|---|---|---|
| `rvv` | RISC-V Vector | 向量寄存器，VLEN 可变，sew/lmul 可配 |
| `simd` | ARM NEON, x86 AVX | 固定宽度 SIMD，128/256/512 bit |
| `tensor_core` | NVIDIA Tensor Core | 矩阵乘法加速器，MMA 指令 |
| `systolic` | Google TPU, 自研 NPU | 脉动阵列，权重驻留，数据流驱动 |
| `simt` | CUDA cores, AMD CU | 线程束 (warp/wavefront)，共享内存 |
| `dsp` | Hexagon, Cadence | VLIW/SIMD 混合，窄位宽优化 |

每种硬件一个 YAML 文件。**换硬件只需换 YAML，编译器自动调整 lowering 策略。**

---

## 6. 各层详细设计

### 6.1 Tensor IR — "算法是什么？"

**职责**：表达 DNN 计算的**全局语义**。不涉及任何执行细节。

**表达的内容**：
- **算子类型**：conv2d, matmul, attention, pooling, reduce, elemwise, reshape, ...
- **算子参数**：kernel_size, stride, padding, dilation, groups
- **数据流图**：张量之间的生产者-消费者关系
- **数据属性**：shape, dtype (f32/f16/bf16/i8), layout (NCHW/NHWC/...)
- **数值属性**：alpha, beta 等缩放因子

**本层关键变换**：
- **算子融合**：检测 conv+bias+activation 序列，合并为 `fused_conv_bias_act`
- **Layout 推断**：根据后续算子需求，为中间张量选择最优 layout

**本层的"不"**：
- 不做 tiling
- 不做循环生成
- 不做任何与硬件相关的决策
- 不做常量折叠等通用优化（由上游 MLIR canonicalizer 完成）

### 6.2 Loop IR — "如何计算？"

**职责**：将每个算子展开为**具体的循环结构和访存模式**。算法选择在此层完成。

**表达的内容**：
- **循环结构**：`affine.for %i = 0 to N step 1`
- **访存操作**：`affine.load %tensor[%i, %j]` / `affine.store %val → %tensor[%i, %j]`
- **访存模式标注**：连续访问 / 跨步访问（stride）/ 间接访问（gather/scatter）
- **数据依赖**：循环间的依赖距离（用于后续 pipeline 分析）
- **计算操作**：`affine.addf`, `affine.mulf`, `affine.maxf` 等简单运算

**算法选择示例**：

```
Tensor IR: conv2d(input=1×3×224×224, filter=64×3×7×7, stride=2, pad=3)
                            │
                            ▼  算法选择（此层完成）
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  im2col + GEMM       Winograd F(2,3)        Direct Conv
  (通用，kernel≤7)     (kernel=3,stride=1)     (depthwise)
        │
        ▼
  Loop IR:
    // im2col 展开
    affine.for %n = 0 to 1
      affine.for %h = 0 to 112
        affine.for %w = 0 to 112
          // 将 3×7×7 邻域展开为列向量
          ...
    // GEMM
    affine.for %m = 0 to 64
      affine.for %n = 0 to 12544
        affine.for %k = 0 to 147
          %a = affine.load %A[%m, %k]
          %b = affine.load %B[%k, %n]
          %c = affine.load %C[%m, %n]
          %c = affine.addf %c, affine.mulf %a, %b
          affine.store %c → %C[%m, %n]
```

**本层的"不"**：
- 不做 tiling（循环切分由 Schedule IR 完成）
- 不做循环重排（由 Schedule IR 完成）
- 不做并行标注（由 Schedule IR 完成）
- 不做指令选择（由 Hardware IR 完成）

### 6.3 Schedule IR — "如何组织计算？"

**职责**：这是**整个框架最重要的层**。在 Loop IR 的基础上，加入执行策略：

- **Tile**：将大循环切分为适合硬件 cache/寄存器的小块
- **Reorder**：重排循环顺序以优化数据局部性
- **Pipeline**：插入软件流水线（双缓冲），隐藏访存延迟
- **Parallel**：标注可并行的循环维度
- **Unroll**：循环展开（包括完全展开和部分展开）
- **Vectorize**：标注可向量化的内层循环
- **MemSpace**：分配数据到内存层级（global/shared/register）

**Cost Model 在此层深度介入**：

```
Schedule IR 搜索流程:
┌────────────────────────────────────────────────────────┐
│                                                        │
│  Loop IR (三重循环 GEMM)                                │
│       │                                                │
│       ▼                                                │
│  生成候选 Schedule:                                     │
│    Candidate 1: tile(M=64,N=64,K=32), reorder(M,N,K)  │
│    Candidate 2: tile(M=128,N=64,K=16), reorder(N,M,K) │
│    Candidate 3: tile(M=32,N=128,K=64), reorder(K,M,N) │
│    ...  (共 N 个候选)                                   │
│       │                                                │
│       ▼                                                │
│  Cost Model 估算每个候选:                                │
│    C1: latency=120, mem_traffic=2.1MB, power=4.3W     │
│    C2: latency=90,  mem_traffic=1.8MB, power=3.9W  ✓  │
│    C3: latency=145, mem_traffic=2.5MB, power=4.8W     │
│       │                                                │
│       ▼                                                │
│  选择 Candidate 2，应用到 IR                            │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**搜索空间**：tile 大小、循环顺序、展开因子、pipeline 深度。这是一个组合优化问题。早期可以用启发式搜索（模拟退火/遗传算法），后期可接入 ML-based 搜索。

**Schedule IR 的输出**：

```
// 输入 Loop IR:
affine.for %m = 0 to 64
  affine.for %n = 0 to 12544
    affine.for %k = 0 to 147
      ...

// 输出 Schedule IR:
schedule.tile (%m, %n, %k) into (%m_tile=64, %n_tile=128, %k_tile=16)
schedule.reorder (%k_tile, %m, %n, %k_inner)
schedule.memspace %A_tile → shared, %B_tile → shared, %C_tile → register
schedule.pipeline depth=2 (%A_tile_load, %B_tile_load)
schedule.vectorize %n_inner
```

### 6.4 Hardware IR — "如何使用硬件？"

**职责**：将 Schedule IR 映射到**具体的硬件指令**。从 Hardware Description (YAML) 自动推导映射规则。

**映射生成逻辑**：

```
Schedule IR 操作          →    硬件能力 (来自 YAML)    →    Hardware IR 指令
═════════════════════          ════════════════════        ══════════════════
schedule.vectorize(dim)   +    arch: rvv, sew: 32     →    rvv.vle / rvv.vfmacc
                                                             / rvv.vse 序列

schedule.pipeline(depth)  +    dma: 2, sram: 128KB    →    双缓冲 load/store
                                                             交替指令序列

affine.load(stride=LD)    +    has_strided_load: true  →    rvv.vlse
                          +    has_strided_load: false →    rvv.vle + manual stride
                                                             (降级为多步操作)
```

**关键设计**：当硬件不支持某个特性时，自动降级。例如硬件没有 strided load，则用连续 load + rearrange 替代。

**与 cudnn-rvv-mlir 的关系**：cudnn-rvv-mlir 的 RVV dialect 是这个层的一个具体实现。短期可以复用 RVV dialect 作为 RVV 后端的 Hardware IR 实现，长期用 Hardware Description 自动生成。

### 6.5 LLVM IR — "如何编码？"

这层不再做任何架构决策。纯粹的编码转换：

```
RVV:  rvv.vfmacc %a, %b, %c  →  @llvm.riscv.vfmacc.nxv8f32(%a, %b, %c)
CUDA: tile_mma %a, %b, %c     →  @llvm.nvvm.mma.sync.aligned.m16n8k16...
x86:  vec.fma %a, %b, %c      →  <8 x float> @llvm.x86.avx512.vfmadd...
```

这一层占整体代码量的不到 20%。

---

## 7. 项目目录结构

```
OpenComputeFlow/
├── README.md
├── docs/
│   └── DESIGN_ZH.md                    ← 本文档
│
├── frontend/                           # 前端入口
│   ├── cudnn_frontend/                 # cuDNN Frontend API → Tensor IR
│   └── pytorch_frontend/               # PyTorch FX → Tensor IR (未来)
│
├── ir/                                 # 四层 IR 定义（核心模块）
│   ├── tensor_ir/                      # Tensor IR: 算子语义 + 数据依赖图
│   │   ├── TensorOps.td               # MLIR TableGen 定义
│   │   ├── TensorDialect.cpp
│   │   └── TensorTypes.cpp
│   ├── loop_ir/                        # Loop IR: Affine 循环 + 访存模式
│   │   ├── LoopOps.td
│   │   ├── LoopDialect.cpp
│   │   └── LoopAnalysis.cpp           # 依赖距离分析
│   ├── schedule_ir/                    # Schedule IR: Tile/Pipeline/Reorder
│   │   ├── ScheduleOps.td
│   │   ├── ScheduleDialect.cpp
│   │   └── ScheduleTransform.cpp      # 变换施加逻辑
│   └── hardware_ir/                    # Hardware IR: 硬件指令序列
│       ├── HardwareOps.td
│       └── HardwareDialect.cpp
│
├── transform/                          # Lowering Passes（将上层 IR 降为下层 IR）
│   ├── tensor_to_loop/                 # Tensor IR → Loop IR
│   │   ├── AlgorithmSelect.cpp        # 算法选择
│   │   ├── ConvToImplicitGemm.cpp     # 卷积展开
│   │   └── TensorToLoop.cpp
│   ├── loop_to_schedule/               # Loop IR → Schedule IR
│   │   ├── TileCandidateGen.cpp       # 候选生成
│   │   ├── ScheduleSelect.cpp         # Cost-Model 驱动选择
│   │   └── LoopToSchedule.cpp
│   ├── schedule_to_hardware/           # Schedule IR → Hardware IR
│   │   ├── InstructionSelect.cpp      # 从 Hardware Description 推导
│   │   └── ScheduleToHardware.cpp
│   ├── fusion/                         # 算子融合（Tensor IR 层）
│   │   └── FusionPlan.cpp
│   └── hardware_to_llvm/               # Hardware IR → LLVM IR
│       └── HardwareToLLVM.cpp
│
├── cost_model/                         # Cost Model（一等公民）
│   ├── CostEstimator.h                # API: estimate(ir, dims, hw) → {lat, bw, energy}
│   ├── CostEstimator.cpp
│   ├── MemoryModel.cpp                # 访存分析
│   ├── ComputeModel.cpp               # 计算分析
│   ├── MicroBenchmark.cpp             # 微基准数据采集
│   └── Roofline.cpp                   # Roofline 模型
│
├── hardware/                           # 硬件描述与后端
│   ├── descriptions/                   # YAML 硬件描述文件
│   │   ├── riscv_v_vector.yaml
│   │   ├── riscv_v_vector_vlen1024.yaml
│   │   ├── arm_neon_v8.yaml
│   │   ├── arm_sve.yaml
│   │   ├── x86_avx512.yaml
│   │   ├── nvidia_sm80.yaml
│   │   └── generic_systolic.yaml
│   ├── hw_parser/                      # YAML 解析 + 能力查询 API
│   │   ├── HardwareDescription.h
│   │   ├── HardwareDescription.cpp
│   │   └── InstructionDB.h            # 指令表：按硬件 + 操作查询
│   └── backends/                       # 各后端特化实现（如需要）
│       ├── riscv_vector/
│       ├── arm_neon/
│       └── cuda/
│
├── backend/
│   └── llvm/                            # LLVM IR 发射（薄层）
│       └── LLVMEmitter.cpp
│
├── visualizer/                          # IR 可视化（调试 + 研究用）
│   ├── IRGraphViewer.cpp              # 各层 IR 图形化
│   ├── LoweringTraceViewer.cpp        # Lowering 过程回放
│   └── CostLandscapeViewer.cpp        # Cost Model 搜索空间可视化
│
├── benchmark/                           # 性能基准
│   ├── kernels/                        # 标准测试 kernel
│   │   ├── gemm_bench.cpp
│   │   ├── conv2d_bench.cpp
│   │   └── attention_bench.cpp
│   └── runner/                         # 运行器
│       └── BenchmarkRunner.cpp
│
└── test/                                # 测试
    ├── lit/                             # MLIR LIT 测试
    │   ├── tensor_ir/
    │   ├── loop_ir/
    │   ├── schedule_ir/
    │   └── hardware_ir/
    └── unit/                            # C++ 单元测试
```

你会发现，LLVM 相关代码 (<10 个文件) 占比不到 20%。

---

## 8. 与 cudnn-rvv-mlir 的关系

```
OpenComputeFlow                        cudnn-rvv-mlir
═══════════════                        ═══════════════
                                       
Tensor IR (新)                          CDF Dialect (骨架)
    │                                       │
Loop IR (新)                             CIR Dialect (骨架)
    │                                       │
Schedule IR (新)                        (无对应，内嵌在 lowering 里)
    │                                       │
Hardware IR (新)    ───对接───→          RVV Dialect (成熟 ✓)
    │                                       │
LLVM IR                                  LLVM IR
```

**短期策略**：
- cudnn-rvv-mlir 的 RVV dialect + RVV→LLVM lowering 保持，继续完善
- OpenComputeFlow 的上层 IR (Tensor→Loop→Schedule→Hardware) 新开发
- Hardware IR 通过 adapter 对接 cudnn-rvv-mlir 的 RVV dialect

**长期策略**：
- 当 Hardware Description 框架成熟后，Hardware IR 可以替代 cudnn-rvv-mlir 的 RVV 方言
- cudnn-rvv-mlir 转为"单后端的成熟实现参考"，OpenComputeFlow 承接全部上层抽象

---

## 9. 路线图

### Phase 1：核心 IR 定义 + 单后端打通（当前 ~ 6 个月）

**目标**：Tensor IR → Loop IR → Schedule IR → Hardware IR (RVV) → LLVM IR → 跑出第一个 GEMM

- [ ] 定义 Tensor IR dialect（conv, matmul, elemwise, reduce）
- [ ] 实现 Tensor IR → Loop IR lowering（im2col+GEMM 路径）
- [ ] 定义 Loop IR dialect（affine.for, affine.load/store）
- [ ] 定义 Schedule IR dialect（tile, reorder, pipeline, vectorize, memspace）
- [ ] 实现简单的分析型 Cost Model（Roofline + 线性模型）
- [ ] 实现 Schedule IR → Hardware IR lowering（对接 cudnn-rvv-mlir 的 RVV dialect）
- [ ] 实现 Hardware IR → LLVM IR（复用 cudnn-rvv-mlir）
- [ ] **里程碑：GEMM 端到端跑通，Cost Model 能自动选择 tile 大小**

### Phase 2：Cost Model 强化 + 多算法支持（6-12 个月）

**目标**：Cost Model 能驱动算法选择，支持 Winograd

- [ ] Cost Model 接入 micro-benchmark 数据
- [ ] 实现 Winograd F(2,3) 算法路径（Loop IR 层）
- [ ] 实现算子融合 pass（CDF 层 conv+bias+act）
- [ ] Schedule IR 搜索空间扩展（支持更多循环重排策略）
- [ ] **里程碑：同一卷积，Cost Model 自动在 im2col 和 Winograd 之间选择**

### Phase 3：多硬件后端（12-18 个月）

**目标**：换 YAML 就能换硬件

- [ ] Hardware Description (YAML) 完整定义 + 解析器
- [ ] ARM NEON 后端（向量化 backend）
- [ ] x86 AVX-512 后端
- [ ] CUDA/NVPTX 后端（SIMT 模型）
- [ ] **里程碑：同一份 Tensor IR，三套 YAML，三个后端均能跑通 GEMM**

### Phase 4：可视化 + Benchmark 完善（18-24 个月）

**目标**：成为真正的"研究平台"

- [ ] IR 可视化工具（Lowering trace viewer）
- [ ] Cost landscape 可视化
- [ ] 标准 Benchmark 套件（GEMM, Conv2D, Attention, MoE）
- [ ] 与 hand-tuned kernel 的性能对比报告
- [ ] **里程碑：能向社区展示一份完整的"从 API 到汇编"的 lowering 追踪**

---

## 10. 设计哲学总结

### 10.1 这个项目的核心不是 LLVM

LLVM 只占项目不到 20% 的代码量。OpenComputeFlow 的核心价值在于：

1. **IR 分层的边界定义** — 为什么这样分，而不是那样分
2. **Cost Model 驱动的决策** — 不让 lowering 是硬编码规则
3. **Hardware Description 的解耦** — 算法与硬件描述的分离
4. **Lowering 过程的可解释性** — 每一步都知道"为什么这样做"

### 10.2 判断一个设计决策是否正确的标准

每当你考虑某个模块应该放在哪一层时，问自己：

> **这个信息在这一层能回答什么问题？这个问题的答案在下层还有意义吗？**

如果一个问题在某一层回答之后，下层不需要再重新思考——抽象边界就是对的。如果下层仍然需要携带这个信息做二次决策，说明抽象边界需要调整。

### 10.3 这个项目证明的不是"我会 MLIR"

而是：

> **"我理解计算是如何跨越 API、IR、调度、硬件，一步步落地执行的；我还能设计这条路径中的抽象和边界。"**

对于 AI 芯片架构方向来说，这种能力比"会用一个框架"重要得多。

---

## 附录 A：与 MLIR 上游组件的关系

OpenComputeFlow 构建在 MLIR 之上，但有自己的设计哲学：

| 组件 | 使用 MLIR 上游？ | 备注 |
|---|---|---|
| Tensor IR | 自研 dialect | 比 linalg 更高层，带 DNN 专用语义 |
| Loop IR | 基于 `affine` dialect 扩展 | 复用上游 loop 基础设施 |
| Schedule IR | 自研 dialect | 上游无对应。这是核心创新层 |
| Hardware IR | 自研 dialect | 上游无对应。类似 cudnn-rvv-mlir 但硬件无关 |
| Cost Model | 自研 | 上游无完整的 Cost Model 框架 |
| LLVM IR | 100% 上游 | 不做任何修改 |
| Visualizer | 自研 | 研究和调试工具 |

---

## 附录 B：为什么"不绑定 RISC-V 但是用 MLIR"

MLIR 是基础设施，不是目标。它提供了：

- 方言定义框架（TableGen）
- Pass 管理器和 Pattern Rewriting
- 内置的 `func`/`arith`/`scf`/`affine`/`memref` 等通用方言
- LIT 测试框架

但这些是工具，不是项目的价值所在。项目的价值在于**在这些工具之上构建的抽象层次和决策逻辑**。RISC-V 只是一个后端目标——换一个 YAML，就可以换一个目标。
