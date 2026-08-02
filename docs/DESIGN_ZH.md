# OpenComputeFlow 架构设计文档

> **An Experimental AI Compute Architecture Framework**
>
> *Exploring explainable and measurable lowering from DNN semantics to hardware execution.*

本文描述 OpenComputeFlow 的目标架构与演进约束。仓库当前处于设计阶段，文中的目录、方言和接口均是提案；只有通过相应阶段的验收条件后，才能视为已实现能力。

---

## 1. 项目定位

### 1.1 要解决的问题

OpenComputeFlow 研究 DNN 计算如何从高层算子语义逐步映射到具体硬件。重点不只是“能否生成代码”，而是让每个关键决策都具备：

- **明确的语义边界**：该层保留什么信息、消除什么信息
- **可验证的合法性**：变换前后满足哪些不变量
- **可解释的选择依据**：为何选择某个算法、布局或调度
- **可复现的结果**：使用哪个目标描述、Cost Model 和候选集合
- **可度量的效果**：预测值与实测值之间的误差是多少

一个典型问题是：前端表达的卷积、矩阵乘和归约，如何在保持数值语义的前提下，经过算法选择、循环组织、内存规划和目标合法化，最终变成可执行代码。

LLVM/MLIR 是实现基础设施，不是项目要重新实现的部分。项目的研究价值主要位于高层语义、调度搜索、代价建模、目标能力建模和 lowering 可解释性。

### 1.2 长期目标

OpenComputeFlow 希望成为 AI 编译与芯片架构协同研究的实验平台：

- 定义和比较不同的计算抽象，验证其边界是否稳定
- 插入不同 Cost Model，比较搜索质量、预测误差和编译开销
- 在同一计算语义下比较不同算法、布局、调度和硬件映射
- 复用统一的分析与搜索框架，同时允许后端表达各自执行模型
- 记录完整 lowering trace，解释每个候选被接受或拒绝的原因

### 1.3 非目标

以下内容不作为早期目标：

- 替代 MLIR、LLVM 的通用优化、指令选择或机器码生成能力
- 仅凭一个 YAML 文件支持一种全新的 ISA 或执行模型
- 在第一个版本覆盖全部 DNN 算子、动态控制流、分布式执行和训练
- 保证 Cost Model 第一次预测就等价于真实硬件测量
- 用自研方言重复实现 linalg、scf、affine、vector、memref 已有能力

### 1.4 成功标准

项目是否成功不能用方言数量或代码量衡量。每个端到端样例至少应回答：

1. 输入计算的形状、布局、数据类型和数值语义是什么？
2. 生成了哪些合法候选，哪些候选因何被淘汰？
3. 被选候选的预测代价、置信度和选择目标是什么？
4. 每层 IR 是否通过 verifier，lowering 后是否保持语义？
5. 目标代码是否能执行，结果是否通过参考实现校验？
6. 预测与实测的误差是多少，结果能否用同一配置复现？

---

## 2. 核心理念：Progressive Lowering

### 2.1 分层原则

分层的目的不是强制创建五个自研 Dialect，而是隔离五类不同问题：

| 阶段 | 回答的问题 | 主要保留的信息 | 主要消除的信息 |
|---|---|---|---|
| Tensor IR | 计算语义是什么？ | 算子、张量、形状、布局约束、数值策略 | 前端 API 对象和调用细节 |
| Compute/Loop IR | 计算如何展开？ | 迭代域、索引映射、归约、访存关系 | DNN 算子名称或已选算法的高层封装 |
| Schedule Plan | 计算如何组织？ | tile、融合、并行、向量化、内存层级等决策 | 未选择的候选 |
| Hardware IR | 如何合法使用目标能力？ | 目标操作、同步、掩码、DMA/线程/向量语义 | 与目标无关的调度意图 |
| Backend/Runtime | 如何形成可调用程序？ | ABI、目标特征、地址空间、运行时调用、目标代码 | 编译期 IR |

“一层只降低一个维度”是指导原则，不是绝对禁令。目标信息可以参与高层候选选择，但不应泄漏为高层 IR 的指令级语义。例如 Tensor IR 可以知道目标偏好 NHWC，却不应出现 vfmacc。

### 2.2 为什么区分 Tensor、Compute 和 Schedule

**Tensor IR 与 Compute/Loop IR 分开**，因为前者表达领域语义和图关系，后者表达一种具体算法的迭代与数据访问。同一个 conv2d 可以对应 direct convolution、implicit GEMM 或 Winograd；一旦展开为计算域，高层算子语义通常不可完整恢复。

**Schedule Plan 与 Compute/Loop IR 分开**，因为计算定义和执行策略具有不同生命周期。一个计算定义可以生成多个 schedule 候选，Cost Model 需要在不复制或破坏原始 IR 的情况下评估这些候选。

Schedule Plan 不是“另一份已经变形的循环”。它是作用于 payload IR 的变换程序或参数化计划。计划被选择并应用后，产生显式的 scheduled payload IR，后续 Hardware lowering 消费的是后者。

建议优先复用 MLIR Transform Dialect 表达和执行计划，仅在上游接口不足时扩展自定义 transform op。这样可以避免重新设计 handle 生命周期、变换组合和失败传播。

### 2.3 统一路径的边界

目标主路径为：

~~~text
Frontend Graph
    -> Tensor IR
    -> algorithm/decomposition candidates
    -> Compute IR
    -> schedule candidates
    -> Scheduled Compute IR
    -> Target-specific Hardware IR
    -> Backend IR + Runtime ABI
    -> Executable
~~~

Conv2D、Matmul、Reduce 和部分 Attention 子图可在 Compute IR 汇合为结构化迭代、归约和数据搬运。但“所有 DNN 算子在 Loop IR 以下完全相同”并不成立，至少以下语义需要专门建模或推迟支持：

- 稀疏或间接索引
- 数据依赖控制流和动态序列长度
- 随机数、有状态算子和副作用
- 跨设备通信与 collective
- 异步执行、DMA 和显式同步
- 具有特殊数值要求的归约、量化和 transcendental 运算

统一的是分析和决策框架，不是强行把不同执行模型压缩成同一组指令。

---

## 3. 总体架构

~~~text
 cuDNN Frontend graph / PyTorch export / ONNX / textual MLIR
                              |
                              v
 +------------------------------------------------------------------+
 | Tensor IR                                                        |
 | DNN semantics, shape/layout constraints, dtype, numerical policy |
 +------------------------------+-----------------------------------+
                                | generate legal decompositions
                                v
 +------------------------------------------------------------------+
 | Compute IR                                                       |
 | linalg/scf/affine/arith/tensor or a thin project abstraction     |
 +------------------------------+-----------------------------------+
                                | generate -> check -> estimate
                                v
 +--------------------------+   selected plan   +--------------------+
 | Schedule Plan            | ----------------> | Scheduled Compute |
 | Transform IR + metadata  |                   | explicit loops,    |
 +--------------------------+                   | copies and vectors |
                                                +---------+----------+
                                                          |
                       +----------------------------------+-----------+
                       | Target profile + backend plugin              |
                       | capabilities, constraints, calibrated costs |
                       +----------------------------------+-----------+
                                                          |
                                                          v
 +------------------------------------------------------------------+
 | Hardware IR                                                      |
 | vector/RVV, NVVM, target dialect, sync and memory-space semantics|
 +------------------------------+-----------------------------------+
                                |
                                v
 +------------------------------------------------------------------+
 | Backend + Runtime                                                |
 | LLVM dialect/IR or another backend, ABI, allocation, launch      |
 +------------------------------+-----------------------------------+
                                |
                                v
                  object / executable / device binary
~~~

### 3.1 数据面与决策面

架构分成两条相互关联但不混合的路径：

- **数据面**：承载计算语义并逐层 lowering 的 payload IR
- **决策面**：生成候选、检查合法性、估算代价、选择并记录决策的 Schedule/Cost Model

Cost Model 不直接修改 payload IR。它只对稳定的候选描述返回评估结果；变换执行器负责应用被选计划，verifier 负责验证结果。

### 3.2 每个阶段的契约

每个 lowering pass 必须声明：

- 接受哪些 Dialect、op、类型和动态形状
- 产出 IR 的合法 Dialect 集合
- 需要哪些目标能力和数值语义前提
- 保持哪些属性，允许放宽哪些属性
- 失败是“候选不合法”“目标不支持”还是“编译器缺陷”
- 是否存在语义等价的 fallback

Dialect Conversion 的 legality 定义应成为实现契约的一部分。Pass 成功后不得静默残留未声明的非法 op。

### 3.3 Lowering trace

每次编译生成机器可读的 trace，至少包含：

~~~yaml
pipeline_version: 1
input_fingerprint: "..."
target_profile: "rvv-example@1"
backend_version: "..."
cost_model:
  id: "analytical-v1"
  calibration_id: "..."
decisions:
  - site: "conv2d_0"
    candidates_generated: 12
    candidates_legal: 7
    selected: "tile_oc8_oh4_ow16"
    objective: {latency_us: 41.2}
    rejected:
      - id: "tile_oc32_oh8_ow32"
        reason: "working_set_exceeds_l1"
~~~

Trace 是调试、可视化和复现实验的共同数据源，不应依赖解析日志文本。

---

## 4. 各层详细设计

### 4.1 Tensor IR：“计算语义是什么？”

Tensor IR 表达前端无关的 DNN 语义和图结构。早期可自定义薄层 Dialect，但应明确与 tensor、linalg 或其他上游 Dialect 的转换边界。

**必须表达的内容**：

- 算子语义及版本，如 matmul、conv2d、reduce、elementwise
- ranked tensor、静态/动态维度和形状约束
- 逻辑布局与物理布局；二者不能只用一个 NCHW/NHWC 字符串代替
- 输入、输出、常量、可变状态和 alias/side-effect 信息
- 存储类型、计算类型和累加类型
- broadcasting、padding、group、stride、dilation 等算子属性
- 量化参数的作用域、zero point、scale 和饱和/舍入规则
- fast-math、NaN/Inf、确定性和允许误差等数值策略

**本层变换**：

- 形状推断、类型检查和约束传播
- 常量折叠、canonicalization、CSE 等可复用上游优化
- 语义合法的算子融合候选生成
- layout 候选生成与 layout propagation
- 算法 decomposition 候选生成

融合和 layout 可以参考目标与 Cost Model，但选择结果必须保持 Tensor IR 声明的数值和副作用语义。目标相关的偏好不等于目标指令进入 Tensor IR。

**阶段不变量**：

- 每个 op 的 shape/type verifier 均通过
- 动态维度的等式或范围约束没有被无依据地静态化
- 广播、alias 和副作用顺序明确
- 数值策略可沿 lowering 传递，不能在后续层丢失

### 4.2 Compute/Loop IR：“计算如何展开？”

Compute IR 描述选定算法的结构化计算，包括迭代域、索引映射、归约和数据访问。初期不建议创建一套重复 affine.for/load/store 的 Loop Dialect，应组合使用：

- linalg：结构化计算和可变换的迭代语义
- tensor：bufferization 前的值语义
- scf：通用循环、动态边界和控制流
- affine：只用于已证明满足 affine 约束的循环和索引
- arith、math：标量计算
- memref：bufferization 后的显式内存访问

affine 不能表达一般的间接访问，浮点运算也应使用 arith.addf、arith.mulf 等，而不是不存在的 affine.addf。动态形状或非 affine 索引应保留在 scf/linalg，不能为了统一形式强行改写。

算法选择发生在 Tensor IR 到 Compute IR 的边界：

~~~text
tensor.conv2d
    |
    +-- direct convolution --------> Compute candidate A
    +-- implicit GEMM -------------> Compute candidate B
    +-- Winograd (条件满足时) ------> Compute candidate C
~~~

候选生成器负责给出适用条件；合法性检查器先排除不满足 kernel、stride、dtype、workspace 或数值要求的候选；Cost Model 只比较合法候选。算法名及来源要保留在 trace 中，但无须作为低层执行语义永久存在。

该图描述目标架构的算法候选机制；Phase 1 的 Conv2D MVP 只实例化 direct convolution，implicit GEMM 和 Winograd 不进入首阶段搜索空间。

**阶段不变量**：

- 迭代域覆盖完整且无非预期重叠
- 归约 identity、结合/交换假设和累加类型明确
- 边界、padding 和 tail 语义明确
- 读写集合与 alias 分析一致
- 尚未做目标指令选择

### 4.3 Schedule Plan：“计算如何组织？”

Schedule Plan 表达对 Compute IR 的变换意图：

- tile、split、fuse、reorder
- parallel、map-to-thread/block
- vectorize、unroll
- promotion、bufferization 和 memory-space placement
- software pipeline、prefetch、double buffering
- producer-consumer fusion 与计算位置

Plan 应通过稳定 handle 引用 payload IR，并定义 handle 失效和失败传播规则。优先采用 MLIR Transform Dialect；项目自定义内容作为 Transform Dialect extension，而不是独立且互不兼容的调度系统。

示意：

~~~text
Compute payload: structured direct-conv(...)

Transform plan:
  match direct-conv
  tile [oc=8, oh=4, ow=16]
  promote input/filter tiles when legal
  vectorize output-width or output-channel dimension
  apply canonicalization

Materialized payload:
  explicit tiled loops + subviews + copies + vector operations
~~~

Plan 本身不是 Hardware IR。应用计划后必须重新执行 verifier、内存容量检查和 Dialect legality 检查。若计划应用失败，该候选被标记为无效，不允许留下半变换状态继续编译。

**候选生成顺序**：

~~~text
Generate
  -> symbolic legality and shape guards
  -> resource feasibility
  -> analytical estimate
  -> optional measured/learned refinement
  -> choose
  -> materialize on a fresh or rollback-capable payload
  -> verify
~~~

MVP 应从规则约束下的枚举或 beam search 开始。模拟退火、遗传算法和 ML-based search 只有在搜索空间、基准集和编译预算明确后再引入。

### 4.4 Hardware IR：“如何合法使用目标能力？”

Hardware IR 表达目标族特有且不能由通用 vector/gpu 等 Dialect 完整承载的语义，例如：

- RVV 的 scalable vector、VL、mask 和 tail policy
- SIMT 的线程层级、barrier、address space 和 warp primitive
- 矩阵加速器的 MMA shape、fragment layout 和 accumulator 约束
- NPU 的 DMA、片上存储、事件和异步依赖

Hardware Description 提供能力和约束，**backend plugin 提供语义 lowering**。两者缺一不可：

~~~text
Scheduled Compute IR
    + Target Profile (declarative facts)
    + Backend Plugin (rewrite/legalization code)
    -> Hardware IR
~~~

当目标缺少某项能力时，处理顺序为：

1. 使用已注册、经过验证的等价 expansion
2. 回退到更通用的 IR 层并重新调度
3. 若无合法路径，返回带上下文的“不支持”诊断

不能根据 YAML 中的指令字符串猜测语义，也不能默认“连续 load + rearrange”总能等价替代 strided/gather 访问。

### 4.5 Backend 与 Runtime：“如何形成可调用程序？”

这一步不仅是机械编码，还包括：

- MLIR Dialect 到 LLVM Dialect/LLVM IR、NVVM 或其他目标 IR 的转换
- 数据布局、地址空间和 calling convention
- memref/tensor descriptor ABI
- host/device 边界、kernel launch 和同步
- workspace 查询、分配与释放
- 目标特征、链接、对象文件和可执行文件生成
- 必要的 runtime library 调用

寄存器分配和最终机器指令选择通常由 LLVM 等后端完成，不属于“LLVM IR 本身”。OpenComputeFlow 应尽量复用上游能力，但必须测试本项目产生的 ABI 和目标特征是否正确。

---

## 5. Cost Model 与搜索

### 5.1 定位

Cost Model 是决策引擎，不是正确性裁判。正确性和资源合法性由 verifier、约束检查和 backend legality 保证；Cost Model 只在合法候选之间排序。

它也不是对实测调优的替代。合理闭环是：

~~~text
Generate -> Check -> Estimate -> Choose -> Lower -> Verify
                                      |
                                      +-> optional Measure -> Calibrate
~~~

### 5.2 输入与输出

**输入**：

- 标准化 problem signature：shape、dtype、layout、数值策略
- 候选的结构化特征：tile、循环序、访存、并行度、向量宽度
- target profile 与 backend 版本
- 资源使用估算：寄存器、片上内存、workspace、代码尺寸
- 编译模式和预算：AOT/JIT、最大搜索时间、是否允许实测

**输出**：

~~~text
Estimate {
  status: ok | unavailable
  objectives: {
    latency_us,
    throughput_items_per_s,
    energy_mj,
    code_size_bytes,
    workspace_bytes
  }
  uncertainty: {kind, value}
  assumptions: [...]
  model_id
  calibration_id
}
~~~

status 只表示模型能否对已经合法的候选给出估价，不重新裁定候选合法性。所有指标必须带单位。功耗、能耗和吞吐不能混为单个无量纲分数。

### 5.3 目标与约束

默认优化目标应由用户或部署配置明确指定，例如：

- 最小化 p50 latency，约束 workspace <= 64 MiB
- 最大化 throughput，约束功耗和数值误差
- 在 latency 与 energy 的 Pareto frontier 中选择

若多个候选预测差异小于模型误差或置信区间，应使用稳定 tie-breaker，例如更小 workspace、更少代码尺寸或更通用的 shape coverage，而不是宣称某候选确定最优。

### 5.4 分层决策

| 决策点 | 典型候选 | 必要的先验检查 |
|---|---|---|
| Tensor | fusion、layout、decomposition | shape、数值语义、副作用、workspace |
| Compute | direct/implicit GEMM/Winograd | 算法适用条件与参考等价性 |
| Schedule | tile、reorder、vectorize、pipeline | 依赖、容量、寄存器、并行合法性 |
| Hardware | 已注册 expansion/目标 op 选择 | target capability、类型和 mask/sync 语义 |

不要让同一决策在多层被独立做两次。上层选定的决定应以结构化 provenance 传入下层；若下层发现不可满足，应触发显式回退或重新搜索。

### 5.5 模型组成

~~~text
                         +-------------------+
Candidate features ----> | Cost Model API    | ----> estimate + uncertainty
Target profile --------> | versioned         |
                         +---------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                    |
       Analytical Model     Calibration DB      Learned Model
       compute/memory       microbenchmarks      optional later
~~~

早期采用可解释的分析模型：

- Roofline 只用于粗粒度上界，不能替代 cache、pipeline 和 occupancy 模型
- Compute Model 估算关键资源吞吐与依赖链
- Memory Model 估算各层级流量、重用、对齐和冲突
- Overhead Model 覆盖 launch、同步、循环尾部和 setup 开销

### 5.6 校准、缓存与在线选择

校准数据必须绑定：

- device/CPU 型号与可影响性能的固件或微架构标识
- ISA feature、频率策略、线程数和内存配置
- compiler/backend 版本与关键 flags
- microbenchmark schema 和采集时间

搜索结果缓存键至少包含 problem signature、动态 shape bucket、目标指纹、数值策略、pipeline 版本和 model/calibration ID。

对于动态形状，支持三种策略：

1. 生成覆盖合法范围的通用 schedule
2. 按 shape bucket 生成多个版本并在运行时 dispatch
3. 在 JIT 模式为新 shape 编译并缓存

任何 runtime dispatch 都必须有覆盖全部合法输入的 fallback。

---

## 6. Hardware Description 与后端插件

### 6.1 Hardware Description 的职责

Hardware Description 是版本化的 **Target Profile**，用于描述编译器可查询的事实：

- ISA/执行模型和 feature
- 数据类型、向量或 MMA shape 支持
- 内存层级、容量、对齐、地址空间和 DMA 能力
- 线程/核心拓扑与同步能力
- backend 已注册的 lowering capability
- 可选的性能参数与 calibration 引用

它不负责定义指令语义、任意 rewrite 模板或 LLVM intrinsic 名称。语义 lowering 必须由经过测试的 backend plugin 实现。

### 6.2 示例 Schema

以下是说明性示例，不是最终 schema：

~~~yaml
schema_version: 1
target_id: "rvv-example"
profile_version: 1
backend: "rvv"

features:
  isa: "riscv64"
  extensions: ["v", "f", "d"]
  vector:
    scalable: true
    vlen_bits:
      mode: runtime
      minimum: 128
    element_widths: [8, 16, 32, 64]
    lmul: ["mf8", "mf4", "mf2", "m1", "m2", "m4", "m8"]
    supports_masked_ops: true

memory:
  cache_line_bytes: 64
  levels:
    - {name: "l1d", kind: "cache", size_bytes: 32768}
    - {name: "l2", kind: "cache", size_bytes: 1048576}
    - {name: "dram", kind: "external"}
  minimum_alignment_bytes: 16

capabilities:
  - "vector.load.unit_stride"
  - "vector.load.strided"
  - "vector.fma.f32"
  - "vector.reduce.add.f32"

calibration:
  id: "rvv-example-latency-v1"
~~~

RVV 的 VLEN 可以是运行时相关的 scalable 属性，不能无条件把 vlen: 256 当成所有目标的固定编译期常量。CPU cache 也不应被描述成 SIMT/NPU 风格的 shared memory，除非目标确实暴露了可显式管理的 SRAM。

### 6.3 Backend Plugin 接口

每个目标族至少实现：

~~~text
BackendPlugin {
  id/version
  validateTargetProfile(profile)
  queryCapabilities()
  checkCandidate(candidate, profile)
  populateLoweringPatterns()
  populateBackendConversion()
  getRuntimeABI()
}
~~~

Target Profile 变化分为两类：

- **同一 backend family 内的变体**：例如不同 VLEN 下限、cache 容量或 MMA shape，通常可以通过替换 profile 适配
- **新的执行模型或 ISA**：例如从 RVV 转到 SIMT，必须增加或复用对应 backend plugin

因此项目承诺应是“在已支持的 backend family 内用 Target Profile 描述硬件变体”，而不是“换 YAML 即自动获得任意新后端”。

### 6.4 Schema 治理

- schema 必须有版本号和 JSON Schema/等价机器验证
- 未知必填字段、单位错误和互相矛盾的 capability 必须报错
- 性能提示与 correctness capability 分离，缺失性能数据不得改变合法性
- backend 启动时校验 profile，不能在 lowering 中延迟暴露配置错误
- profile 和 calibration 文件进入 lowering trace 的内容哈希

---

## 7. 正确性、失败处理与验证

### 7.1 数值语义

每个 Tensor op 都必须声明或继承：

- 输入/输出存储类型与内部计算、累加类型
- 浮点 contraction、reassociation 和 fast-math 策略
- NaN/Inf、signed zero、溢出和饱和规则
- reduction 顺序是否允许变化
- 是否要求 bit-exact、确定性或误差容限

算法和 schedule 候选只有在满足该策略时才合法。Winograd、低精度累加、FMA contraction 和并行 reduction 不能仅凭性能收益自动启用。

### 7.2 动态形状与边界

- 静态、符号和运行时维度使用统一 shape constraint 表达
- tile 不能整除时必须定义 remainder loop、mask 或 padding 策略
- vector tail 的 inactive lane 行为必须与目标 policy 一致
- 运行时 guard 应显式进入 IR 和 trace
- 零维、空 tensor、极小 shape 和超大 shape 必须有定义行为

Phase 1 可以限制输入范围，但必须由 verifier 或入口检查明确拒绝范围外输入，不能产生未定义代码。

### 7.3 Bufferization、内存与 Alias

设计必须明确：

- tensor value semantics 何时转换成 memref/buffer
- buffer ownership、生命周期和释放责任
- in-place bufferization 的 alias 前提
- workspace 的大小、对齐、地址空间和查询接口
- host/device 传输、异步 copy 和同步依赖
- promotion 到快存储失败时的 fallback

内存空间分配不是单纯 schedule 标签。它与 runtime allocation、地址空间转换、同步和容量检查共同构成可执行契约。

### 7.4 失败分类

| 类别 | 示例 | 处理 |
|---|---|---|
| 输入无效 | shape/type/attribute 不满足 op 语义 | 前端诊断并停止 |
| 候选不合法 | tile 超容量、依赖阻止并行 | 淘汰候选并记录原因 |
| 目标不支持 | 缺少 dtype 或同步能力 | 尝试已验证 fallback，否则报错 |
| 模型不可用 | 无 calibration、模型拒绝 shape | 使用保守模型/默认 schedule 并告警 |
| 编译器缺陷 | pass 成功后残留非法 op | 失败并输出最小化所需上下文 |

禁止静默改变 dtype、数值策略或结果布局以使 lowering 通过。

### 7.5 测试矩阵

| 层次 | 测试方法 | 核心断言 |
|---|---|---|
| Dialect | parser/printer、verifier 单测 | round-trip 与非法输入诊断 |
| Pass | LIT/FileCheck、pass failure test | 结构、legality 和不变量 |
| 语义 | 与朴素参考实现 differential test | dtype/shape/边界/误差 |
| Backend | LLVM verifier、目标汇编检查 | intrinsic、ABI、target feature |
| 运行时 | emulator/真实设备端到端 | 结果、workspace、同步和错误 |
| Cost Model | 预测对实测数据集 | 误差分布、排序准确率、回归 |
| 性能 | 固定环境 benchmark | 相对 baseline、方差和显著性 |

随机测试应记录 seed；性能测试应记录 warm-up、采样次数、频率策略和环境指纹。

### 7.6 可复现性

一个可复现实验包至少包含：

- 输入 IR 或其内容哈希
- pipeline 配置和 pass 参数
- target profile 与 calibration 哈希
- 编译器、LLVM/MLIR、backend plugin 版本
- 候选集合、选择结果和随机 seed
- 目标代码哈希
- 参考结果、实测数据与环境信息

---

## 8. Runtime 与部署模型

### 8.1 编译模式

- **AOT**：为约定 shape 或 shape bucket 生成目标文件/设备二进制
- **JIT**：根据运行时 shape 和设备 profile 编译并缓存
- **Hybrid**：AOT 通用 fallback + JIT/离线调优的专用版本

Phase 1 只要求 AOT，但公共接口不能假定所有 shape 都是编译期常量。

### 8.2 Kernel ABI

最小 ABI 需要规定：

- 参数顺序、标量宽度和 calling convention
- tensor/memref descriptor 的 layout
- shape、stride、offset 和 alignment 的传递方式
- workspace 查询与传入方式
- 状态码和诊断信息
- 线程、stream/context 或设备句柄

ABI 必须版本化，并有 C/C++ 侧端到端测试。不能只验证生成的 LLVM IR 能通过 parser。

### 8.3 编译缓存与 Dispatch

缓存键和 runtime guard 必须与 Cost Model 使用的 shape bucket、target fingerprint 和数值策略一致。加载缓存项时重新验证 ABI 版本和目标兼容性；不兼容项视为 miss，不能继续执行。

---

## 9. 建议的项目目录

目录按“语义、变换、决策、目标、运行时、验证”划分。下面是提议结构，不代表文件已经存在：

~~~text
OpenComputeFlow/
├── include/opencomputeflow/
│   ├── Dialect/
│   │   └── Tensor/
│   ├── Transform/
│   ├── CostModel/
│   ├── Target/
│   └── Runtime/
├── lib/
│   ├── Dialect/
│   │   └── Tensor/
│   ├── Transform/
│   │   ├── Decompose/
│   │   ├── Schedule/
│   │   └── Lowering/
│   ├── CostModel/
│   ├── Target/
│   │   ├── RVV/
│   │   └── Profiles/
│   ├── Runtime/
│   └── Trace/
├── tools/
│   ├── ocf-opt/
│   ├── ocf-compile/
│   └── ocf-benchmark/
├── runtime/
├── test/
│   ├── Dialect/
│   ├── Transform/
│   ├── CostModel/
│   ├── Target/
│   └── E2E/
├── benchmark/
├── docs/
└── cmake/
~~~

设计取舍：

- 初期仅在确有领域语义时创建 Tensor Dialect
- Compute IR 优先复用上游 Dialect，不预设独立 Loop Dialect
- Schedule 基于 Transform Dialect extension
- Hardware Dialect 按目标族组织，不制造“硬件无关的具体指令”这一矛盾概念
- backend plugin、Target Profile 和 Runtime ABI 放在同一目标契约下测试

---

## 10. 与 cudnn-rvv-mlir 的关系

当前相邻 cudnn-rvv-mlir 工程可提供 RVV Dialect 与 RVV 到 LLVM 的已有实现；CDF、CIR 及其 lowering 仍处于骨架或未接入构建状态。因此集成策略应基于明确接口，而不是假定所有层已经成熟。

~~~text
OpenComputeFlow                         cudnn-rvv-mlir

Tensor/Compute/Schedule
        |
Scheduled vector/target intent
        |
RVV backend adapter  ---------------->  RVV Dialect
                                          |
                                          v
                                      RVV -> LLVM
~~~

**短期策略**：

- 将 cudnn-rvv-mlir 的 RVV 路径视为可复用 backend 候选
- 先建立 adapter 与版本化接口，不复制 CDF/CIR 骨架
- 用端到端测试验证 RVV op、scalable vector、mask/VL 和 ABI 语义
- 在依赖方式确定前，避免直接引用其内部 C++ 类型作为公共 API

**长期策略**：

- 上游 vector/LLVM 能直接表达的能力优先复用上游
- 目标特有且稳定的语义保留在 RVV backend plugin
- Hardware Description 负责选择和约束 backend，不替代 Dialect 的语义定义
- 是否合并、依赖或淘汰 cudnn-rvv-mlir，由维护成本、测试覆盖和上游兼容性决定，不预先写死

---

## 11. 路线图与验收门槛

路线图采用能力门槛，不使用缺少人力和硬件前提的固定月份承诺。

### MVP Conv2D 契约

第一个端到端闭环以 Conv2D 为唯一领域算子，并主动限制语义范围：

| 项目 | MVP 约束 |
|---|---|
| 模式 | forward inference |
| 数据类型 | input、filter、output 和 accumulator 均为 f32 |
| 布局 | input/output 为连续 NCHW，filter 为连续 OIHW |
| Tensor shape | input=[N,C,H,W]、filter=[OC,C,KH,KW]、output=[N,OC,OH,OW] |
| 形状约束 | N、C、H、W、OC、KH、KW 均为正的编译期常量 |
| 参数 | groups=1、dilation=1；stride_h/stride_w 和四侧 padding 为编译期常量 |
| 算法 | 仅 direct convolution |
| 融合 | 不含 bias、activation 或其他 post-op |
| 执行 | AOT、单线程、单 RVV 目标 |

输出空间维度遵循：

~~~text
OH = floor((H + pad_top + pad_bottom - KH) / stride_h) + 1
OW = floor((W + pad_left + pad_right - KW) / stride_w) + 1

Y[n, oc, oh, ow] =
  sum(ic, kh, kw,
      padded_X[n, ic, oh * stride_h + kh, ow * stride_w + kw]
      * filter[oc, ic, kh, kw])
~~~

这里采用深度学习框架常见的 cross-correlation 语义，filter 不翻转，归约初值为 f32 零。Tensor verifier 必须检查 stride 为正、padding 非负、输入通道与 filter 通道一致，并且 OH、OW 为正。padding 区域按零值处理。算法候选、动态 shape、group/depthwise、dilation、量化和融合留到后续阶段。

### Phase 0：契约与基础设施

**范围**：

- 明确支持的 MLIR/LLVM 版本和依赖方式
- 定义最小 Tensor op：conv2d，并实现 MVP 契约 verifier
- 确定 Compute IR 复用 linalg/scf/vector 的路径
- 实现 Target Profile schema、解析和校验
- 定义 lowering trace schema
- 建立 parser/verifier/LIT 和 C++ 单测框架

**验收**：

- conv2d Tensor IR 可 parse/print round-trip
- 非法 shape、layout、dtype 和卷积参数有稳定诊断
- 示例 Target Profile 可通过 schema 与 backend 校验
- CI 能运行基础测试

### Phase 1：单后端最小闭环

**范围**：

- f32 direct Conv2D：Tensor -> Linalg/Compute -> Schedule -> RVV -> LLVM
- 一个明确的 RVV target profile
- AOT kernel ABI 和 host runner
- baseline schedule + 至少两组合法的 OC/OH/OW tile 候选
- 静态 shape 与非整除 tail；动态 shape 在入口明确拒绝
- 分析型 Cost Model 与结构化 trace

**验收**：

- 在 emulator 或指定 RVV 环境执行正确
- 至少覆盖 3x3/stride1/pad1、7x7/stride2/pad3，以及不能整除 tile 或 VL 的输出宽度
- 对零维、通道不匹配、非正输出维度、非法 rank/layout/dtype 明确拒绝
- 与朴素 direct-conv 参考实现在声明误差内一致
- Cost Model 能输出候选、单位、选择原因和预测误差
- 从输入、配置到目标代码和实测结果可复现

### Phase 2：稳健性与多算法卷积

**范围**：

- 动态 shape bucket、runtime guard 和缓存
- microbenchmark calibration 与模型误差报告
- 增加 implicit GEMM，与 MVP direct convolution 形成两条合法路径
- fusion/layout 候选与 bufferization/workspace 契约
- 失败回退和诊断完善

**验收**：

- Cost Model 只在满足数值、workspace 和 target 约束的算法间选择
- 未见过的 shape 落到通用 fallback
- 预测排序在固定验证集上优于规则 baseline
- 缓存失配和 ABI 不兼容不会执行旧代码

Winograd 在 Phase 2 后单独立项；先完成适用条件、数值误差和 workspace 评估，再进入候选集合。

### Phase 3：第二个后端族

**范围**：

- 优先增加另一个 CPU vector backend（如 x86 或 Arm），验证通用 Compute/Schedule 抽象
- 分离共用 vector lowering 与目标专用 legality
- 比较同一 Tensor IR 在两个 target profile 上的不同 schedule

**验收**：

- 两个 backend 共享前端和候选框架，不共享错误的 ISA 假设
- 同 family 的硬件变体只需 profile；新 ISA 通过 plugin 接入
- 两端均完成正确性、ABI、trace 和性能回归

SIMT/CUDA 是不同执行模型，应在 CPU vector 双后端稳定后作为独立阶段设计，而不是仅增加第三份 YAML。

### Phase 4：研究平台能力

**范围**：

- lowering trace 与 cost landscape 可视化
- Conv2D、GEMM、Reduce、Attention 子图 benchmark
- 多目标/Pareto 搜索与可插拔模型
- 与 vendor library、上游编译器和 hand-tuned kernel 比较

**验收**：

- 报告同时展示正确性、编译时间、运行性能和预测误差
- 可视化直接读取结构化 trace
- 新模型可通过稳定 API 接入并在固定数据集评估

---

## 12. 关键决策与风险

### 12.1 实现前必须关闭的决策

| 决策 | 建议默认值 | 原因 |
|---|---|---|
| 第一前端 | textual Tensor IR + 最小 C++ builder | 先隔离前端 API 复杂度 |
| 第一算子 | 受限 f32 direct Conv2D | 对齐项目目标，并覆盖布局、padding、stride、归约和 tail |
| Compute 表达 | linalg + scf/affine | 复用成熟变换基础设施 |
| Schedule 表达 | Transform Dialect extension | 明确 plan 与 payload |
| 第一目标 | 一台确定的 RVV 设备或 emulator | profile 和实测必须有基准 |
| 集成方式 | RVV adapter + 固定版本 | 降低相邻工程内部变化影响 |
| 编译模式 | AOT | 先关闭 JIT/cache/runtime 变量 |

### 12.2 主要风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 自研 IR 与上游能力重叠 | 维护成本高、转换冗余 | 新增 Dialect 前写语义差异说明 |
| Schedule 只是注释而非可执行计划 | 无法验证和复现 | 使用 Transform IR 并测试 materialization |
| YAML 被当作后端代码生成器 | 语义不完整、错误 fallback | Target Profile + backend plugin 分工 |
| Cost Model 越过合法性 | 选出快但错误的候选 | verifier/constraint 先于 estimate |
| 仅验证 IR 文本 | ABI 或运行结果错误 | 端到端 differential test |
| Conv MVP 同时引入多算法、融合和动态 shape | MVP 长期无法闭环 | Phase 1 固定 direct、f32、静态 shape 和单一布局 |
| 路线图过早覆盖 Attention/MoE/CUDA | 后续阶段失焦 | 先通过 Conv2D 单算子阶段门槛 |
| 相邻 RVV 工程接口不稳定 | 集成反复返工 | adapter、版本锁定、契约测试 |

---

## 13. 设计原则总结

1. **语义优先**：lowering 可以改变表示，不能静默改变计算契约。
2. **合法性先于性能**：Cost Model 只比较合法候选。
3. **计划与结果分离**：Schedule Plan 可搜索、可记录，应用后产生显式 payload IR。
4. **能力描述与语义实现分离**：Target Profile 描述事实，backend plugin 实现 lowering。
5. **保守失败**：无合法 expansion 时明确报错或回退，不猜测等价实现。
6. **复用上游**：只有在 DNN 或目标语义确有缺口时新增 Dialect。
7. **端到端验收**：parser 通过不等于正确，生成 LLVM IR 也不等于可调用。
8. **可解释且可复现**：每个选择都有结构化依据、版本和环境指纹。

判断一个抽象是否值得独立成层，可以问：

> 这一层是否拥有稳定且独特的语义、不变量和消费者？如果移除它，是否会迫使两个不同问题耦合？

如果答案是否定的，应优先使用已有 Dialect、interface 或普通数据结构，而不是新增一层 IR。

---

## 附录 A：与 MLIR 上游组件的关系

| OpenComputeFlow 概念 | 优先复用的 MLIR 组件 | 自研边界 |
|---|---|---|
| Tensor IR | Builtin/Tensor、可转换到 Linalg | DNN 专用语义、数值与 layout contract |
| Compute IR | Linalg、SCF、Affine、Arith、Math | 必要时提供薄 op/interface |
| Schedule Plan | Transform Dialect | 项目特有的候选参数与 transform extension |
| Scheduled Compute | Linalg/SCF/Affine/Vector/MemRef | provenance 与目标约束属性 |
| Hardware IR | Vector、GPU、NVVM、LLVM 及目标 Dialect | 上游不能表达的目标语义 |
| Lowering | Pattern Rewrite、Dialect Conversion、Pass Manager | 项目 pipeline 与 legality |
| Backend | LLVM Dialect/LLVM IR translation 等 | backend plugin、ABI 和 runtime glue |
| 测试 | LIT、FileCheck、LLVM verifier | differential、benchmark、trace 校验 |

相关上游设计参考：

- [MLIR Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
- [MLIR Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Vector Dialect](https://mlir.llvm.org/docs/Dialects/Vector/)
- [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [MLIR LLVM IR Target](https://mlir.llvm.org/docs/TargetLLVMIR/)

---

## 附录 B：阶段不变量检查表

| 边界 | 必须检查 |
|---|---|
| Frontend -> Tensor | op 版本、shape/type/layout、数值策略、副作用 |
| Tensor -> Compute | decomposition 适用条件、归约语义、边界与 workspace |
| Compute -> Schedule | 依赖、动态 guard、资源约束、候选 provenance |
| Plan -> Scheduled Compute | handle/变换成功、payload verifier、容量与 legality |
| Scheduled Compute -> Hardware | capability、mask/tail、地址空间、同步和类型 |
| Hardware -> Backend | ABI、data layout、target feature、非法 op 清零 |
| Backend -> Executable | verifier、链接、运行时符号、目标兼容性 |
| Executable -> Result | 参考结果、误差策略、确定性、性能采样环境 |
