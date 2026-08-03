# OpenComputeFlow

<p align="center">
  <img src="assets/opencomputeflow-icon.png" width="180" alt="OpenComputeFlow project icon">
</p>

> **An Operator-to-Architecture Co-design Framework**
>
> *Making AI operator-to-hardware mapping explicit, explainable, and measurable.*

## 项目简介

OpenComputeFlow 是一个面向 AI 算子的软硬件映射研究框架。它位于算子语义与硬件执行之间，研究一个算子应如何分解、调度、放置数据并映射到目标架构，以及为什么选择这种映射。

项目不以复刻 TVM、XLA 或通用 MLIR 编译栈为目标。MLIR/LLVM 和生成 kernel 是验证手段，核心产物是三类可复用、可比较的契约：

- **AI Compute Contract**：描述算子语义、计算模式、数据访问和可调度维度
- **Hardware Mapping Contract**：描述计算、存储、通信资源及映射约束与决策
- **Performance Evidence**：用分析模型、校准数据和实测解释候选的性能差异

项目当前处于设计阶段。首个 MVP 是受限的 f32 direct Conv2D，用一个算子打通 Operator -> Architecture 的最小研究闭环：

- 静态 shape，input/output 使用 NCHW，filter 使用 OIHW
- groups=1、dilation=1，stride 和 padding 为编译期常量
- AOT、单线程、单 RVV 目标
- 输出至少两组合法 Mapping Candidate，记录选择与淘汰原因
- 分解 compute、memory、tail/overhead 代价并与实测比较
- 通过 RVV -> LLVM -> executable 验证映射可执行且结果正确

## 设计原则

- **语义优先**：lowering 不得静默改变计算、数值或布局契约
- **映射是一等公民**：算法、tile、布局、内存放置和并行映射必须结构化表达
- **合法性先于性能**：Cost Model 只比较已经合法的候选
- **预测必须可校准**：每个性能结论都应能追溯到假设、目标参数或实测数据
- **能力与实现分离**：Target Profile 描述硬件事实，backend plugin 验证映射可执行
- **优先复用上游**：复用 MLIR Linalg、SCF、Affine、Vector 和 Transform Dialect
- **kernel 是验证手段**：端到端代码生成用于检验抽象，不以建设通用编译器为目标

完整架构、MVP 契约、阶段门槛和风险分析见 [中文设计文档](docs/DESIGN_ZH.md)。

## 当前实现

仓库已完成 Phase 0A，并开始 Phase 0B，提供契约参考与证据路径：

- 受限 f32 direct Conv2D 语义、shape 推导和参考实现
- 版本化 RVV Target Profile 与 Mapping Candidate legality
- compute/memory/overhead 分解的未校准分析估算
- Contract、Mapping、Estimate、Measurement 和 Trace v1 JSON Schema
- 预测与实测严格分离的 Evidence Trace，以及稳定 golden fingerprint
- 基于 LLVM Support 的 C++ reader 与 Python/C++ canonical SHA-256 一致性测试
- contract、mapping、estimate 和 trace 的可运行 JSON 示例

核心参考测试不需要第三方依赖；完整 Schema 测试需要开发依赖：

~~~bash
python3 -m pip install -r requirements-dev.txt
PYTHONPATH=python python3 -m unittest discover -s tests -v
PYTHONPATH=python python3 tools/ocf_phase0.py
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build
ctest --test-dir build --output-on-failure
~~~

分阶段任务、测试要求和退出门槛见 [开发计划](docs/DEVELOPMENT_PLAN_ZH.md)，字段兼容与指纹规则见 [Schema 版本规则](docs/SCHEMA_VERSIONING_ZH.md)。

## 路线图

- **Phase 0**：定义 Conv2D Compute、Mapping、Target 和 Evidence 契约
- **Phase 1**：在 RVV 上完成可解释、可测量的 direct Conv2D 映射闭环
- **Phase 2**：校准性能模型，扩展算法、layout、fusion 和 shape 候选
- **Phase 3**：映射到具有不同内存或并行模型的第二类架构
- **Phase 4**：扩展 GEMM/Attention，并形成可复用的软硬件协同实验平台

## 仓库状态

当前仓库主要包含设计文档，目录和接口将在通过相应阶段验收后逐步实现。
