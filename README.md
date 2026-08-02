# OpenComputeFlow

<p align="center">
  <img src="assets/opencomputeflow-icon.svg" width="180" alt="OpenComputeFlow project icon">
</p>

> **An Experimental AI Compute Architecture Framework**
>
> *Exploring explainable and measurable lowering from DNN semantics to hardware execution.*

## 项目简介

OpenComputeFlow 是一个研究 DNN 计算如何逐步映射到硬件的实验框架，重点关注 IR 抽象边界、可执行调度计划、Cost Model、目标能力建模，以及 lowering 的可解释性与可复现性。

项目当前处于设计阶段。首个 MVP 是受限的 f32 direct Conv2D：

- 静态 shape，input/output 使用 NCHW，filter 使用 OIHW
- groups=1、dilation=1，stride 和 padding 为编译期常量
- AOT、单线程、单 RVV 目标
- Tensor -> Compute -> Schedule -> RVV -> LLVM 端到端闭环

## 设计原则

- **语义优先**：lowering 不得静默改变计算、数值或布局契约
- **合法性先于性能**：Cost Model 只比较已经合法的候选
- **计划与结果分离**：Schedule Plan 被选择和应用后生成显式 payload IR
- **能力与实现分离**：Target Profile 描述硬件，backend plugin 实现语义 lowering
- **优先复用上游**：复用 MLIR Linalg、SCF、Affine、Vector 和 Transform Dialect
- **端到端验收**：验证 ABI、运行结果、预测误差和复现信息，不只检查 IR 文本

完整架构、MVP 契约、阶段门槛和风险分析见 [中文设计文档](docs/DESIGN_ZH.md)。

## 路线图

- **Phase 0**：Conv2D 语义契约、Target Profile、trace 与测试基础设施
- **Phase 1**：f32 direct Conv2D 在 RVV 上端到端运行
- **Phase 2**：动态 shape、校准 Cost Model、implicit GEMM 与融合/layout 候选
- **Phase 3**：第二个 CPU vector backend
- **Phase 4**：可视化、可插拔模型与标准 benchmark

## 仓库状态

当前仓库主要包含设计文档，目录和接口将在通过相应阶段验收后逐步实现。
