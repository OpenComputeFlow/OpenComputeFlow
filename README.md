# OpenComputeFlow

> **An Experimental AI Compute Architecture Framework**
>
> *Exploring the abstraction from DNN APIs to hardware execution.*

## 这是什么？

OpenComputeFlow 不是一个"又一个 MLIR Compiler"。它是一个研究**DNN 计算如何逐步映射到硬件**的实验框架。

核心问题：一个 `cudnnConvolutionForward` 调用，应该经过哪些抽象层次，才能变成硬件指令？每一层的边界在哪里？为什么？

## 项目结构

```
frontend/        -- cuDNN / PyTorch 前端入口
ir/              -- 四层 IR 定义 (Tensor → Loop → Schedule → Hardware)
transform/       -- Progressive Lowering Passes
cost_model/      -- Cost Model（驱动全部 lowering 决策）
hardware/        -- Hardware Description (YAML) + 多后端
backend/llvm/    -- LLVM IR 发射（薄层，<20% 代码量）
visualizer/      -- IR 可视化工具
benchmark/       -- 性能基准测试
```

## 设计哲学

- **每一层回答一个独立的问题**
  - Tensor IR: "算法是什么？"
  - Loop IR: "如何计算？"
  - Schedule IR: "如何组织计算？"
  - Hardware IR: "如何使用硬件？"
  - LLVM IR: "如何编码？"

- **Cost Model 是一等公民** — Generate → Estimate → Choose → Lower，不做硬编码规则

- **不绑定硬件** — 换一个 YAML 描述文件，就换一个后端

详见 [docs/DESIGN_ZH.md](docs/DESIGN_ZH.md)

## 路线图

- **Phase 1**: 核心 IR 定义 + RVV 后端打通（GEMM 端到端）
- **Phase 2**: Cost Model 强化 + 多算法支持（Winograd）
- **Phase 3**: 多硬件后端（ARM/x86/CUDA）
- **Phase 4**: 可视化 + Benchmark 完善

## 相关项目

- [cudnn-rvv-mlir](https://github.com/.../cudnn-rvv-mlir) — RISC-V RVV 后端参考实现
