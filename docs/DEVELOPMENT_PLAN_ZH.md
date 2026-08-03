# OpenComputeFlow 开发计划

本文把 [架构设计](DESIGN_ZH.md) 转换为可执行的工程迭代。计划以能力门槛而不是日期为主：上一个迭代没有通过正确性、契约和可复现性测试时，不进入下一个迭代。

---

## 1. 交付策略

项目采用纵向小步闭环：

~~~text
Freeze contract
    -> build reference
    -> generate mapping
    -> check legality
    -> estimate and explain
    -> materialize IR
    -> execute and measure
    -> compare and calibrate
~~~

每个迭代必须同时给出：

- **范围**：本次明确支持和拒绝的语义
- **机器可读产物**：contract、profile、candidate、trace 或 executable
- **验证方式**：固定命令、输入和断言
- **退出门槛**：进入下一迭代前必须满足的条件
- **非目标**：避免在闭环完成前扩大算子和后端范围

Python 参考路径用于冻结契约和数值语义，不是生产 compiler backend。C++/MLIR 实现必须与参考路径使用相同 golden data，并通过 differential test。

---

## 2. 迭代总览

| 迭代 | 核心问题 | 主要产物 | 退出门槛 |
|---|---|---|---|
| Phase 0A | Conv 与 Mapping 的最小契约是什么？ | Python reference、RVV profile、unit tests | 契约/数值/合法性/估算测试稳定通过 |
| Phase 0B | 契约能否跨语言稳定消费？ | JSON Schema、golden fixtures、Measurement/Trace | Python schema 与 C++ 读取结果一致 |
| Phase 0C | MLIR 工程能否稳定构建和测试？ | CMake、`ocf-opt`、LIT、CI | clean build、smoke test、版本锁定 |
| Phase 1A | Conv 语义如何进入结构化 Compute IR？ | `ocf.conv2d` 或薄语义层、Linalg lowering | verifier 与 reference differential test 通过 |
| Phase 1B | 映射如何成为可执行计划？ | candidate generator、legality、Transform Plan | 至少两组候选可 materialize，非法候选有稳定诊断 |
| Phase 1C | RVV 映射是否可执行？ | RVV adapter、LLVM lowering、AOT runner | emulator/设备数值正确，ABI 测试通过 |
| Phase 1D | 性能解释是否可证伪？ | benchmark、measurement、calibration、report | 固定验证集报告误差、排序与 regret |
| Phase 2 | Conv 映射能否覆盖更多选择？ | implicit GEMM、layout/fusion/dynamic shape | 多算法合法选择与 fallback 可解释 |
| Phase 3 | 抽象能否跨执行模型？ | SIMT 或显式 SRAM/DMA backend | 同一 Compute Contract 在异构目标形成不同合法映射 |
| Phase 4 | 抽象能否复用到新计算模式？ | GEMM、Attention 子图 | 新算子复用点与抽象缺口均有报告 |

---

## 3. Phase 0A：参考契约与语义基线

### 3.1 当前范围

- f32 forward-inference direct Conv2D
- input/output NCHW，filter OIHW
- 静态正整数 shape，groups=1，dilation=1
- 编译期 stride 与四侧非负 padding
- 无 bias、activation、quantization 和其他 post-op
- AOT、单线程 RVV target profile

### 3.2 当前产物

- `Conv2DContract`：shape、layout、dtype、数值和算法约束
- `TargetProfile`：版本、能力、稳定资源 ID 和说明性性能参数
- `MappingCandidate`：loop order、tile、vector/tail、资源绑定和 provenance
- `direct_conv2d`：显式 f32 累加的 cross-correlation 参考实现
- `analytical-v1`：compute/memory/overhead 分解的未校准估算
- `ocf_phase0.py`：输出 contract、profile、candidate 和 estimate 的可运行样例

### 3.3 自动化测试

~~~bash
PYTHONPATH=python python3 -m unittest discover -s tests -v
PYTHONPATH=python python3 tools/ocf_phase0.py
~~~

测试至少覆盖：

- Conv shape 推导、理论 MAC/FLOP 和稳定 fingerprint
- 3x3/stride1/pad1、7x7/stride2/pad3 和非整除 tile/VL 场景
- channel、group、dilation、dtype、layout 和输出维度非法诊断
- cross-correlation、零 padding、f32 accumulation 和 buffer 长度
- Target Profile round-trip、资源引用、容量和 masked-tail legality
- Estimate 单位、组成、瓶颈、假设和确定性

### 3.4 退出门槛

- 所有测试在零第三方 Python 依赖下通过
- 相同 contract/profile/candidate 产生稳定 fingerprint 和 JSON
- 非法输入不得静默修正或继续估价
- 未校准 estimate 的 `confidence` 必须为 0，不得伪装成实测结论
- 示例输出可以作为 Phase 0B golden fixture 的生成源

### 3.5 明确非目标

- 不在 Python 中实现搜索框架、JIT 或生产 runtime
- 不把说明性性能参数当作真实 RVV calibration
- 不在 Phase 0A 引入 ONNX/cuDNN frontend parser
- 不在 MLIR 工具链未锁定前提交未经测试的 Dialect 骨架

---

## 4. Phase 0B：跨语言 Schema 与证据契约

### 4.1 开发项

1. 为 Conv2D、Target Profile、Mapping Candidate、Estimate、Measurement 和 Trace 提供 JSON Schema。
2. 生成合法与非法 golden fixtures，并固定 canonical JSON 与 SHA-256 规则。
3. 增加 `Measurement`：样本、warm-up、统计值、频率策略、设备和环境指纹。
4. 增加 `Trace`：候选全集、淘汰原因、选择目标、estimate 与 measurement 分离。
5. 实现最小 C++ reader，不依赖 MLIR，先验证跨语言一致性。

### 4.2 测试

- schema positive/negative fixtures
- Python -> JSON -> C++ -> JSON round-trip
- 未知必填字段、错误单位、错误资源 ID 和版本不兼容诊断
- 修改 contract/profile 任意语义字段后 fingerprint 必须变化
- estimate 不得写入 measurement 字段，功能 emulator 数据不得进入 calibration

### 4.3 退出门槛

- Python 与 C++ 对全部 golden fixtures 得到相同接受/拒绝结论
- schema 版本升级规则和兼容策略写入文档
- trace 能完整回答一个候选为何生成、淘汰或被选择

---

## 5. Phase 0C：MLIR 工程与测试基础设施

### 5.1 开发项

1. 锁定 LLVM/MLIR commit、CMake 最低版本和构建方式。
2. 建立 `include/`、`lib/`、`tools/ocf-opt`、`test/` 和 `cmake/`。
3. 接入 MLIR TableGen、Pass、Dialect registry、LIT/FileCheck。
4. `ocf-opt --version` 输出 OpenComputeFlow、LLVM/MLIR 和 contract schema 版本。
5. CI 执行 Python reference、C++ unit 和 LIT smoke tests。

### 5.2 退出门槛

- 新目录 clean configure/build/test 成功
- 不依赖开发者 shell 中隐式存在的 SDK 环境变量
- LLVM/MLIR 版本不匹配时在 configure 阶段明确失败

---

## 6. Phase 1：Conv2D 单算子映射闭环

### 6.1 Phase 1A：Operator -> Compute

- 实现最小 Conv2D 语义 op 或 importer，并复用 Tensor/Linalg 类型
- verifier 与 Python `Conv2DContract` 对齐
- direct Conv lowering 到 Linalg/SCF，保留 contract fingerprint 和 provenance
- 用小 shape 与 Python reference 做 differential test

退出门槛：合法样例 lowering 后数值一致；所有范围外语义有稳定诊断。

### 6.2 Phase 1B：Compute -> Mapping -> Scheduled Compute

- 规则枚举 baseline 和至少两组 OC/OH/OW tile/vector/tail candidate
- legality 按 Target Profile 检查资源、mask、tail、依赖和 guard
- 用 Transform Dialect materialize candidate，不允许 pass 私自改变映射
- materialization 后校验 payload 与 Mapping Candidate 一致

退出门槛：候选可独立序列化、估价和 materialize；失败不会污染原 payload。

### 6.3 Phase 1C：RVV -> LLVM -> AOT

- 明确与 `cudnn-rvv-mlir` 的版本化 adapter 边界
- 完成 scalable vector、VL、mask/tail 与 LLVM lowering
- 定义 C ABI、buffer descriptor、错误码和 host runner
- emulator 验证指令路径与数值；真实设备用于性能测量

退出门槛：设计文档规定的三个 Conv 类别均能执行并通过 reference comparison。

### 6.4 Phase 1D：Measure -> Calibrate -> Report

- 冻结 benchmark shapes、warm-up、采样次数和环境指纹
- microbenchmark 校准 compute throughput、memory bandwidth 和 overhead
- 输出误差分布、rank correlation、top-k hit 和 selected-vs-oracle regret
- 对最大误差样例记录模型假设失效原因

退出门槛：选择结果几何平均性能不差于 baseline，所有结论可由 trace 与原始测量复现。

---

## 7. 测试分层与提交规则

| 修改类型 | 提交前必须运行 |
|---|---|
| Python contract/reference | Phase 0 unittest + sample CLI |
| Schema/golden | schema negative tests + round-trip |
| Dialect/verifier | C++ unit + LIT parser/verifier |
| Lowering/Transform | LIT legality + reference differential |
| Backend/ABI | LLVM verifier + target check + host E2E |
| Cost Model | fixed dataset regression + ranking report |
| Benchmark | correctness check + environment fingerprint |

每个提交只推进一个可验证能力。涉及契约字段的修改必须同时更新版本、golden fixture、reader 和 trace；不得通过放宽测试容差掩盖数值语义变化。

---

## 8. 当前进度

- [x] Phase 0A：Conv2D 语义契约与稳定 fingerprint
- [x] Phase 0A：Target Profile 与 Mapping Candidate legality
- [x] Phase 0A：direct Conv f32 reference
- [x] Phase 0A：未校准 analytical estimate 与可运行样例
- [x] Phase 0A：首批自动化测试
- [x] Phase 0A：非法 dtype/layout/dilation、资源和 provenance negative tests
- [x] Phase 0B：六类 artifact 的 JSON Schema 与 negative tests
- [x] Phase 0B：Measurement、DecisionRecord、CompilationTrace 与字段隔离
- [x] Phase 0B：canonical fingerprint 规则与 golden fingerprint
- [x] Phase 0B：最小 C++ reader、canonical SHA-256 与跨语言 golden round-trip
- [ ] Phase 0C：锁定 MLIR 工具链并建立 CMake/LIT 工程

Phase 0B 已通过退出门槛。下一步进入 Phase 0C：锁定 MLIR 22.1.8，建立 `ocf-opt`、LIT/FileCheck 和版本 smoke test；在这些基础设施通过前不开发 Dialect op。
