# Contract Schema 与版本规则

OpenComputeFlow 的 contract、mapping 和 evidence 是跨模块接口，不是临时日志。本文件定义 v1 的版本与内容指纹规则。

## 1. 版本字段

- `schema_version`：JSON 结构或字段语义发生不兼容变化时递增
- `profile_version`：同一 `target_id` 的 correctness/performance facts 变化时递增
- `pipeline_version`：候选生成、选择或 lowering 流程发生可观察变化时递增
- `model_id`：Performance Model 的公式或特征发生变化时使用新 ID
- `calibration_id`：校准数据集、设备、环境或采集协议变化时使用新 ID

reader 必须拒绝未知的必填版本，不得猜测兼容性。v1 schema 使用 `additionalProperties: false` 的核心对象；新增字段需要显式评估是兼容扩展还是 v2。

## 2. 内容指纹

v1 使用 SHA-256：

1. 对完整 `to_dict()` 结果按 key 递归排序
2. JSON 使用 UTF-8、ASCII escaping、无额外空白，分隔符为 `,` 和 `:`
3. integer 使用十进制；数值为整数的 float 也规范化为 integer 文本；其余 float 使用最短往返表示；`-0.0` 规范化为 `0`
4. NaN 与 Inf 不是合法 contract 数字，fingerprint 计算必须拒绝
5. 对所得字节计算小写十六进制 SHA-256

Python 参考实现位于 `content_fingerprint`，C++ 参考 reader 位于 `tools/ocf-contract-check`。跨语言 reader 必须通过 `tests/golden/phase0-fingerprints-v1.json`，才能宣称实现 v1 指纹兼容。

内容指纹不是版本号的替代：版本说明如何解释字段，指纹说明消费的是否为完全相同的内容。

## 3. 引用规则

- Mapping Candidate 同时保存 `target_profile_ref` 与 `target_profile_fingerprint`
- Candidate 通过 `compute_contract_ref` 绑定确切的 Compute Contract
- Estimate 通过 `candidate_ref` 绑定确切 candidate
- Measurement 绑定 candidate 与 Target Profile，但与 Estimate 分开存储
- Trace 只保存 Estimate/Measurement 指纹引用，不复制或混写两类事实

任何被引用内容变化都会产生新指纹。加载 trace 时若引用缺失或指纹不一致，必须失败，不得使用“名称相同”的对象替代。

## 4. Estimate 与 Measurement

Estimate 是模型输出，必须包含模型 ID、假设、confidence 和性能分解。Measurement 是观测事实，必须包含：

- 真实硬件或可信 cycle-accurate model 的 `source_kind`
- 带时区的采集时间
- 原始 samples、warm-up 和可重新计算的统计值
- device、backend、clock policy 和线程数
- metric 与 unit 的固定对应关系

功能模拟器 wall-clock 数据不得构造为 Measurement。修改 samples 后若 summary 不一致，reader 必须拒绝。

## 5. Schema 验证与语义验证

JSON Schema 检查字段、类型、枚举、范围和额外字段；Python/C++ contract verifier 检查跨字段语义，例如 Conv 输出公式、channel 一致性、资源引用和 candidate 集合关系。两层都通过才表示 artifact 有效。

开发依赖安装与测试：

~~~bash
python3 -m pip install -r requirements-dev.txt
PYTHONPATH=python python3 -m unittest discover -s tests -v
~~~
