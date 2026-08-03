# OpenComputeFlow 构建说明

## 工具链基线

Phase 0C 固定使用以下工具链：

- CMake 3.20 或更高版本
- Ninja
- Python 3
- LLVM/MLIR `22.1.8`
- 上游 tag：`llvmorg-22.1.8`
- 上游 commit：`ca7933e47d3a3451d81e72ac174dcb5aa28b59d1`

CMake 使用 `EXACT` 查找 LLVM 和 MLIR。版本不匹配会在 configure 阶段失败，不允许依赖另一个 22.x 版本碰巧兼容。

## 配置和构建

LLVM 与 MLIR 必须来自同一构建树。显式传入两个 package 路径：

~~~bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
~~~

项目拒绝在源码目录内构建。构建过程不读取 RVV SDK、设备环境或其他开发者 shell 状态。

## 测试

~~~bash
PYTHONPATH=python python3 -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
cmake --build build --target check-opencomputeflow
build/bin/ocf-opt --version
~~~

`ctest` 覆盖 C++ contract reader 和 LIT smoke suite；`check-opencomputeflow` 提供与 MLIR 上游工程一致的回归测试入口。

当前 `ocf-opt` 注册 Conv 映射路径所需的上游 MLIR dialect、通用 transform passes 和 OCF Conv 方言。`ocf.conv2d` 目前是 Phase 1A 的语义/verifier 增量；Linalg lowering 尚未接入。

## CI

GitHub Actions 将参考契约测试和 MLIR 构建测试拆分。MLIR job 从固定的 `llvmorg-22.1.8` tag 构建项目实际使用的 targets，再运行 CTest 和 LIT；不使用滚动的 LLVM 22 snapshot package。
