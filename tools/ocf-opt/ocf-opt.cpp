#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace {

bool isVersionRequest(int argc, char **argv) {
  return argc == 2 && llvm::StringRef(argv[1]) == "--version";
}

void printVersion() {
  llvm::outs() << "OpenComputeFlow " << OCF_PROJECT_VERSION << '\n'
               << "LLVM/MLIR " << LLVM_VERSION_STRING << '\n'
               << "Contract schema " << OCF_SCHEMA_VERSION << '\n';
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  if (isVersionRequest(argc, argv)) {
    printVersion();
    return 0;
  }

  mlir::registerTransformsPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect,
                  mlir::transform::TransformDialect,
                  mlir::vector::VectorDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "OpenComputeFlow optimizer driver\n", registry));
}
