// RUN: ocf-opt %s | FileCheck %s
// RUN: ocf-opt --show-dialects | FileCheck %s --check-prefix=DIALECT

// CHECK-LABEL: module {
// CHECK: func.func @identity(%arg0: f32) -> f32 {
// CHECK-NEXT: return %arg0 : f32
// CHECK: }
// DIALECT-DAG: affine
// DIALECT-DAG: arith
// DIALECT-DAG: func
// DIALECT-DAG: linalg
// DIALECT-DAG: memref
// DIALECT-DAG: scf
// DIALECT-DAG: tensor
// DIALECT-DAG: transform
// DIALECT-DAG: vector
module {
  func.func @identity(%arg0: f32) -> f32 {
    return %arg0 : f32
  }
}
