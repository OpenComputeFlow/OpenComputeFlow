// RUN: ocf-opt %s | FileCheck %s

module {
  func.func @conv(%input: tensor<1x3x8x8xf32>,
                  %filter: tensor<8x3x3x3xf32>) -> tensor<1x8x8x8xf32> {
    %output = "ocf.conv2d"(%input, %filter) {
      dilations = array<i64: 1, 1>,
      groups = 1 : i64,
      padding = array<i64: 1, 1, 1, 1>,
      strides = array<i64: 1, 1>
    } : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>) -> tensor<1x8x8x8xf32>
    return %output : tensor<1x8x8x8xf32>
  }
}

// CHECK: "ocf.conv2d"
// CHECK: tensor<1x8x8x8xf32>
