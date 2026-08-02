"""Numerical reference implementation for the Phase 0 Conv2D contract."""

from __future__ import annotations

import struct
from typing import Iterable, List, Sequence, Tuple

from .contracts import Conv2DContract, ContractError


def to_f32(value: float) -> float:
    """Round through IEEE-754 binary32 to make the reference's accumulation explicit."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def direct_conv2d(
    contract: Conv2DContract,
    input_values: Sequence[float],
    filter_values: Sequence[float],
) -> Tuple[float, ...]:
    """Evaluate ordered, non-fused cross-correlation as the canonical f32 oracle."""
    contract.validate()
    expected_input = _product(contract.input_shape)
    expected_filter = _product(contract.filter_shape)
    if len(input_values) != expected_input:
        raise ContractError(f"input has {len(input_values)} values, expected {expected_input}")
    if len(filter_values) != expected_filter:
        raise ContractError(f"filter has {len(filter_values)} values, expected {expected_filter}")

    n, channels, height, width = contract.input_shape
    output_channels, _, kernel_h, kernel_w = contract.filter_shape
    output_n, _, output_h, output_w = contract.output_shape
    stride_h, stride_w = contract.strides
    pad_top, _, pad_left, _ = contract.padding
    output = [to_f32(0.0)] * _product(contract.output_shape)

    def input_index(batch: int, channel: int, row: int, col: int) -> int:
        return ((batch * channels + channel) * height + row) * width + col

    def filter_index(out_channel: int, channel: int, row: int, col: int) -> int:
        return ((out_channel * channels + channel) * kernel_h + row) * kernel_w + col

    def output_index(batch: int, out_channel: int, row: int, col: int) -> int:
        return ((batch * output_channels + out_channel) * output_h + row) * output_w + col

    for batch in range(output_n):
        for out_channel in range(output_channels):
            for out_row in range(output_h):
                for out_col in range(output_w):
                    accumulator = to_f32(0.0)
                    for channel in range(channels):
                        for kernel_row in range(kernel_h):
                            input_row = out_row * stride_h + kernel_row - pad_top
                            for kernel_col in range(kernel_w):
                                input_col = out_col * stride_w + kernel_col - pad_left
                                if 0 <= input_row < height and 0 <= input_col < width:
                                    input_value = to_f32(input_values[input_index(batch, channel, input_row, input_col)])
                                else:
                                    input_value = to_f32(0.0)
                                filter_value = to_f32(filter_values[filter_index(out_channel, channel, kernel_row, kernel_col)])
                                accumulator = to_f32(accumulator + to_f32(input_value * filter_value))
                    output[output_index(batch, out_channel, out_row, out_col)] = accumulator
    return tuple(output)


def _product(values: Iterable[int]) -> int:
    product = 1
    for value in values:
        product *= value
    return product
