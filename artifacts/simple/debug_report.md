# Kuma — Debug Report

## Summary
- Total parameters : 883
- Weight blob size : 3,532 bytes  (0.0034 MB)
- Graph nodes      : 10
- Unique ops       : 3

## Inputs
- `arg4_1` : shape=[1, 3, 32, 32]  dtype=float32

## Outputs
- `add` : shape=[1, 3, 32, 32]  dtype=float32

## ATen Ops Encountered
| Op | Count |
|----|------:|
| `aten.add.Tensor` | 1 |
| `aten.convolution.default` | 2 |
| `aten.gelu.default` | 1 |

## Weights
| Name | Shape | Elements | Bytes | Offset |
|------|-------|--------:|------:|-------:|
| `conv1.bias` | [16] | 16 | 64 | 0 |
| `conv1.weight` | [16, 3, 3, 3] | 432 | 1,728 | 64 |
| `conv2.bias` | [3] | 3 | 12 | 1,792 |
| `conv2.weight` | [3, 16, 3, 3] | 432 | 1,728 | 1,804 |

## Graph Nodes
| ID | Name | Op | Target | Shape | Dtype |
|----|------|----|--------|-------|-------|
| 0 | `arg0_1` | placeholder | `arg0_1` | [16, 3, 3, 3] | float32 |
| 1 | `arg1_1` | placeholder | `arg1_1` | [16] | float32 |
| 2 | `arg2_1` | placeholder | `arg2_1` | [3, 16, 3, 3] | float32 |
| 3 | `arg3_1` | placeholder | `arg3_1` | [3] | float32 |
| 4 | `arg4_1` | placeholder | `arg4_1` | [1, 3, 32, 32] | float32 |
| 5 | `convolution` | call_function | `aten.convolution.default` | [1, 16, 32, 32] | float32 |
| 6 | `gelu` | call_function | `aten.gelu.default` | [1, 16, 32, 32] | float32 |
| 7 | `convolution_1` | call_function | `aten.convolution.default` | [1, 3, 32, 32] | float32 |
| 8 | `add` | call_function | `aten.add.Tensor` | [1, 3, 32, 32] | float32 |
| 9 | `output` | output | `output` |  |  |
