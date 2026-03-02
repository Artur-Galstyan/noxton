# NN Layers — Overview

`noxton.nn` provides JAX/Equinox implementations of common neural network building blocks. All classes are `eqx.Module` subclasses and accept `key: PRNGKeyArray` and `dtype` constructor arguments.

## Module list

| Module | Class(es) | Description |
|---|---|---|
| [Attention](nn/attention.md) | `MultiheadAttention`, `SqueezeExcitation` | Multi-head scaled dot-product attention; SE channel recalibration |
| [Normalization](nn/normalization.md) | `BatchNorm`, `LayerNorm`, `LocalResponseNormalization` | Batch, layer, and LRN normalization |
| [Convolution](nn/convolution.md) | `ConvNormActivation` | Conv → Norm → Activation block |
| [Embedding](nn/embedding.md) | `EmbeddingWithPadding`, `EmbeddingBag` | Embedding tables with padding support |
| [Regularization](nn/regularization.md) | `StochasticDepth` | DropPath / stochastic depth |
| [Linear](nn/linear.md) | `BatchedLinear` | Linear layer with arbitrary batch dimensions |
| [State Space (Mamba)](nn/mamba.md) | `SelectiveStateSpaceModel`, `MambaBlock`, `ResidualBlock`, `Mamba` | Mamba sequence model components |
| [Transformer](nn/transformer.md) | `TransformerEncoderLayer`, `TransformerDecoderLayer`, `TransformerEncoder`, `TransformerDecoder`, `Transformer`, `VisionTransformer` | Transformer encoder/decoder stack; ViT |

## Importing

```python
from noxton.nn import (
    MultiheadAttention,
    SqueezeExcitation,
    ConvNormActivation,
    BatchNorm,
    LayerNorm,
    LocalResponseNormalization,
    EmbeddingWithPadding,
    EmbeddingBag,
    StochasticDepth,
    BatchedLinear,
    SelectiveStateSpace,
    SelectiveStateSpaceModel,
    MambaBlock,
    Mamba,
    TransformerEncoder,
    TransformerDecoder,
    Transformer,
    VisionTransformer,
)
```

## Abstract base classes

Two abstract base classes define the interface for normalization layers used internally by `ConvNormActivation`:

- **`AbstractNorm`** — stateless normalisation (e.g. `LayerNorm`). `__call__(x) -> x`.
- **`AbstractNormStateful`** — stateful normalisation (e.g. `BatchNorm`). `__call__(x, state) -> (x, state)`.

Custom norm layers can extend either base class to be compatible with `ConvNormActivation`.
