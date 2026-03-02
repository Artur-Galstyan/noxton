# ESM (Protein Language Models)

```python
from noxton.models import ESM3, ESMC
```

Noxton includes two protein language model architectures from EvolutionaryScale:

- **ESMC** — a compact, efficient protein sequence model
- **ESM3** — a multimodal model that jointly reasons over protein *sequence*, *structure*, *secondary structure (SS8)*, *solvent accessibility (SASA)*, and *function annotations*

Both models require the optional `huggingface-hub` dependency and will download weights from HuggingFace Hub on first use.

!!! note
    Install the `hub` extra for ESM support:
    ```bash
    pip install "noxton[hub]"
    ```

---

## ESMC

A Transformer-based protein sequence model. Given a tokenised amino-acid sequence it returns per-residue logits over the vocabulary.

### `from_pretrained`

```python
ESMC.from_pretrained(
    model: Literal["esmc_300m", "esmc_600m"],
    key: PRNGKeyArray | None = None,
    dtype = None,
) -> ESMC
```

Weights are downloaded from HuggingFace Hub and cached locally.

| `model` | Parameters | HuggingFace repo |
|---|---|---|
| `esmc_300m` | 300M | `EvolutionaryScale/esmc-300m-2024-12` |
| `esmc_600m` | 600M | `EvolutionaryScale/esmc-600m-2024-12` |

### `__call__`

```python
esmc(
    sequence_tokens: Array,            # (L,) int32
    sequence_id: Array | None = None,  # (L,) bool attention mask (True = real token)
) -> tuple[Array, Array, Array]
    # (sequence_logits, last_hidden_state, all_hidden_states)
    # sequence_logits: (L, 64)
    # last_hidden_state: (L, d_model)
    # all_hidden_states: (n_layers, L, d_model)
```

### Vocabulary

```python
SEQUENCE_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>",
                  "L", "A", "G", "V", "S", "E", "R", "T",
                  "I", "D", "P", "K", "Q", "N", "F", "Y",
                  "M", "H", "W", "C", "X", "B", "U", "Z",
                  "O", ".", "-", "|", "<mask>"]
```

Special token indices: `BOS=0`, `PAD=1`, `EOS=2`, `MASK=32`.

### Example

```python
import jax
import jax.numpy as jnp
from noxton.models import ESMC

model = ESMC.from_pretrained("esmc_300m", dtype=jnp.float16)

# Encode a short protein sequence (BOS + residues + EOS)
# Vocabulary indices: BOS=0, L=4, A=5, G=6, EOS=2
tokens = jnp.array([0, 4, 5, 6, 4, 5, 2], dtype=jnp.int32)

logits, embedding, hiddens = model(tokens)
# logits.shape    -> (7, 64)
# embedding.shape -> (7, 960)  [for esmc_300m]
```

---

## ESM3

A multimodal protein language model that reasons jointly over sequence, structure, secondary structure, solvent accessibility, and function annotations (Hayes et al., 2024).

### `from_pretrained`

```python
ESM3.from_pretrained(
    model: Literal["esm3_open"] = "esm3_open",
    key: PRNGKeyArray | None = None,
    dtype = None,
) -> ESM3
```

| `model` | Parameters | HuggingFace repo |
|---|---|---|
| `esm3_open` | 1.4B | `EvolutionaryScale/esm3-sm-open-v1` |

### `__call__`

All inputs are optional. Any `None` input is replaced with an appropriate mask or padding token.

```python
esm3(
    sequence_tokens: Array | None = None,              # (L,) int32
    structure_tokens: Array | None = None,             # (L,) int32
    ss8_tokens: Array | None = None,                   # (L,) int32
    sasa_tokens: Array | None = None,                  # (L,) int32
    function_tokens: Array | None = None,              # (L, 8) int32
    residue_annotation_tokens: Array | None = None,    # (L, 16) int32
    average_plddt: Array | None = None,                # scalar
    per_res_plddt: Array | None = None,                # (L,)
    structure_coords: Array | None = None,             # (L, 3, 3) N/CA/C coords
    chain_id: Array | None = None,                     # (L,) int32
    sequence_id: Array | None = None,                  # (L,) bool attention mask
) -> tuple[Array, Array, Array, Array, Array, Array, Array]
    # (sequence_logits, structure_logits, secondary_structure_logits,
    #  sasa_logits, function_logits, residue_logits, embedding)
```

### Example: sequence-only forward pass

```python
import jax
import jax.numpy as jnp
from noxton.models import ESM3

model = ESM3.from_pretrained("esm3_open")

# Sequence tokens only — all other modalities are masked
tokens = jnp.array([0, 4, 5, 6, 4, 5, 2], dtype=jnp.int32)

outputs = model(sequence_tokens=tokens)
seq_logits = outputs[0]
# seq_logits.shape -> (7, vocab_size)
```
