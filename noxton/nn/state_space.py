import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float, PRNGKeyArray

from noxton.functions import selective_scan
from noxton.utils import default_floating_dtype


class SelectiveStateSpace(eqx.Module):
    """Selective State Space (SSM) block from the Mamba architecture.

    Implements the input-dependent SSM layer described in *Mamba:
    Linear-Time Sequence Modeling with Selective State Spaces* (Gu & Dao,
    2023).  Given an input sequence of shape ``(seq_length, d_inner)``, the
    block:

    1. Projects each token to ``(dt_rank + 2 * d_inner * d_state)`` to obtain
       ``delta`` (time step), ``B`` (input matrix), and ``C`` (output matrix).
    2. Projects ``delta`` from ``dt_rank`` to ``d_inner`` and applies
       ``softplus``.
    3. Runs the selective scan recurrence over the sequence.
    4. Projects the scan output back to ``d_model`` via ``out_proj``.

    The state-transition matrix ``A`` is stored in log-space (``A_log``) and
    is always negative after ``exp``, ensuring a stable recurrence.

    Args:
        d_model: Dimensionality of the block output (and the output
            projection target).
        d_inner: Expanded inner dimensionality for the SSM recurrence.
        dt_rank: Rank of the low-rank time-step projection.
        d_state: Latent state dimensionality ``N``.
        use_input_proj_bias: Whether to add bias to the input projection.
            Defaults to ``False``.
        use_delta_proj_bias: Whether to add bias to the delta projection.
            Defaults to ``False``.
        key: JAX PRNG key for parameter initialisation.
        dtype: Floating-point dtype for all parameters.  Defaults to the
            project default when ``None``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> ssm = SelectiveStateSpace(
        ...     d_model=32, d_inner=64, dt_rank=8, d_state=16, key=key
        ... )
        >>> x = jax.random.normal(key, (128, 64))  # (seq_len, d_inner)
        >>> ssm(x).shape
        (128, 32)
    """

    input_proj: eqx.nn.Linear
    delta_proj: eqx.nn.Linear
    A_log: Float[Array, "d_inner d_state"]
    D: Float[Array, "d_inner"]
    out_proj: eqx.nn.Linear

    d_inner: int = eqx.field(static=True)
    dt_rank: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dt_rank: int,
        d_state: int,
        use_input_proj_bias: bool = False,
        use_delta_proj_bias: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None

        self.d_model = d_model
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state

        keys = jax.random.split(key, 4)
        proj_dim = self.dt_rank + 2 * self.d_inner * self.d_state
        self.input_proj = eqx.nn.Linear(
            self.d_model,
            proj_dim,
            use_bias=use_input_proj_bias,
            key=keys[0],
            dtype=dtype,
        )

        self.delta_proj = eqx.nn.Linear(
            dt_rank,
            d_inner,
            use_bias=use_delta_proj_bias,
            key=keys[1],
            dtype=dtype,
        )

        A = jnp.arange(1, d_state + 1, dtype=jnp.float32)
        A = jnp.tile(A, (d_inner, 1))
        self.A_log = jnp.log(A)

        self.D = jnp.ones(d_inner, dtype=dtype)

        self.out_proj = eqx.nn.Linear(
            d_inner, d_model, use_bias=False, key=keys[2], dtype=dtype
        )

    def __call__(self, x: Float[Array, "seq_length d_inner"]):
        """Run the selective SSM over an input sequence.

        Args:
            x: Input sequence of shape ``(seq_length, d_inner)``.

        Returns:
            Output array of shape ``(seq_length, d_model)``.
        """
        L, _ = x.shape
        A = -jnp.exp(self.A_log.astype(jnp.float32))
        D = self.D.astype(jnp.float32)

        delta_b_c = jax.vmap(self.input_proj)(x)
        delta, B, C = jnp.split(
            delta_b_c,
            [self.dt_rank, self.dt_rank + self.d_inner * self.d_state],
            axis=-1,
        )

        B = B.reshape(L, self.d_inner, self.d_state)
        C = C.reshape(L, self.d_inner, self.d_state)

        delta = jax.nn.softplus(jax.vmap(self.delta_proj)(delta))

        y = selective_scan(x, delta, A, B, C, D)

        return jax.vmap(self.out_proj)(y)
