import abc

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


class AbstractNorm(eqx.Module):
    @abc.abstractmethod
    def __call__(self, x: Array, *_, **__) -> Array: ...


class AbstractNormStateful(eqx.nn.StatefulLayer):
    @abc.abstractmethod
    def __call__(
        self, x: Array, state: eqx.nn.State, *_, key: PRNGKeyArray | None = None, **__
    ) -> tuple[Array, eqx.nn.State]: ...
