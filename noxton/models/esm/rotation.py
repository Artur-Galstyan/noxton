from abc import ABC

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Self
from jaxtyping import Array, PRNGKeyArray

from noxton.functions import graham_schmidt, normalize, qinv, qmul, qrot


def _sqrt_subgradient(x: Array) -> Array:
    return jnp.where(x > 0, jnp.sqrt(x), 0.0)


class Rotation(ABC):
    @classmethod
    def identity(cls, shape: tuple[int, ...], **tensor_kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    def random(cls, shape: tuple[int, ...], key: PRNGKeyArray, **tensor_kwargs) -> Self:
        raise NotImplementedError

    def __getitem__(self, idx: Any) -> Self:
        raise NotImplementedError

    @property
    def tensor(self) -> Array:
        raise NotImplementedError

    @property
    def shape(self) -> tuple:
        raise NotImplementedError

    def as_matrix(self) -> "RotationMatrix":
        raise NotImplementedError

    def as_quat(self, normalize: bool = False) -> "RotationQuat":
        raise NotImplementedError

    def compose(self, other: "Rotation") -> "Rotation":
        raise NotImplementedError

    def convert_compose(self, other: Self) -> Self:
        raise NotImplementedError

    def apply(self, p: Array) -> Array:
        raise NotImplementedError

    def invert(self) -> Self:
        raise NotImplementedError

    @property
    def dtype(self) -> jnp.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> jax.Device:
        return self.tensor.device

    def tensor_apply(self, func) -> Self:
        # Applys a function to the underlying tensor
        return type(self)(
            jnp.stack(
                [func(self.tensor[..., i]) for i in range(self.tensor.shape[-1])],
                axis=-1,
            )  # ty: ignore too-many-positional-arguments
        )


class RotationQuat(Rotation, eqx.Module):
    _quats: Array
    _normalized: bool = eqx.field(static=True)

    def __init__(self, quats: Array, normalized=False):
        assert quats.shape[-1] == 4
        self._normalized = normalized
        # Force float32 as well
        if normalized:
            self._quats = normalize(quats.astype(jnp.float32), axis=-1)
            self._quats = jnp.where(
                self._quats[..., :1] >= 0, self._quats, -self._quats
            )
        else:
            self._quats = quats.astype(jnp.float32)

    @classmethod
    def identity(cls, shape, **tensor_kwargs):
        q = jnp.ones((*shape, 4), **tensor_kwargs)
        mult = jnp.array([1, 0, 0, 0])
        return RotationQuat(q * mult)

    @classmethod
    def random(cls, shape, key: PRNGKeyArray, **tensor_kwargs):
        quat = jax.random.normal(key=key, shape=(*shape, 4), **tensor_kwargs)
        return RotationQuat(quat, normalized=True)

    def __getitem__(self, idx: Any) -> "RotationQuat":
        if isinstance(idx, (int, slice)) or idx is None:
            indices = (idx,)
        else:
            indices = tuple(idx)
        return RotationQuat(self._quats[indices + (slice(None),)])

    @property
    def shape(self) -> tuple:
        return self._quats.shape[:-1]

    def compose(self, other: Rotation) -> Rotation:
        assert isinstance(other, RotationQuat)
        return RotationQuat(qmul(self._quats, other._quats))

    def convert_compose(self, other: Rotation):
        return self.compose(other.as_quat())

    def as_matrix(self) -> "RotationMatrix":
        q = self.normalized().tensor
        r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        two_s = 2.0 / jnp.linalg.norm(q, axis=-1)

        o = jnp.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return RotationMatrix(o.reshape(q.shape[:-1] + (3, 3)))

    def as_quat(self, normalize: bool = False) -> "RotationQuat":
        return self

    def apply(self, p: Array) -> Array:
        return qrot(self.normalized()._quats, p)

    def invert(self) -> "RotationQuat":
        return RotationQuat(qinv(self._quats))

    @property
    def tensor(self) -> Array:
        return self._quats

    def normalized(self) -> "RotationQuat":
        return self if self._normalized else RotationQuat(self._quats, normalized=True)


class RotationMatrix(Rotation, eqx.Module):
    _rots: Array

    def __init__(self, rots: Array):
        if rots.shape[-1] == 9:
            rots = rots.reshape(*rots.shape[:-1], 3, 3)
        assert rots.shape[-1] == 3
        assert rots.shape[-2] == 3
        # Force full precision
        rots = rots.astype(jnp.float32)
        self._rots = rots

    @classmethod
    def identity(cls, shape, **tensor_kwargs):
        rots = jnp.eye(3, **tensor_kwargs)
        rots = rots.reshape(*[1 for _ in range(len(shape))], 3, 3)
        rots = jnp.broadcast_to(rots, (*shape, rots.shape[-2], rots.shape[-1]))
        return cls(rots)

    @classmethod
    def random(cls, shape, key: PRNGKeyArray, **tensor_kwargs):
        return RotationQuat.random(shape, **tensor_kwargs).as_matrix()

    def __getitem__(self, idx: Any) -> "RotationMatrix":
        indices = (idx,) if isinstance(idx, int) or idx is None else tuple(idx)
        return RotationMatrix(self._rots[indices + (slice(None), slice(None))])

    @property
    def shape(self) -> tuple:
        return self._rots.shape[:-2]

    def as_matrix(self) -> "RotationMatrix":
        return self

    def as_quat(self, normalize: bool = False) -> RotationQuat:
        flat = self._rots.reshape(*self._rots.shape[:-2], 9)
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = [flat[..., i] for i in range(9)]
        q_abs = _sqrt_subgradient(
            jnp.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                axis=-1,
            )
        )
        # we produce the desired quaternion multiplied by each of r, i, j, k
        quat_by_rijk = jnp.stack(
            [
                x
                for lst in [
                    [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01],
                    [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20],
                    [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21],
                    [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2],
                ]
                for x in lst
            ],
            axis=-1,
        )
        quat_by_rijk = quat_by_rijk.reshape(*quat_by_rijk.shape[:-1], 4, 4)

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = jnp.array(0.1, dtype=q_abs.dtype)
        quat_candidates = quat_by_rijk / (2.0 * jnp.maximum(q_abs[..., None], flr))

        one_hot = jax.nn.one_hot(q_abs.argmax(axis=-1), num_classes=q_abs.shape[-1])
        quat = (quat_candidates * one_hot[..., :, None]).sum(axis=-2)

        return RotationQuat(quat)

    def compose(self, other: Rotation) -> Rotation:
        assert isinstance(other, RotationMatrix)
        return RotationMatrix(self._rots @ other._rots)

    def convert_compose(self, other: Rotation):
        return self.compose(other.as_matrix())

    def apply(self, p: Array) -> Array:
        if self._rots.shape[-3] == 1:
            # This is a slight speedup over einsum for batched rotations
            return p @ jnp.swapaxes(self._rots, -1, -2).squeeze(-3)
        else:
            # einsum way faster than bmm!
            return jnp.einsum("...ij,...j", self._rots, p)

    def invert(self) -> "RotationMatrix":
        return RotationMatrix(jnp.swapaxes(self._rots, -1, -2))

    @property
    def tensor(self) -> Array:
        return self._rots.reshape(*self._rots.shape[:-2], -1)

    def to_3x3(self) -> Array:
        return self._rots

    @staticmethod
    def from_graham_schmidt(
        x_axis: Array, xy_plane: Array, eps: float = 1e-12
    ) -> "RotationMatrix":
        return RotationMatrix(graham_schmidt(x_axis, xy_plane, eps))
