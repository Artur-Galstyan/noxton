import equinox as eqx
import jax.numpy as jnp
import torch
from statedict2pytree.converter import autoconvert

bn_eqx = eqx.nn.make_with_state(eqx.nn.BatchNorm)(
    input_size=32, axis_name="batch", mode="batch", momentum=0.9
)

bn_torch = torch.nn.BatchNorm1d(32)

print(bn_torch.state_dict())


bn_eqx, bn_s = autoconvert(bn_eqx, bn_torch.state_dict())


o_t = bn_torch(torch.ones(4, 32))
o_t = bn_torch(torch.ones(4, 32))
o_t = bn_torch(torch.ones(4, 32))
o_t = bn_torch(torch.ones(4, 32))


print(bn_torch.running_mean, bn_torch.running_var)


o_j, bn_s = eqx.filter_vmap(
    bn_eqx, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(jnp.ones((4, 32)), bn_s)
o_j, bn_s = eqx.filter_vmap(
    bn_eqx, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(jnp.ones((4, 32)), bn_s)
o_j, bn_s = eqx.filter_vmap(
    bn_eqx, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(jnp.ones((4, 32)), bn_s)
o_j, bn_s = eqx.filter_vmap(
    bn_eqx, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(jnp.ones((4, 32)), bn_s)


print(bn_s.get(bn_eqx.batch_state_index))
