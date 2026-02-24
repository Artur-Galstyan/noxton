import equinox as eqx
import numpy as np
import torch
from statedict2pytree.converter import autoconvert

bn_eqx = eqx.nn.make_with_state(eqx.nn.BatchNorm)(
    input_size=32, axis_name="batch", mode="batch"
)

bn_torch = torch.nn.BatchNorm1d(32)


bn_eqx, bn_s = autoconvert(bn_eqx, bn_torch.state_dict())


print(np.allclose(np.array(bn_eqx.weight), bn_torch.weight.detach().numpy()))


o_t = bn_torch(torch.ones(32))

print(o_t)
