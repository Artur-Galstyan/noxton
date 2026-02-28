import equinox as eqx
import jax


class Model(eqx.Module):
    inference: bool = eqx.field(static=True)
    lin: eqx.nn.Linear

    def __init__(self):
        self.inference = False
        self.lin = eqx.nn.Linear(10, 10, key=jax.random.key(2))


model = Model()
print(model)

model = eqx.nn.inference_mode(model)
print(model)
