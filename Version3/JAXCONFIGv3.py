
JAXCONFIG = r"""
[Model]
logic='FuzzyLogic'
logic_kwargs={'weight': 750}
tnorm='ProductTNorm'
tnorm_kwargs={}

[Optimizer]
method='JaxDeepReactivePolicy'
method_kwargs={'topology': [128, 64]}
optimizer='rmsprop'
optimizer_kwargs={'learning_rate': 0.001}
batch_size_train=10
batch_size_test=10
action_bounds={'tVel': (0, 3), 'heading' : (-3.14, 3.14)}

[Training]
key=42
epochs=300
train_seconds=300
"""

with open('jax.cfg', 'w') as f:
    f.write(JAXCONFIG)
