import nengo
import numpy as np


def circ_conv(n_neurons=200, dimensions=4, seed=None):
    with nengo.Network(seed=seed, label="Circular convolution") as model:
        v = np.random.randn(2, dimensions)
        v /= np.linalg.norm(v, axis=1)[:, None]

        model.input_a = nengo.Node(np.array(v[0]))
        model.input_b = nengo.Node(np.array(v[1]))

        model.cconv = nengo.networks.CircularConvolution(
            n_neurons=n_neurons, dimensions=dimensions)
        nengo.Connection(model.input_a, model.cconv.A)
        nengo.Connection(model.input_b, model.cconv.B)

        model.p_result = nengo.Probe(model.cconv.output, synapse=0.02)

    return model
