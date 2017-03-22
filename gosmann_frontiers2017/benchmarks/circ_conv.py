"""Circular convolution."""

import nengo
import numpy as np


def circ_conv(n_neurons=500, dimensions=500, seed=None):
    n_neurons = int(n_neurons)
    dimensions = int(dimensions)
    if seed is not None:
        seed = int(seed)

    with nengo.Network(seed=seed, label="Circular convolution") as model:
        v = np.random.randn(2, dimensions)
        v /= np.linalg.norm(v, axis=1)[:, None]

        model.input_a = nengo.Node(np.array(v[0]))
        model.input_b = nengo.Node(np.array(v[1]))

        model.cconv = nengo.networks.CircularConvolution(
            n_neurons=n_neurons, dimensions=dimensions)
        nengo.Connection(model.input_a, model.cconv.A)
        nengo.Connection(model.input_b, model.cconv.B)

        model.p_input_a = nengo.Probe(model.input_a, synapse=0.01)
        model.p_input_b = nengo.Probe(model.input_b, synapse=0.01)
        model.p_output = nengo.Probe(model.cconv.output, synapse=0.01)

    return model


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = circ_conv()
    with nengo.Simulator(model) as sim:
        sim.run(1.)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(sim.trange(), sim.data[model.p_input_a])
    plt.subplot(3, 1, 2)
    plt.plot(sim.trange(), sim.data[model.p_input_b])
    plt.subplot(3, 1, 3)
    plt.plot(sim.trange(), sim.data[model.p_output])
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
