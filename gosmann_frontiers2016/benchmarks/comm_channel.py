import matplotlib.pyplot as plt
import nengo
import numpy as np


def comm_channel(n_neurons=50, dimensions=1, seed=None):
    with nengo.Network(seed=seed, label="Communication channel") as model:
        model.ens_a = nengo.Ensemble(
            n_neurons=n_neurons // 2, dimensions=dimensions)
        model.ens_b = nengo.Ensemble(
            n_neurons=n_neurons // 2, dimensions=dimensions)

        model.stimulus = nengo.Node(0.5 * np.ones(dimensions))
        nengo.Connection(model.stimulus, model.ens_a)
        nengo.Connection(model.ens_a, model.ens_b)

        model.p_input = nengo.Probe(model.stimulus, synapse=0.01)
        model.p_output = nengo.Probe(model.ens_b, synapse=0.01)

    return model


if __name__ == '__main__':
    model = comm_channel()
    with nengo.Simulator(model) as sim:
        sim.run(1.)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[model.p_input])
    plt.subplot(2, 1, 2)
    plt.plot(sim.trange(), sim.data[model.p_output])
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
