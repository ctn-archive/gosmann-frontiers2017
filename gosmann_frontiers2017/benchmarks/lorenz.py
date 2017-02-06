"""Lorenz attractor."""

import nengo


def lorenz(n_neurons=2000, seed=None):
    tau = 0.1
    sigma = 10.
    beta = 8. / 3.
    rho = 28.

    with nengo.Network(seed=seed, label="Lorenz attractor") as model:
        model.state = nengo.Ensemble(
            n_neurons=n_neurons, dimensions=3, radius=60.)

        def feedback(x):
            dx0 = -sigma * x[0] + sigma * x[1]
            dx1 = -x[0] * x[2] - x[1]
            dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho
            return [dx0 * tau + x[0],
                    dx1 * tau + x[1],
                    dx2 * tau + x[2]]

        nengo.Connection(
            model.state, model.state, function=feedback, synapse=tau)
        model.p_state = nengo.Probe(model.state, synapse=tau)

    return model


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    model = lorenz()
    with nengo.Simulator(model) as sim:
        sim.run(6.)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    d = sim.data[model.p_state]
    ax.plot(d[:, 0], d[:, 1], d[:, 2])
    plt.show()
