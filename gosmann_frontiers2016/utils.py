import nengo


def activate_direct_mode(network):
    """Activates direct mode for a network.

    This sets the neuron type of all ensembles to ``nengo.Direct()`` except if
    there is a connection from or to the ensembles neurons (this includes
    probes on the ensemble's neurons).

    Parameters
    ----------
    network : nengo.Network
        Network to activate direct mode for.
    """
    requires_neurons = set()

    for c in network.all_connections:
        if isinstance(c.pre_obj, nengo.ensemble.Neurons):
            requires_neurons.add(c.pre_obj.ensemble)
        if isinstance(c.post_obj, nengo.ensemble.Neurons):
            requires_neurons.add(c.post_obj.ensemble)
    for p in network.all_probes:
        if isinstance(p.obj, nengo.ensemble.Neurons):
            requires_neurons.add(p.obj.ensemble)

    for e in network.all_ensembles:
        if e not in requires_neurons:
            e.neuron_type = nengo.Direct()
