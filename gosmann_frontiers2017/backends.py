"""Defines the instantiation of various backends."""

import nengo
import nengo_ocl
import pyopencl as cl

import gosmann_frontiers2017.optimized.simulator as optsim


def get_devices(device_type):
    """Gets a list of devices.

    This will look for the first platform with devices of the given type and
    return a list of all those devices on the platform

    Parameters
    ----------
    device_type : :class:`pyopencl.device_type`
        Type of devices to return.

    Returns
    -------
    list
        Devices of requested type from first platform that has devices of the
        type.
    """
    for platform in cl.get_platforms():
        devices = platform.get_devices(device_type)
        if len(devices) > 0:
            return devices
    raise ValueError("No device of given type found.")


def reference():
    """Returns the Nengo reference simulator."""
    return nengo.Simulator


def ocl(device_type):
    """Nengo OCL simulator.

    Parameters
    ----------
    device_type : :class:`pyopencl.device_type`
        The type of device to run this simulator on.

    Returns
    -------
    class
        Nengo OCL Simulator
    """
    context = cl.Context(get_devices(device_type))
    return lambda model: nengo_ocl.Simulator(model, context=context)


def ocl_gpu():
    """Returns Nengo OCL simulator running on the GPU."""
    return ocl(cl.device_type.GPU)


def ocl_cpu():
    """Returns Nengo OCL simulator running on the CPU."""
    return ocl(cl.device_type.CPU)


def optimized():
    """Returns the Nengo reference simulator with the optimizer."""
    return optsim.Simulator
