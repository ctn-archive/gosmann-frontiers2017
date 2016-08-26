import nengo
import nengo_ocl
import pyopencl as cl


def get_devices(device_type):
    for platform in cl.get_platforms():
        devices = platform.get_devices(device_type)
        if len(devices) > 0:
            return devices
    raise ValueError("No device of given type found.")


def reference():
    return nengo.Simulator


def ocl(device_type):
    context = cl.Context(get_devices(device_type))
    return lambda model: nengo_ocl.Simulator(model, context=context)


def ocl_gpu():
    return ocl(cl.device_type.GPU)


def ocl_cpu():
    return ocl(cl.device_type.CPU)
