#!/usr/bin/env python3
"""Small FP32 correctness and transfer probe for an NVIDIA OpenCL GPU."""
import ctypes as C
import json
import statistics
import time

cl = C.CDLL('libOpenCL.so.1')
P, U, Z, I = C.c_void_p, C.c_uint, C.c_size_t, C.c_int

def bind(name, result, *arguments):
    fn = getattr(cl, name)
    fn.restype, fn.argtypes = result, arguments
    return fn

get_platforms = bind('clGetPlatformIDs', I, U, C.POINTER(P), C.POINTER(U))
get_devices = bind('clGetDeviceIDs', I, P, C.c_ulong, U, C.POINTER(P), C.POINTER(U))
get_info = bind('clGetDeviceInfo', I, P, U, Z, P, C.POINTER(Z))
create_context = bind('clCreateContext', P, P, U, C.POINTER(P), P, P, C.POINTER(I))
create_queue = bind('clCreateCommandQueue', P, P, P, C.c_ulong, C.POINTER(I))
create_buffer = bind('clCreateBuffer', P, P, C.c_ulong, Z, P, C.POINTER(I))
create_program = bind('clCreateProgramWithSource', P, P, U, C.POINTER(C.c_char_p), C.POINTER(Z), C.POINTER(I))
build_program = bind('clBuildProgram', I, P, U, C.POINTER(P), C.c_char_p, P, P)
get_build_info = bind('clGetProgramBuildInfo', I, P, P, U, Z, P, C.POINTER(Z))
create_kernel = bind('clCreateKernel', P, P, C.c_char_p, C.POINTER(I))
set_arg = bind('clSetKernelArg', I, P, U, Z, P)
write_buffer = bind('clEnqueueWriteBuffer', I, P, P, U, Z, Z, P, U, P, P)
read_buffer = bind('clEnqueueReadBuffer', I, P, P, U, Z, Z, P, U, P, P)
enqueue = bind('clEnqueueNDRangeKernel', I, P, P, U, P, C.POINTER(Z), P, U, P, C.POINTER(P))
finish = bind('clFinish', I, P)
get_profile = bind('clGetEventProfilingInfo', I, P, U, Z, P, C.POINTER(Z))

def check(status):
    if status != 0:
        raise RuntimeError(f'OpenCL error {status}')

def info(device, key):
    count = Z()
    check(get_info(device, key, 0, None, C.byref(count)))
    value = C.create_string_buffer(count.value)
    check(get_info(device, key, count.value, value, None))
    return value.value.decode()

count = U()
check(get_platforms(0, None, C.byref(count)))
platforms = (P * count.value)()
check(get_platforms(count, platforms, None))
device = None
for platform in platforms:
    n = U()
    status = get_devices(platform, 4, 0, None, C.byref(n))
    if status == -1:
        continue
    check(status)
    candidates = (P * n.value)()
    check(get_devices(platform, 4, n, candidates, None))
    for candidate in candidates:
        vendor = U()
        check(get_info(candidate, 0x1001, C.sizeof(vendor), C.byref(vendor), None))
        if vendor.value == 0x10de:
            device = P(candidate)
            break
    if device:
        break
if device is None:
    raise SystemExit('No NVIDIA OpenCL GPU found; CPU fallback is not used.')

report = {'device': info(device, 0x102b), 'device_version': info(device, 0x102f),
          'driver': info(device, 0x102d), 'extensions': info(device, 0x1030),
          'workload': 'out[i] = 2 * a[i] + b[i]; FP32; 1048576 elements; 32 timed iterations',
          'caveat': 'Small standalone memory/compute probe; this is not an LLM benchmark or an offload implementation.'}
resources = []
def created(kind, handle, error):
    check(error.value)
    if not handle:
        raise RuntimeError(f'No {kind} handle')
    resources.append((kind, handle))
    return handle

try:
    error = I()
    context = created('Context', create_context(None, 1, C.byref(device), None, None, C.byref(error)), error)
    queue = created('CommandQueue', create_queue(context, device, 2, C.byref(error)), error)
    n = 1 << 20
    size = n * 4
    a = (C.c_float * n)(*(float(i & 255) for i in range(n)))
    b = (C.c_float * n)(*(float(i & 63) for i in range(n)))
    out = (C.c_float * n)()
    buffers = [created('MemObject', create_buffer(context, 1, size, None, C.byref(error)), error) for _ in range(3)]
    start = time.perf_counter()
    check(write_buffer(queue, buffers[0], 1, 0, size, a, 0, None, None))
    check(write_buffer(queue, buffers[1], 1, 0, size, b, 0, None, None))
    report['host_to_device_seconds'] = time.perf_counter() - start
    source = C.c_char_p(b'__kernel void axpy(__global const float *a, __global const float *b, __global float *out) { size_t i = get_global_id(0); out[i] = 2.0f * a[i] + b[i]; }')
    program = created('Program', create_program(context, 1, C.byref(source), None, C.byref(error)), error)
    status = build_program(program, 1, C.byref(device), b'-cl-std=CL1.2', None, None)
    if status:
        log_size = Z()
        check(get_build_info(program, device, 0x1183, 0, None, C.byref(log_size)))
        log = C.create_string_buffer(log_size.value)
        check(get_build_info(program, device, 0x1183, log_size.value, log, None))
        raise RuntimeError(log.value.decode())
    kernel = created('Kernel', create_kernel(program, b'axpy', C.byref(error)), error)
    for index, buffer in enumerate(buffers):
        argument = P(buffer)
        check(set_arg(kernel, index, C.sizeof(argument), C.byref(argument)))
    global_size = Z(n)
    check(enqueue(queue, kernel, 1, None, C.byref(global_size), None, 0, None, None))
    check(finish(queue))
    events = []
    start = time.perf_counter()
    for _ in range(32):
        event = P()
        check(enqueue(queue, kernel, 1, None, C.byref(global_size), None, 0, None, C.byref(event)))
        resources.append(('Event', event))
        events.append(event)
    check(finish(queue))
    report['timed_queue_wall_seconds'] = time.perf_counter() - start
    durations = []
    for event in events:
        begin, end = C.c_ulonglong(), C.c_ulonglong()
        check(get_profile(event, 0x1282, 8, C.byref(begin), None))
        check(get_profile(event, 0x1283, 8, C.byref(end), None))
        durations.append((end.value - begin.value) / 1e9)
    report['kernel_seconds'] = durations
    report['kernel_median_effective_GB_s'] = 3 * size / statistics.median(durations) / 1e9
    start = time.perf_counter()
    check(read_buffer(queue, buffers[2], 1, 0, size, out, 0, None, None))
    report['device_to_host_seconds'] = time.perf_counter() - start
    report['mismatched_elements'] = sum(v != 2 * (i & 255) + (i & 63) for i, v in enumerate(out))
    report['checked_elements'] = n
    report['passed'] = report['mismatched_elements'] == 0
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit(1)
finally:
    for kind, handle in reversed(resources):
        bind('clRelease' + kind, I, P)(handle)
