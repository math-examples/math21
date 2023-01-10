import ctypes

import numpy as npori
from typing import Optional, Tuple
from .m_types import *


def math21_python_convert_type_to_dtype(type):
    if type == m21type.m21_type_NumN.value or type == m21type.m21_type_TenN.value:
        # dt = npori.dtype(int)
        dt = npori.uint32
        byteSize = 4
    elif type == m21type.m21_type_NumZ.value or type == m21type.m21_type_TenZ.value:
        # dt = npori.dtype(int)
        dt = npori.int32
        byteSize = 4
    elif type == m21type.m21_type_NumR.value or type == m21type.m21_type_TenR.value:
        dt = npori.float64
        byteSize = 8
    elif type == m21type.m21_type_NumR32.value:
        dt = npori.float32
        byteSize = 4
    else:
        print('type = {}'.format(type))
        raise ValueError("type not support")
    # print('type = {}, dt = {}'.format(m21type(type).name, dt))
    return dt, byteSize


def math21_python_convert_dtype_to_type(dt):
    if dt == npori.uint32:
        type = m21type.m21_type_NumN.value
        byteSize = 4
    elif dt == npori.int32:
        type = m21type.m21_type_NumZ.value
        byteSize = 4
    elif dt == npori.float32:
        type = m21type.m21_type_NumR32.value
        byteSize = 4
    elif dt == npori.float64:
        type = m21type.m21_type_NumR.value
        byteSize = 8
    else:
        raise ValueError("type not support")
    return type, byteSize


def ctypes2buffer(cptr, length):
    # if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
    #     raise TypeError('expected char pointer')
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError('memmove failed')
    return res


def ctypes2buffer_3(cptr):
    res = bytearray(cptr.raw)
    return res


def math21_python_convert_ndarray_to_c_array(x):
    x_bytes = bytearray(x)
    nBytes = len(x_bytes)
    # bytes = x.tobytes()
    # print(x_bytes==bytes)
    x_p = (ctypes.c_char * nBytes).from_buffer(x_bytes)
    return x_p


def math21_python_convert_ndarray_to_float_p_00(x):
    c_float_p = ctypes.POINTER(ctypes.c_float)
    x = x.astype(npori.float32)
    x_p = x.ctypes.data_as(c_float_p)
    return x_p


# int: 32 or 64
def math21_python_convert_ndarray_to_int_p(x):
    assert 0
    x = x.astype(npori.int)
    return math21_python_convert_ndarray_to_c_array(x)


# create
def math21_python_convert_ndarray_to_tensor_with_create(x):
    type, _ = math21_python_convert_dtype_to_type(x.dtype)

    d = npori.array(x.shape)
    dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumN.value)
    d = d.astype(dt)
    d_p = math21_python_convert_ndarray_to_c_array(d)
    d = math21_tensor_1d_create(m21type.m21_type_TenN.value, d.size, d_p, 0)

    # bytes = x.tobytes()
    # print(x_bytes==bytes)
    x_bytes = bytearray(x)
    nBytes = len(x_bytes)
    x_p = (ctypes.c_char * nBytes).from_buffer(x_bytes)
    x_p = ctypes.cast(x_p, ctypes.c_void_p)
    y = math21_tensor_nd_create(type, d, x_p, 1)
    math21_tensor_destroy(d)
    return y


# create
def math21_python_tensor_create_from_ndarray(x):
    return math21_python_convert_ndarray_to_tensor_with_create(x)


def math21_python_convert_tensor_to_1d_ndarray(rawtensor):
    d = math21_tensor_copy_shape(rawtensor)
    n = math21_tensor_size(rawtensor)
    dt, byteSize = math21_python_convert_type_to_dtype(rawtensor.type)
    nBytes = byteSize * n
    x_p = math21_tensor_data(rawtensor)
    x_bytes = ctypes2buffer(x_p, nBytes)
    x_vector = npori.frombuffer(x_bytes, dtype=dt)
    math21_tensor_destroy(d)
    return x_vector


def math21_python_convert_tensor_to_ndarray(rawtensor):
    x_vector = math21_python_convert_tensor_to_1d_ndarray(rawtensor)
    d = math21_tensor_copy_shape(rawtensor)
    newshape = math21_python_convert_tensor_to_1d_ndarray(d)
    x = npori.reshape(x_vector, newshape)
    math21_tensor_destroy(d)
    return x


def math21_python_convert_ndarray_to_tensor(x):
    shapeHeader = npori.array([0])
    shapeHeader = npori.append(shapeHeader, x.ndim)
    shapeHeader = npori.append(shapeHeader, x.shape)
    shapeHeader = npori.append(shapeHeader, x.size)
    shapeHeader = shapeHeader.astype(npori.uint32)
    shapeHeader_bytes = shapeHeader.tobytes()
    io_out = open("sin_data.bin", "wb")
    io_out.write(shapeHeader_bytes)
    bytes = x.tobytes()
    io_out.write(bytes)

    # print(len(bytes))

    # io_in = open("sin_data.bin", "rb")
    # shapeHeader_bytes_deserialized = io_in.read()
    # print(len(shapeHeader_bytes_deserialized))
    # print(len(shapeHeader_bytes_deserialized))
    # print(len(shapeHeader_bytes_deserialized))
    # print(len(shapeHeader_bytes_deserialized))
    #
    # deserialized_bytes = npori.frombuffer(bytes, dtype=npori.single)
    # print(len(deserialized_bytes))
    # deserialized_x = npori.reshape(deserialized_bytes, newshape=(2, 2))
    # assert npori.array_equal(x, deserialized_x), "Deserialization failed..."


def math21_python_save_ndarray(x, path):
    dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumR.value)
    x = x.astype(dt)
    x_raw = math21_python_tensor_create_from_ndarray(x)
    math21_point_save(x_raw, path.encode('utf-8'))
    math21_tensor_destroy(x_raw)

def math21_python_load_ndarray(path):
    x_raw = math21_point_load(path.encode('utf-8'))
    return math21_python_convert_tensor_to_ndarray(x_raw)


