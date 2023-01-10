from .ad import *
from .m_convert import *


def math21_python_ndarray_type_to_NumR(x_1d):
    dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumR.value)
    return x_1d.astype(dt)


def math21_python_ad_point_set(px, x_1d):
    x_1d = math21_python_ndarray_type_to_NumR(x_1d)
    math21_python_set_ndarray_to_ad_point(px, x_1d)


def math21_python_create_ad_point_input_from_ndarray(params):
    params_raw = math21_python_tensor_create_from_ndarray(params)
    px = math21_point_create_ad_point_input(params_raw)
    math21_tensor_destroy(params_raw)
    return px


def math21_python_create_ad_point_const_from_ndarray(params):
    params_raw = math21_python_tensor_create_from_ndarray(params)
    px = math21_point_create_ad_point_const(params_raw)
    math21_tensor_destroy(params_raw)
    return px


def math21_python_set_ndarray_to_ad_point(px, params):
    params_raw = math21_python_tensor_create_from_ndarray(params)
    math21_point_ad_point_set_value(px, params_raw)
    math21_tensor_destroy(params_raw)


def math21_python_get_ndarray_from_ad_point(px):
    x_raw = math21_point_ad_point_get_value(px)
    return math21_python_convert_tensor_to_ndarray(x_raw)
