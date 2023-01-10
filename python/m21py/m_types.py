from ctypes import *
import sys
import enum

if sys.platform == 'linux':
    m21_lib_path = "/home/mathxyz/workspace/libmath21.so"
    star_lib_path = "/home/mathxyz/workspace/libstar.so"
else:
    m21_lib_path = "D:/mathxyz/workspace/libmath21.dll"
    star_lib_path = "D:/mathxyz/workspace/libstar.dll"

m21_lib = CDLL(m21_lib_path, RTLD_GLOBAL)


# creating enumerations using class
class m21type(enum.Enum):
    m21_type_none = 0
    m21_type_default = 1
    m21_type_NumN = 2
    m21_type_NumZ = 3
    m21_type_NumR = 4
    m21_type_Seqce = 5
    m21_type_Tensor = 6
    m21_type_Digraph = 7
    m21_type_vector_float_c = 8
    m21_type_vector_char_c = 9
    m21_type_NumR32 = 10
    m21_type_NumR64 = 11
    m21_type_TenN = 12
    m21_type_TenZ = 13
    m21_type_TenR = 14
    m21_type_PointAd = 15
    m21_type_NumSize = 16
    m21_type_NumN8 = 17
    m21_type_TenN8 = 18
    m21_type_string = 19
    m21_type_TenStr = 20
    m21_type_SeqTenN = 21
    m21_type_SeqTenZ = 22
    m21_type_SeqTenR = 23
    m21_type_SeqSeqStr = 24


class m21point(Structure):
    _fields_ = [("type", c_int),
                ("p", c_void_p),
                ("refCount", c_void_p)]


math21_tensor_1d_create = m21_lib.math21_tensor_1d_create
math21_tensor_1d_create.argtypes = [c_int, c_int, c_void_p, c_int]
math21_tensor_1d_create.restype = m21point

math21_tensor_nd_create = m21_lib.math21_tensor_nd_create
math21_tensor_nd_create.argtypes = [c_int, m21point, c_void_p, c_int]
math21_tensor_nd_create.restype = m21point

def math21_tensor_destroy(x):
    f = m21_lib.math21_tensor_destroy
    f.argtypes = [m21point]
    f(x)
    x.type = 0
    x.p = 0


math21_point_destroy = m21_lib.math21_point_destroy
math21_point_destroy.argtypes = [m21point]
math21_point_destroy.restype = m21point

math21_point_log = m21_lib.math21_point_log
math21_point_log.argtypes = [m21point]

math21_point_save = m21_lib.math21_point_save
math21_point_save.argtypes = [m21point, c_char_p]

math21_point_load = m21_lib.math21_point_load
math21_point_load.argtypes = [c_char_p]
math21_point_load.restype = m21point

math21_tensor_copy_shape = m21_lib.math21_tensor_copy_shape
math21_tensor_copy_shape.argtypes = [m21point]
math21_tensor_copy_shape.restype = m21point

math21_tensor_size = m21_lib.math21_tensor_size
math21_tensor_size.argtypes = [m21point]
math21_tensor_size.restype = c_int

math21_tensor_data = m21_lib.math21_tensor_data
math21_tensor_data.argtypes = [m21point]
math21_tensor_data.restype = c_void_p

math21_point_create_ad_point_const = m21_lib.math21_point_create_ad_point_const
math21_point_create_ad_point_const.argtypes = [m21point]
math21_point_create_ad_point_const.restype = m21point

math21_point_create_ad_point_input = m21_lib.math21_point_create_ad_point_input
math21_point_create_ad_point_input.argtypes = [m21point]
math21_point_create_ad_point_input.restype = m21point

math21_point_ad_point_set_value = m21_lib.math21_point_ad_point_set_value
math21_point_ad_point_set_value.argtypes = [m21point, m21point]
math21_point_ad_point_set_value.restype = c_void_p

math21_point_ad_point_get_value = m21_lib.math21_point_ad_point_get_value
math21_point_ad_point_get_value.argtypes = [m21point]
math21_point_ad_point_get_value.restype = m21point
