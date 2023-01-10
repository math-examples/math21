import numpy as np
import tensorflow as tf

# Random input data to test
from m21py import *


def train():
    np.random.seed(21)
    tf.set_random_seed(21)

    in_data = np.random.randn(4, 416, 416, 3) * 120.56
    data_x_input = np.transpose(in_data, (0, 3, 1, 2)).copy()
    data_x_input = data_x_input.astype(np.float32)
    data_x_p = math21_python_convert_ndarray_to_c_array(data_x_input)

    m21 = Math21Recognize()

    function_form = "/home/mathxyz/workspace/m2tf/models/my_captcha_train.yolov3.cfg"
    function_paras = "/home/mathxyz/workspace/captcha/app/my_captcha/backup/my_captcha_train.theta.backup"
    # fnet = m21.math21_ml_function_net_create_from_file(function_form.encode('utf-8'), None, 0)
    fnet = m21.math21_ml_function_net_create_from_file(function_form.encode('utf-8'), function_paras.encode('utf-8'), 0)

    # m21.math21_ml_function_net_data_feed(fnet, data_x_p, None)

    m21.math21_ml_function_net_set_mbs(fnet, 1)
    prediction_y_p = m21.math21_ml_function_net_predict_input(fnet, data_x_p)
    # prediction_y = math21_python_convert_c_array_to_ndarray(prediction_y_p, (4, 18, 52, 52), m21type.m21_type_NumR32.value)
    # print(prediction_y)

    # loss_value = m21.math21_ml_function_net_train_one_mini_batch_in_function(fnet)
    # print(loss_value)

    # m21.math21_ml_function_net_node_log_by_name(fnet, 0, "data_x".encode('utf-8'))
    # m21.math21_ml_function_net_node_log_by_name(fnet, 106, "conv2d/K".encode('utf-8'))
    # m21.math21_ml_function_net_node_log_by_name(fnet, 2, "conv2d/y".encode('utf-8'))
    # var_data_p = m21.math21_ml_function_net_node_get_data_to_cpu(fnet, 106, "conv2d/K".encode('utf-8'))
    # var_data_p = m21.math21_ml_function_net_node_get_data_to_cpu(fnet, 106, "*/K".encode('utf-8'))

    # var_data_p = m21.math21_ml_function_net_node_get_data_to_cpu(fnet, 106, "*/y".encode('utf-8'))
    # var_data = math21_python_convert_rawtensor_to_ndarray(var_data_p, (4, 18, 52, 52))

    # var_data_p = m21.math21_ml_function_net_node_get_rawtensor_to_cpu(fnet, 1, "*/y".encode('utf-8'))
    # var_data = math21_python_convert_rawtensor_to_ndarray(var_data_p)
    var_data_p = m21.math21_ml_function_net_node_get_rawtensor_to_cpu(fnet, 2, "*/y".encode('utf-8'))
    # var_data = math21_python_convert_rawtensor_to_ndarray(var_data_p)
    print(var_data.shape)
    print(var_data[0, 0, 0, :10])

    # print(in_data.shape)
    # print(in_data)


train()
