"""Implements a Gaussian mixture model, in which parameters are fit using
   gradient descent.  This example runs on 2-dimensional data, but the model
   works on arbitrarily-high dimension."""

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

# import numpy as np
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, hessian_vector_product, jacobian
from scipy.optimize import minimize
from autograd.scipy.special import logsumexp
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.misc.flatten import flatten_func, flatten
#from examples.data import make_pinwheel

# import numpy as np
# import matplotlib.pyplot as plt
from m21py.ad import *
from m21py.m_convert import *
from m21py.m_ndarray import *
import time

t = time.time()

# n_component = 10
n_component = 3
n_feature = 2

def make_pinwheel(radial_std, tangential_std, num_classes, num_per_class, rate,
                  rs=npr.RandomState(0)):
    """Based on code by Ryan P. Adams."""
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    features = rs.randn(num_classes*num_per_class, 2) \
               * np.array([radial_std, tangential_std])
    features[:, 0] += 1
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return np.einsum('ti,tij->tj', features, rotations)

def gmm_log_likelihood_dn(params, data, n):
    params_raw = math21_python_tensor_create_from_ndarray(params)
    data_raw = math21_python_tensor_create_from_ndarray(data)
    y_raw = math21_test_ad_gmm_log_likelihood_dn(params_raw, data_raw, n_component, n_feature, n)
    if y_raw.type == 0:
        raise ValueError("emtpy")
    math21_tensor_destroy(params_raw)
    math21_tensor_destroy(data_raw)
    y = math21_python_convert_tensor_to_ndarray(y_raw)
    return y


def get_x_from_gmm_log_likelihood(params):
    params_raw = math21_python_tensor_create_from_ndarray(params)
    px = math21_point_create_ad_point_input(params_raw)
    math21_tensor_destroy(params_raw)
    return px

def get_y_from_gmm_log_likelihood(params_point, data):
    data_raw = math21_python_tensor_create_from_ndarray(data)
    py = math21_test_ad_get_f_gmm_log_likelihood(params_point, data_raw, n_component, n_feature)
    if py.type == 0:
        raise ValueError("emtpy")
    math21_tensor_destroy(data_raw)
    return py


# 1*2, 2*1, 2*1*1
# gfg1 = np.array([[1, 2, 3], [1, 2, 3]])
# gfg2 = np.array([[3], [4]])
# gfg3 = np.array([[[3]], [[4]]])

# print(np.broadcast_arrays(gfg1, gfg2,gfg3))
# exit(0)
cov = np.array([[11, 2, 3],
                [2, 24, 5],
                [3, 5, 36]])

mean = np.array([1, 1, 1])
# x = np.array([1, 2, 3])
x = np.array([[1, 2, 3],
              [4, 5, 6]])
x = x.astype(np.float64)

def f(x):
    return logsumexp(x, axis=0)
    # return logsumexp(x)
    # return np.sum(x)


# df = grad(f)
df = jacobian(f)
y = f(x)
dy = df(x)
# print(y)
# print(dy)
# exit(0)

def init_gmm_params(num_components, D, scale, rs=npr.RandomState(0)):
    return {'log proportions': rs.randn(num_components) * scale,
            'means': rs.randn(num_components, D) * scale,
            'lower triangles': np.zeros((num_components, D, D)) + np.eye(D)}


def log_normalize(x):
    return x - logsumexp(x)


def unpack_gmm_params(params):
    normalized_log_proportions = log_normalize(params['log proportions'])
    return normalized_log_proportions, params['means'], params['lower triangles']


# params: {proportion, mean, cov} with shape [n_component, n_component*n_feature, n_component*n_feature*n_feature]
# data: n_data * n_feature, with n_feature = 2
def gmm_log_likelihood(params, data):
    cluster_lls = []
    # print(data)
    # i in n_component
    for log_proportion, mean, cov_sqrt in zip(*unpack_gmm_params(params)):
        cov = np.dot(cov_sqrt.T, cov_sqrt)
        cluster_lls.append(log_proportion + mvn.logpdf(data, mean, cov))
    # np.vstack(cluster_lls): n_component * n_data
    return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))


def plot_ellipse(ax, mean, cov_sqrt, alpha, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    circle_pts = np.vstack([np.cos(angles), np.sin(angles)]).T * 2.0
    cur_pts = mean + np.dot(circle_pts, cov_sqrt)
    plt.plot(cur_pts[:, 0], cur_pts[:, 1], '-', alpha=alpha)
    # ax.plot(cur_pts[:, 0], cur_pts[:, 1], '-', alpha=alpha)


def plot_gaussian_mixture(params, ax):
    for log_proportion, mean, cov_sqrt in zip(*unpack_gmm_params(params)):
        alpha = np.minimum(1.0, np.exp(log_proportion) * 10)
        plot_ellipse(ax, mean, cov_sqrt, alpha)

#################### main

init_params = init_gmm_params(num_components=n_component, D=n_feature, scale=0.1)
# print(init_params)

data = make_pinwheel(radial_std=0.3, tangential_std=0.05, num_classes=3,
                     num_per_class=100, rate=0.4)

# data = make_pinwheel(radial_std=0.3, tangential_std=0.05, num_classes=3,
#                      num_per_class=4, rate=0.4)

# print(data)
mydata = np.array([[-0.39603642, -1.47717844],
                   [0.25644726, -1.2728885],
                   [-0.55655118, -1.45844877],
                   [0.15256218, -1.27596051],
                   [-0.96915379, -0.01378327],
                   [-1.04556437, 0.01938928],
                   [-1.16825597, 0.37942548],
                   [-1.11491919, 0.20318167],
                   [1.34219476, 0.54403161],
                   [0.71343337, 0.83036026],
                   [-0.03691002, 0.23347363],
                   [0.99859678, 0.76817688]])
# mydata = data

def test_gmm():

    def objective(x_nd):
        return -gmm_log_likelihood(x_nd, mydata)


    def f_nd(x_nd):
        return objective(x_nd)


    def df_nd(x_nd):
        return grad(objective)(x_nd)


    flattened_obj, unflatten, flattened_init_params = \
        flatten_func(objective, init_params)


    def f_1d(x_1d):
        y = flatten(f_nd(unflatten(x_1d)))[0]
        return y

    def df_1d(x_1d):
        return flatten(df_nd(unflatten(x_1d)))[0]

    # x_1d = np.array([0.17640523, 0.04001572, 0.0978738, 1., 0., 0.,
    #                  1., 1., 0., 0., 1., 1.,
    #                  0., 0., 1., 0.22408932, 0.1867558, -0.09772779,
    #                  0.09500884, -0.01513572, -0.01032189])

    # y = df_1d(x_1d)
    # print(y)
    # exit(0)

    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)


    # plt.show(block=False)

    def callback(flattened_params):
        params = unflatten(flattened_params)
        print("Log likelihood {}".format(-objective(params)))
        plt.cla()
        # ax.cla()
        # ax.plot(mydata[:, 0], mydata[:, 1], 'k.')
        plt.plot(mydata[:, 0], mydata[:, 1], 'k.')
        ax.set_xticks([])
        ax.set_yticks([])
        plot_gaussian_mixture(params, ax)
        plt.draw()
        plt.pause(1.0 / 60.0)
        plt.show(block=False)


    hessp_f_1d = hessian_vector_product(f_1d)

    minimize(f_1d, flattened_init_params,
             jac=df_1d,
             hessp=hessp_f_1d,
             method='Newton-CG', callback=callback)

def test_gmm_my():

    def objective(x_nd):
        return -gmm_log_likelihood(x_nd, mydata)

    flattened_obj, unflatten, flattened_init_params = \
        flatten_func(objective, init_params)

    # x_1d = np.array([0.17640523, 0.04001572, 0.0978738, 1., 0., 0.,
    #              1., 1., 0., 0., 1., 1.,
    #              0., 0., 1., 0.22408932, 0.1867558, -0.09772779,
    #              0.09500884, -0.01513572, -0.01032189])
    x_1d = flattened_init_params

    px = get_x_from_gmm_log_likelihood(x_1d)
    pv = 1
    # pv = get_x_from_gmm_log_likelihood(x_1d)
    py = get_y_from_gmm_log_likelihood(px, mydata)
    pdy = math21_point_ad_grad(px, py)
    # pddy = math21_point_ad_hessian_vector_product(px, py, pv)

    def f_1d_my_v2(x_1d):
        dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumR.value)
        x_1d = x_1d.astype(dt)
        math21_python_set_ndarray_to_ad_point(px, x_1d)
        math21_point_ad_fv(py)
        return -math21_python_get_ndarray_from_ad_point(py)[0]

    def df_1d_my_v2(x_1d):
        dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumR.value)
        x_1d = x_1d.astype(dt)
        math21_python_set_ndarray_to_ad_point(px, x_1d)
        math21_point_ad_fv(pdy)
        return -math21_python_get_ndarray_from_ad_point(pdy)

    def ddf_1d_my_v2(x_1d):
        dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumR.value)
        x_1d = x_1d.astype(dt)
        # math21_python_set_ndarray_to_ad_point(px, x_1d)
        math21_python_set_ndarray_to_ad_point(pv, x_1d)
        math21_point_ad_fv(pddy)
        return -math21_python_get_ndarray_from_ad_point(pddy)

    def f_1d_my_v1(x_1d):
        dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumR.value)
        x_1d = x_1d.astype(dt)
        return -gmm_log_likelihood_dn(x_1d, mydata, 0)

    def df_1d_my_v1(x_1d):
        dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumR.value)
        x_1d = x_1d.astype(dt)
        return -gmm_log_likelihood_dn(x_1d, mydata, 1)

    # f_1d = f_1d_ori
    # df_1d = df_1d_ori
    # f_1d = f_1d_my_v1 # slow
    # df_1d = df_1d_my_v1
    f_1d = f_1d_my_v2 # fast
    df_1d = df_1d_my_v2
    # ddf_1d = ddf_1d_my_v2

    # y = df_1d(x_1d)
    # print(y)
    # exit(0)

    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)


    # plt.show(block=False)

    def callback(flattened_params):
        params = unflatten(flattened_params)
        print("Log likelihood {}".format(-f_1d(flattened_params)))
        # print("Log likelihood {}".format(-objective(params)))
        plt.cla()
        # ax.cla()
        # ax.plot(mydata[:, 0], mydata[:, 1], 'k.')
        plt.plot(mydata[:, 0], mydata[:, 1], 'k.')
        ax.set_xticks([])
        ax.set_yticks([])
        plot_gaussian_mixture(params, ax)
        plt.draw()
        plt.pause(1.0 / 60.0)
        plt.show(block=False)

    minimize(f_1d, flattened_init_params,
             jac=df_1d,
             # hessp=ddf_1d,
             method='Newton-CG', callback=callback)

    # hessp_f_1d = hessian_vector_product(f_1d)
    # minimize(f_1d, flattened_init_params,
    #          jac=df_1d,
    #          hessp=hessp_f_1d,
    #          method='Newton-CG', callback=callback)

    # todo: use
    # math21_point_destroy(px)
    # math21_point_destroy(py)
    # math21_point_destroy(pdy)

if __name__ == '__main__':
    # test_gmm()
    test_gmm_my()

print('time cost {}'.format(time.time() - t))

