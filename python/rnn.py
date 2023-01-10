"""Implements the long-short term memory character model.
This version vectorizes over multiple examples, but each string
has a fixed length."""

from __future__ import absolute_import
from __future__ import print_function
from builtins import range
import numpy as np0
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.special import logsumexp
from os.path import dirname, join
from autograd.misc.optimizers import adam
from autograd.misc.flatten import flatten_func, flatten

# import numpy as np
# import matplotlib.pyplot as plt
from m21py.ad import *
from m21py.m_convert import *
from m21py.m_ndarray import *
import time

t = time.time()


# math21_test_ad_get_f_rnn_log_likelihood

# x = np0.stack()
# x = np0.array([[1,2],[3,4]])
# x = np0.repeat(x,[2, 3], axis=0)
# print(x)
# exit(0)

### Helper functions #################

def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1.0)  # Output ranges from 0 to 1.


def concat_and_multiply(weights, *args):
    cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
    return np.dot(cat_state, weights)


### Define recurrent neural net #######

def init_f_params(input_size, state_size, output_size,
                    param_scale=0.01, rs=npr.RandomState(0)):
    return {'init hiddens': rs.randn(1, state_size) * param_scale,
            'change': rs.randn(input_size + state_size + 1, state_size) * param_scale,
            'predict': rs.randn(state_size + 1, output_size) * param_scale}


def f_predict_ori(params, inputs):
    def update_f(input, hiddens):
        # xh: num_sequences * (input_size + state_size)
        # paras: input_size + state_size + 1, state_size
        # h = x*wx + h*wh + b
        return np.tanh(concat_and_multiply(params['change'], input, hiddens))

    def hiddens_to_output_probs(hiddens):
        # hiddens: num_sequences * state_size
        # paras: state_size + 1, output_size
        # output: num_sequences * output_size
        output = concat_and_multiply(params['predict'], hiddens)
        return output - logsumexp(output, axis=1, keepdims=True)  # Normalize log-probs.

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['init hiddens'], num_sequences, axis=0)
    output = [hiddens_to_output_probs(hiddens)]

    for input in inputs:  # Iterate over time steps.
        hiddens = update_f(input, hiddens)
        output.append(hiddens_to_output_probs(hiddens))
    # output: time_steps * num_sequences * output_size
    return output


# def my_f_log_likelihood(params, inputs, targets):


def f_log_likelihood(params, inputs, targets):
    logprobs = f_predict_ori(params, inputs)
    # print(len(logprobs))
    # print(inputs.shape)
    # exit(0)
    loglik = 0.0
    num_time_steps, num_examples, _ = inputs.shape
    for t in range(num_time_steps):
        loglik += np.sum(logprobs[t] * targets[t])
    return loglik / (num_time_steps * num_examples)


### Dataset setup ##################

def string_to_one_hot(string, maxchar):
    # print(string)
    """Converts an ASCII string to a one-of-k encoding."""
    ascii = np.array([ord(c) for c in string]).T
    y = np.array(ascii[:, None] == np.arange(maxchar)[None, :], dtype=int)
    # print(ascii)
    # print(y)
    return y


def one_hot_to_string(one_hot_matrix):
    return "".join([chr(np.argmax(c)) for c in one_hot_matrix])


def build_dataset(filename, sequence_length, alphabet_size, max_lines=-1):
    """Loads a text file, and turns each line into an encoded sequence."""
    with open(filename) as f:
        content = f.readlines()
    content = content[:max_lines]
    content = [line for line in content if len(line) > 2]  # Remove blank lines
    seqs = np.zeros((sequence_length, len(content), alphabet_size))
    for ix, line in enumerate(content):
        padded_line = (line + " " * sequence_length)[:sequence_length]
        seqs[:, ix, :] = string_to_one_hot(padded_line, alphabet_size)
    return seqs


def get_y_from_rnn_log_likelihood(logprobs, data_targets):
    data_targets_raw = math21_python_tensor_create_from_ndarray(data_targets)
    py = math21_test_ad_get_f_rnn_part_log_likelihood(logprobs, data_targets_raw)
    if py.type == 0:
        raise ValueError("emtpy")
    math21_tensor_destroy(data_targets_raw)
    return py


is_my = 0
# is_my = 1
input_size = 128
output_size = 128
state_size = 40
# input_size = 2
# output_size = 3
# state_size = 4
param_scale = 0.01

# num_iters=1000
num_iters = 300
# num_iters = 10
if __name__ == '__main__':
    num_chars = 128

    seqs = string_to_one_hot("", num_chars)[:, np.newaxis, :]
    dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumR.value)
    seqs = seqs.astype(dt)
    # seqs_raw = math21_python_tensor_create_from_ndarray(seqs)
    # #
    # exit(0)

    # Learn to predict our own source code.
    # text_filename = join(dirname(__file__), 'f.py')
    text_filename = '/home/mathxyz/workspace/autograd-master/examples/rnn.py'
    # sequence_length = 30
    sequence_length = 5
    train_inputs = build_dataset(text_filename, sequence_length=sequence_length,
                                 alphabet_size=num_chars, max_lines=60)
    # print(train_inputs)
    # math21_python_save_ndarray(train_inputs, '/home/mathxyz/workspace/z.bin')

    # exit(0)

    init_params = init_f_params(input_size=input_size, state_size=state_size, output_size=output_size,
                                  param_scale=param_scale)

    init_x_1d, unflatten = flatten(init_params)

    init_x_1d = np.concatenate(np.ravel(init_params[k]) for k in init_params)

    # init_x_1d2 = np.asarray(init_params)

    px = math21_python_create_ad_point_input_from_ndarray(init_x_1d)
    ptrain_inputs = math21_python_create_ad_point_const_from_ndarray(train_inputs)
    ppredict = math21_test_ad_get_f_rnn_predict(px, ptrain_inputs,
                                                input_size, state_size, output_size)
    # exit(0)
    py = get_y_from_rnn_log_likelihood(ppredict, train_inputs)
    pdy = math21_point_ad_grad(px, py)


    def f_1d(x_1d, iter):
        math21_python_ad_point_set(px, x_1d)
        math21_point_ad_fv(py)
        return -math21_python_get_ndarray_from_ad_point(py)[0]


    def f_predict_my(x_1d, train_inputs):
        math21_python_ad_point_set(px, x_1d)
        math21_python_ad_point_set(ptrain_inputs, train_inputs)
        math21_point_ad_fv(ppredict)
        return math21_python_get_ndarray_from_ad_point(ppredict)


    def f_predict_my_2(x_1d, train_inputs):
        math21_ad_clear_graph()
        px = math21_python_create_ad_point_input_from_ndarray(x_1d)
        ptrain_inputs = math21_python_create_ad_point_const_from_ndarray(train_inputs)
        ppredict = math21_test_ad_get_f_rnn_predict(px, ptrain_inputs,
                                                    input_size, state_size, output_size)
        y = math21_python_get_ndarray_from_ad_point(ppredict)
        math21_point_destroy(px)
        math21_point_destroy(ptrain_inputs)
        math21_point_destroy(ppredict)
        return y


    def df_1d(x_1d, iter):
        math21_python_ad_point_set(px, x_1d)
        math21_point_ad_fv(pdy)
        return -math21_python_get_ndarray_from_ad_point(pdy)


    if is_my:
        f_predict = f_predict_my
    else:
        f_predict = f_predict_ori


    def print_training_prediction(weights):
        print("Training text                         Predicted text")
        logprobs = np.asarray(f_predict(weights, train_inputs))
        for t in range(logprobs.shape[1]):
            training_text = one_hot_to_string(train_inputs[:, t, :])
            predicted_text = one_hot_to_string(logprobs[:, t, :])
            print(training_text.replace('\n', ' ') + "|" +
                  predicted_text.replace('\n', ' '))


    def training_loss(params, iter):
        return -f_log_likelihood(params, train_inputs, train_inputs)


    def callback(weights, iter, gradient):
        if iter % 10 == 0:
            print("Iteration", iter, "Train loss:", training_loss(weights, 0))
            # print_training_prediction(weights)


    def callback_my(weights, iter, gradient):
        if iter % 10 == 0:
            print("Iteration", iter, "Train loss:", f_1d(weights, 0))
            # print_training_prediction(weights)


    if is_my:
        callback = callback_my

    # Build gradient of loss function using autograd.
    training_loss_grad = grad(training_loss)

    print("Training RNN...")

    if not is_my:
        trained_params = adam(training_loss_grad, init_params, step_size=0.1,
                              num_iters=num_iters, callback=callback)
    else:
        x_1d = adam(df_1d, init_x_1d, step_size=0.1,
                    num_iters=num_iters, callback=callback)
        # error unflatten
        trained_params = unflatten(x_1d)

    math21_point_destroy(px)
    math21_point_destroy(ptrain_inputs)
    math21_point_destroy(ppredict)
    math21_point_destroy(py)
    math21_point_destroy(pdy)

    print()
    print("Generating text from RNN...")
    num_letters = 30
    for t in range(20):
        text = ""
        for i in range(num_letters):
            seqs = string_to_one_hot(text, num_chars)[:, np.newaxis, :]
            y = f_predict_ori(trained_params, seqs)
            logprobs = y[-1].ravel()
            text += chr(npr.choice(len(logprobs), p=np.exp(logprobs)))
        print(text)

    if is_my:
        print()
        print("Generating text from RNN my...")
        num_letters = 30
        for t in range(20):
            text = ""
            for i in range(num_letters):
                seqs = string_to_one_hot(text, num_chars)[:, np.newaxis, :]
                seqs = math21_python_ndarray_type_to_NumR(seqs)
                y = f_predict_my_2(x_1d, seqs)
                logprobs = y[-1].ravel()
                text += chr(npr.choice(len(logprobs), p=np.exp(logprobs)))
            print(text)

print('time cost {}'.format(time.time() - t))
