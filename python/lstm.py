"""Implements the long-short term memory character model.
This version vectorizes over multiple examples, but each string
has a fixed length."""

from __future__ import absolute_import
from __future__ import print_function
from builtins import range
from os.path import dirname, join
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.special import logsumexp

from autograd.misc.optimizers import adam
from autograd.misc.flatten import flatten_func, flatten
from rnn import string_to_one_hot, one_hot_to_string, \
    build_dataset, sigmoid, concat_and_multiply, get_y_from_rnn_log_likelihood

from m21py.m_ndarray import *
from m21py.m_convert import *
import time

t = time.time()


def init_f_params(input_size, state_size, output_size,
                  param_scale=0.01, rs=npr.RandomState(0)):
    def rp(*shape):
        return rs.randn(*shape) * param_scale

    return {'init hiddens': rp(1, state_size),
            'change': rp(input_size + state_size + 1, state_size),
            'predict': rp(state_size + 1, output_size),
            'init cells': rp(1, state_size),
            'forget': rp(input_size + state_size + 1, state_size),
            'ingate': rp(input_size + state_size + 1, state_size),
            'outgate': rp(input_size + state_size + 1, state_size),
            }


def f_predict_ori(params, inputs):
    def update_f(input, hiddens, cells):
        change = np.tanh(concat_and_multiply(params['change'], input, hiddens))
        forget = sigmoid(concat_and_multiply(params['forget'], input, hiddens))
        ingate = sigmoid(concat_and_multiply(params['ingate'], input, hiddens))
        outgate = sigmoid(concat_and_multiply(params['outgate'], input, hiddens))
        cells = cells * forget + ingate * change
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['predict'], hiddens)
        return output - logsumexp(output, axis=1, keepdims=True)  # Normalize log-probs.

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['init hiddens'], num_sequences, axis=0)
    cells = np.repeat(params['init cells'], num_sequences, axis=0)

    output = [hiddens_to_output_probs(hiddens)]
    for input in inputs:  # Iterate over time steps.
        hiddens, cells = update_f(input, hiddens, cells)
        output.append(hiddens_to_output_probs(hiddens))
    return output


def f_log_likelihood(params, inputs, targets):
    logprobs = f_predict(params, inputs)
    loglik = 0.0
    num_time_steps, num_examples, _ = inputs.shape
    for t in range(num_time_steps):
        loglik += np.sum(logprobs[t] * targets[t])
    return loglik / (num_time_steps * num_examples)


# is_my = 0
is_my = 1
input_size = 128
output_size = 128
state_size = 40
# input_size = 2
# output_size = 3
# state_size = 4
param_scale = 0.01

# num_iters=1000
# num_iters = 300
num_iters = 10
if __name__ == '__main__':
    num_chars = 128

    # seqs = string_to_one_hot("", num_chars)[:, np.newaxis, :]
    # dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumR.value)
    # seqs = seqs.astype(dt)
    # seqs_raw = math21_python_tensor_create_from_ndarray(seqs)
    # #
    # exit(0)
    # text = ""
    # text += chr(99)
    # print(text)
    # exit(0)

    # Learn to predict our own source code.
    # text_filename = join(dirname(__file__), 'f.py')
    # text_filename = 'C:/mathxyz/workspace/autograd-master/examples/rnn.py'
    text_filename = 'D:/mathxyz/workspace/autograd-master/examples/rnn.py'
    # text_filename = 'D:/mathxyz/workspace/z3.txt'
    # text_filename = '/home/mathxyz/workspace/autograd-master/examples/rnn.py'
    # sequence_length = 30
    sequence_length = 5
    train_inputs = build_dataset(text_filename, sequence_length=sequence_length,
                                 alphabet_size=num_chars, max_lines=60)

    # bin_filename = 'D:/mathxyz/workspace/z11.bin'
    # bin_filename = '/home/mathxyz/workspace/z.bin'
    # math21_python_save_ndarray(train_inputs, bin_filename)
    # train_inputs = math21_python_load_ndarray(bin_filename)
    # print(train_inputs)
    # exit(0)

    init_params = init_f_params(input_size=input_size, state_size=state_size, output_size=output_size,
                                param_scale=param_scale)

    # You must make sure flatten works such that init_x_1d_from_flatten equals init_x_1d
    init_x_1d_from_flatten, unflatten = flatten(init_params)

    init_x_1d = np.concatenate(np.ravel(init_params[k]) for k in init_params)
    # bin_filename = 'D:/mathxyz/workspace/z23.bin'
    # bin_filename = 'D:/mathxyz/workspace/z14.bin'
    # bin_filename = 'C:/mathxyz/workspace/z3.bin'
    # bin_filename = '/home/mathxyz/workspace/z3.bin'
    # math21_python_save_ndarray(init_x_1d, bin_filename)
    # init_x_1d = math21_python_load_ndarray(bin_filename)
    # exit(0)

    # init_x_1d2 = np.asarray(init_params)

    px = math21_python_create_ad_point_input_from_ndarray(init_x_1d)
    ptrain_inputs = math21_python_create_ad_point_const_from_ndarray(train_inputs)

    ppredict = math21_test_ad_get_f_lstm_predict(px, ptrain_inputs,
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
        ppredict = math21_test_ad_get_f_lstm_predict(px, ptrain_inputs,
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

    print("Training lstm...")

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
    print("Generating text from lstm...")
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
        print("Generating text from lstm my...")
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
