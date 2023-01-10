from typing import Optional, Tuple
from .m_convert import *

# global_batch_size = 50
# global_n_input = 1  # input is sin(x)
# global_n_steps = 100  # timesteps
# global_n_hidden = 100  # hidden layer num of features
# global_n_outputs = 50  # output is sin(x+1)

global_batch_size = 5
global_n_input = 1  # input is sin(x)
global_n_steps = 2  # timesteps
global_n_hidden = 3  # hidden layer num of features
global_n_outputs = 3  # output is sin(x+1)


def generate_sample(f: Optional[float] = 1.0, t0: Optional[float] = None, batch_size: int = 1,
                    predict: int = 50, samples: int = 100) -> Tuple[npori.ndarray, npori.ndarray, npori.ndarray, npori.ndarray]:
    """
    Generates data samples.

    :param f: The frequency to use for all time series or None to randomize.
    :param t0: The time offset to use for all time series or None to randomize.
    :param batch_size: The number of time series to generate.
    :param predict: The number of future samples to generate.
    :param samples: The number of past (and current) samples to generate.
    :return: Tuple that contains the past times and values as well as the future times and values. In all outputs,
             each row represents one time series of the batch.
    """
    Fs = 100

    T = npori.empty((batch_size, samples))
    Y = npori.empty((batch_size, samples))
    FT = npori.empty((batch_size, predict))
    FY = npori.empty((batch_size, predict))

    _t0 = t0
    for i in range(batch_size):
        t = npori.arange(0, samples + predict) / Fs
        # print(t)
        if _t0 is None:
            t0 = npori.random.rand() * 2 * npori.pi
        else:
            t0 = _t0 + i / float(batch_size)

        freq = f
        if freq is None:
            freq = npori.random.rand() * 3.5 + 0.5

        y = npori.sin(2 * npori.pi * freq * (t + t0))

        T[i, :] = t[0:samples]
        Y[i, :] = y[0:samples]

        FT[i, :] = t[samples:samples + predict]
        FY[i, :] = y[samples:samples + predict]

    return T, Y, FT, FY

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # noinspection PyUnresolvedReferences
    # import seaborn as sns

    t, y, t_next, y_next = generate_sample(f=None, t0=None, batch_size=3)
    print(t.shape)

    n_tests = t.shape[0]
    for i in range(0, n_tests):
        plt.subplot(n_tests, 1, i + 1)
        plt.plot(t[i, :], y[i, :])
        plt.plot(npori.append(t[i, -1], t_next[i, :]), npori.append(y[i, -1], y_next[i, :]), color='red', linestyle=':')

    # plt.xlabel('time [t]')
    # plt.ylabel('signal')
    # plt.show()

    x = npori.array([[0, 1], [2, 3]], npori.double)
    math21_python_convert_ndarray_to_tensor(x)
