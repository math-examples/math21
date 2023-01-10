from .m_types import *

math21_test_c_ad_sin_dn = m21_lib.math21_test_c_ad_sin_dn
math21_test_c_ad_sin_dn.argtypes = [c_double, c_int32]
math21_test_c_ad_sin_dn.restype = c_double

math21_test_c_ad_sin_taylor_appr_dn = m21_lib.math21_test_c_ad_sin_taylor_appr_dn
math21_test_c_ad_sin_taylor_appr_dn.argtypes = [c_double, c_int32]
math21_test_c_ad_sin_taylor_appr_dn.restype = c_double

math21_test_ad_tanh_like_dn = m21_lib.math21_test_ad_tanh_like_dn
math21_test_ad_tanh_like_dn.argtypes = [m21point, c_int32]
math21_test_ad_tanh_like_dn.restype = m21point

math21_test_ad_logsumexp_like = m21_lib.math21_test_ad_logsumexp_like
math21_test_ad_logsumexp_like.argtypes = [m21point]
math21_test_ad_logsumexp_like.restype = m21point

math21_test_ad_gmm_log_likelihood_dn = m21_lib.math21_test_ad_gmm_log_likelihood_dn
math21_test_ad_gmm_log_likelihood_dn.argtypes = [m21point, m21point, c_int32, c_int32, c_int32]
math21_test_ad_gmm_log_likelihood_dn.restype = m21point

math21_test_ad_get_f_gmm_log_likelihood = m21_lib.math21_test_ad_get_f_gmm_log_likelihood
math21_test_ad_get_f_gmm_log_likelihood.argtypes = [m21point, m21point, c_int32, c_int32]
math21_test_ad_get_f_gmm_log_likelihood.restype = m21point

math21_test_ad_get_f_rnn_predict = m21_lib.math21_test_ad_get_f_rnn_predict
math21_test_ad_get_f_rnn_predict.argtypes = [m21point, m21point, c_int32, c_int32, c_int32]
math21_test_ad_get_f_rnn_predict.restype = m21point

math21_test_ad_get_f_rnn_part_log_likelihood = m21_lib.math21_test_ad_get_f_rnn_part_log_likelihood
math21_test_ad_get_f_rnn_part_log_likelihood.argtypes = [m21point, m21point]
math21_test_ad_get_f_rnn_part_log_likelihood.restype = m21point

math21_test_ad_get_f_lstm_predict = m21_lib.math21_test_ad_get_f_lstm_predict
math21_test_ad_get_f_lstm_predict.argtypes = [m21point, m21point, c_int32, c_int32, c_int32]
math21_test_ad_get_f_lstm_predict.restype = m21point

math21_point_ad_grad = m21_lib.math21_point_ad_grad
math21_point_ad_grad.argtypes = [m21point, m21point]
math21_point_ad_grad.restype = m21point

math21_point_ad_hessian_vector_product = m21_lib.math21_point_ad_hessian_vector_product
math21_point_ad_hessian_vector_product.argtypes = [m21point, m21point, m21point]
math21_point_ad_hessian_vector_product.restype = m21point

math21_point_ad_fv = m21_lib.math21_point_ad_fv
math21_point_ad_fv.argtypes = [m21point]
math21_point_ad_fv.restype = c_void_p

math21_ad_clear_graph = m21_lib.math21_ad_clear_graph
math21_ad_clear_graph.argtypes = []