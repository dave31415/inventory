import numpy as np
import cvxopt
import scipy


def identity_spmatrix(n):
    """
    :param n: dimension
    :return: cvxopt sparse identity matrix
    """
    return cvxopt.spmatrix(1.0, range(n), range(n))


def spmatrix2np(spmat):
    """
    Convert a matrix or spmatrix to numpy 2D array
    :param spmat: matrix or spmatrix
    :return: numpy 2D array of type float64
    """
    return np.asarray(cvxopt.matrix(spmat)).squeeze()


def first_deriv_matrix(n):
    """
    Matrix which computes backward difference
    when applied to a vector
    The return vector is shorter by 1 as the
    first element's derivative is not defined
    Copied form cvxpy example code
    :param n: dimension of matrix
    :return: the first derivative from backward difference
    """
    e = np.mat(np.ones((1, n)))
    D = scipy.sparse.spdiags(np.vstack((-e, e)), range(2), n-1, n)
    D_coo = D.tocoo()
    D = cvxopt.spmatrix(D_coo.data, D_coo.row.tolist(), D_coo.col.tolist())
    return D


def get_step_function_matrix(n):
    """
    Upper/lower triangular with all ones
    Can be used for cumulative sum with matrix op
    :param n:
    :return:
    """
    step = identity_spmatrix(n)
    for i in xrange(n):
        for j in xrange(n):
            if i >= j:
                step[i, j] = 1.0
    return step


def get_time_shift_matrix(n, shift):
    """
    Returns a matrix which when multiplied by a vector
    shifts it to the right by some number of indices
    back-filling with zeros
    :param n: dimension of vector
    :param shift: number of indices being shifted
    :return:
    """
    return cvxopt.spmatrix(1.0, shift+np.arange(n), np.arange(n))
