import numpy as np

def graph2net(sA, d, e=.25):
    """
    :param sA : matrix of shape (n,n), binary adjacency matrix for a directed graph; if i->j, then sA(j,i) = 1.
    :param e  : float, epsilon value to control synaptic strengths (default: e = .25)
    :param d  : float, delta value for inhibitory weights (default: d = 2*e)

    :out W    : matrix of shape (n,n), weight matrix with zero diagonal and all other entries <0
    :out A    :
    """

    d = 2*e

    # create matrix A from sA
    A = e*(sA>0) - d*(sA<=0)        # 1 -> eps, 0 -> - delta
    A = A - np.diag(np.diag(A));    # reset diagonal to 0

    # create matrix W from A: W = I - 11^t + A
    n = A.shape[0]
    W = np.eye(n) - np.ones((n,n)) + A

    return W, A