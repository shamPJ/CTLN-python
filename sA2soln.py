from graph2net import graph2net
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from threshlin_ode import threshlin_ode

def sA2soln(sA, T=None, X0=[], epsilon=None, delta=None, theta=None):
    """

    calls: graph2net.m and threshlin_ode.m
    
    :param sA      : matrix of shape (n,n), binary adjacency matrix for a directed graph
    :param T       : array of shape (1,m), amount of time in ode solution for each b-vector, should be a vector of length m
    :param X0      : array of shape (n,), vector of initial conditions in firing rates, (default is random near all-zeros)
    :param epsilon : float, epsilon/delta are the values for weights -1+epsilon, -1-delta in W
    :param delta   : float, epsilon/delta are the values for weights -1+epsilon, -1-delta in W
    :param theta   : array of shape (n,1), vector `b` of external stimulus or scalar value

    :out soln      :
    """

    n = sA.shape[0]    # no. of neurons

    if T == None : T = np.array([100]).reshape(1,-1)
    if len(X0) == 0 : X0 = np.zeros((n,)) + .01*np.random.randn(n);  # break symmetry on init conds
    if epsilon == None : epsilon = .25
    if delta == None :  2*epsilon
    if theta == None : theta = 1

    # create network W from sA
    W = graph2net(sA, epsilon, delta)

    # create external input vector b from theta
    if hasattr(theta, "__len__") == False:
        b = theta*np.ones((n,1))
    else:
        b = theta

    # simulate activity with constant b and initial conds X0
    soln = threshlin_ode(W,b,T,X0)

    # add adjacency matrix to soln struct
    soln['sA'] = sA
    soln['eps'] = epsilon
    soln['delta']= delta

    return soln