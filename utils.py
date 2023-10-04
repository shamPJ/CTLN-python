import matplotlib.pyplot as plt
import numpy as np

def plot_ratecurves(X, time):
    """
    :param X        : array of shape (len(time), n), array of rate vectors x at each time
    :param time     : array of shape (len(time), n),times accompanying X, like soln['time] from output threshlin_ode.py
    """

    n = X.shape[1] # no. of neurons
    print(X)

    for i in range(n):
        plt.plot(time, X[:,i])
    plt.show()

def plot_projection(X, proj=None, interval=None, color='k'):

    """
    Random  2D projection of a solution matrix X.

    :param X        : array of shape (len(time), n), array of rate vectors x at each time
    :param proj     : array of shape (n, 2), vectors of 2 projection directions
    :param interval : tuple, an interval in [0,1] specifying which part of the recording -> ex: input [.75,1] for last 1/4 of simulation time
    :param color    : string, defines line color
    """

    n = X.shape[1]   # no. of neurons (dynamic variables)

    if proj == None : proj = np.random.randn(n,2); # pick 2 random directions

    # restrict to interval
    if interval == None:
        Y = X
    else:
        tt = X.shape[0]
        t0 = max(1,round(tt*interval[0]))
        t1 = min(tt,round(tt*interval[1]))
        Y = X[t0:t1,:]

    # normalize projection vectors and compute projection
    proj1 = proj[:,0]/np.linalg.norm(proj[:,0])
    proj2 = proj[:,1]/np.linalg.norm(proj[:,1])
    Xproj = Y@np.array([proj1,proj2]).reshape(-1,2)
    print(Xproj)
    # plot projection
    plt.plot(Xproj[:,0], Xproj[:,1], c=color)
    plt.show()