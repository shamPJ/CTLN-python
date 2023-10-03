
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# auxiliary functions......................................................
def nonlin(x,W,b):
    """
    Linear comb + ReLu function.

    :param x : numpy array
    :out y   : numpy array, x after applying ReLU non-linearity
    """
    y = W@x.reshape(-1,1) + b
    y = np.maximum(0, x)
    return y.reshape(-1,)

def threshlin_ode(W, b=None, T=None, X0=None):
    """
    The threshlin ode to be solved: x-dot = -x + [Wx + b]_+

    :param W     : symmetric matrix of shape (n,n), matrix of recurrent connections, n = #neurons
    :param b     : array of shape(n,1), input vector; or an (n,m) matrix (m = # of different "switches" between b-vectors)
    :param T     : array of shape (1,m), amount of time in ode solution for each b-vector, should be a vector of length m
    :param X0    : array of shape (n,), vector of initial conditions in firing rates

    :param soln  : dict with keys 'X', 'Y' and 'W','b','T','X0' (same as input). 
                    X - array of shape (len(time), n), array of rate vectors x at each time
                    Y - array of shape (len(time), n), array of []_+ args Wx+b at each time
    """

    
    n = W.shape[0]  # find n = #neurons
    if b == None : b = np.ones((n,1)) # default is b = 1, uniform input
    m = b.shape[1] # number of b-vectors input
    if T == None : T = 10*np.ones((1,m))   # need to specify for each b-vector
    if X0 == None :X0 = np.zeros((n,))
  
    assert n ==  W.shape[1], "W must be a square matrix"
    assert n == b.shape[0], "b must have same dimension as sides of W"

    print("shapes of W, X0, b", W.shape, X0.shape, b.shape)

    # package up solution into a structure...............................
    soln = {}
    soln['W']  = W
    soln['b']  = b
    soln['T']  = T
    soln['X0'] = X0
    soln_X, soln_Y,  soln_time = [], [], []

    # solve threshlin ode for each b-vector, patch solutions...................
    t0 = 0
    for i in range(m):
        
        step = int(T[i][0]/.01)    # use time steps of ".01" in units of timescale
        t_eval = np.linspace(0,T[i],step).reshape(-1,)
        print("t_eval shape ", t_eval.shape)

        sol = solve_ivp(fun=lambda t, x: (-x + nonlin(x,W,b)), t_span=(0,T[i]), y0=X0, t_eval=t_eval)
    
        time, X = sol.t, sol.y    # time steps and values of the solution at t, shape (n, len(time))
        Y = W@X + b     # track arguments Wx+b inside []_+, shape (n, len(time))  

        print("time shape ", time.shape)
        print("X shape ", X.shape)
        print("Y shape ", Y.shape)

        soln_X.append(X)
        soln_Y.append(Y)
        soln_time.append(time)

        X0 = X[:,-1]    # reset initial condition for next b-vector
        t0 = time[-1]   # reset initial time for next b-vector
    
    soln['X'] = np.concatenate(soln_X, axis=1)
    soln['Y'] = np.concatenate(soln_Y, axis=1)
    soln['time'] = np.concatenate(soln_time)

    return soln

A = np.random.rand(2, 2)
W = np.tril(A) + np.tril(A, -1).T
print(-W)
soln = threshlin_ode(-W)

