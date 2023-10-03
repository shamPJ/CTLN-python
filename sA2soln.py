def sA2soln(sA,T,X0,epsilon,delta,theta):
    """
    

    sA = n x n binary adjacency matrix for a directed graph
    T = length of time for ode solution, in units of leak timescale tau
    -> default is T=100
    X0 = n x 1 vector of initial conditions (default is random near all-zeros)
    epsilon/delta are the values for weights -1+epsilon, -1-delta in W
    -> defaults are those from graph2net.m
    theta = n x 1 vector of external stimulus or scalar value

    calls: graph2net.m and threshlin_ode.m
    
    :param sA      : matrix of shape (n,n), binary adjacency matrix for a directed graph
    :param T       : array of shape (1,m), amount of time in ode solution for each b-vector, should be a vector of length m
    :param X0      : array of shape (n,), vector of initial conditions in firing rates
    :param epsilon : 
    :param delta   :
    :param theta   :

    :out soln      :
    """



n = size(sA,1);  no. of neurons

if nargin < 2 || isempty(T)
    T = 100;
end;

if nargin < 3 || isempty(X0)
    X0 = zeros(n,1) + .01*rand(n,1);  break symmetry on init conds
end;

if nargin < 4 || isempty(epsilon)
    epsilon = .25;
end;

if nargin < 5 || isempty(delta)
    delta = 2*epsilon; 
end;

if nargin < 6 || isempty(theta)
    theta = 1;
end;

 create network W from sA
W = graph2net(sA,epsilon,delta);

 create external input vector b from theta
if length(theta) == 1
    b = theta*ones(n,1);
else
    b = theta;
end;

 simulate activity with constant b and initial conds X0
soln = threshlin_ode(W,b,T,X0);

 add adjacency matrix to soln struct
soln.sA = sA;
soln.eps = epsilon;
soln.delta = delta;