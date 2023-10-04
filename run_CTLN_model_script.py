import numpy as np
from sA2soln import sA2soln
from utils import plot_ratecurves, plot_projection

# STEP 1. Input any n x n adjacency matrix, called sA.
# Note: if i->j, then sA(j,i) = 1.

# The sA matrix for the graph in Figure 1C https://arxiv.org/pdf/1804.01487.pdf
sA = np.array([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0]])
print(sA)

# % To create a cyclically symmetric graph on n nodes with each node projecting k edges: i->i+1 ... i->i+k, uncomment these commands: 
# % n = 5; 
# % k = 2; % need k=1 for simple cycle
# % sA = make_kcyclic_graph(n,k);

# % To generate a random oriented graph, uncomment these commands:
# % n = 10; % number of neurons
# % sA = randDigraph(n);

# % To load a pre-stored graph, uncomment the following:
# % load sA_n5_multi_chaos; 

# STEP 2. Simulate dynamics for the corresponding threshlin network model

# simulation parameters
n = sA.shape[0]                      # number of neurons
T = np.array([100]).reshape(1,-1)    # simulation time length, in units of membrane timescale
e = .25                              # epsilson value (default is .25)
d = 2*e                              # delta value (default is 2*e = .5)
theta = 1                            # theta value

# SET INITIAL CONDITIONS
X0 = np.array([0, 0, 0, .2]) # user-specified initial conditions
assert n == X0.shape[0], "X0 has a wrong shape != n"

# solve ODEs (solution is returned in "soln" struct)
soln = sA2soln(sA,T,X0,e,d,theta)

# OR 

# % X0 = .5*np.random.randn(n,) # random initial condition

# % fp = sA2fixpts(sA,e,d,theta); % the fixed points of the CTLN
# % X0 = fp(1,:)' + 0.1*rand(n,1); % set the initial condition to be a small random perturbation around one of the fixed points

# STEP 3. Plot the results!

# plot adjacency matrix and solution!

#plot_soln(soln,proj,colors)
X = soln['X']
time = soln['time']
plot_projection(X, proj=None, interval=None, color='k')
plot_ratecurves(X, time)