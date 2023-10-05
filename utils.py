import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_graph(sA):
    """
    This function plots all the nodes of the graph around a circle then puts
    a directed arrow from node j to node i if sA(i,j)=1

    :param sA     : matrix of shape (n,n), binary adjacency matrix for a directed graph; if i->j, then sA(j,i) = 1.
    """

    n = sA.shape[0] # n.o. nodes

    # determine positions for nodes
    r = 1
    idxs = np.arange(n)
    x = r*np.cos(-idxs*2*np.pi/n + np.pi/2)
    y = r*np.sin(-idxs*2*np.pi/n + np.pi/2)
    pos = {}
    for i in range(n):
        pos[i] = (x[i], y[i])

    # create graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # add edges
    for j in range(n):
        for i in range(n):
            if sA[i,j] == 1:
                G.add_edge(j, i)
    return G, pos

def plot_projection(X, proj=None, interval=None):

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
    return Xproj

def plot_soln(sA, soln, proj=None, colors=[]):
    """
    :param sA     : matrix of shape (n,n), binary adjacency matrix for a directed graph; if i->j, then sA(j,i) = 1.
    :param colors : list, colors of the graph nodes
    """

    X = soln['X']
    time = soln['time']
    n = X.shape[1]
    if len(colors) == 0 : colors = list(mcolors.TABLEAU_COLORS.values())

    G, pos = plot_graph(sA)
    Xproj = plot_projection(X)

    #  create plt object
    #f, axes = plt.subplots(3, 1, figsize=(8,10), gridspec_kw={'height_ratios': [1, 1, 3]})
    fig=plt.figure(figsize=(8,6))

    gs=GridSpec(2,2) # 2 rows, 2 columns
    ax1 = fig.add_subplot(gs[0,0]) 
    ax2 = fig.add_subplot(gs[0,1]) 
    ax3 = fig.add_subplot(gs[1,:]) 

    # plot projection
    nx.draw(G, pos, node_size=600, node_color=colors[:n], arrowsize=20, width=2, with_labels=True, ax=ax1)
    ax2.plot(Xproj[:,0], Xproj[:,1], c='k')
    for i in range(n):
        ax3.plot(time, X[:,i])
    
    plt.savefig('plot.png')
    plt.show()

