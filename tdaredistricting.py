import gudhi as gd
import operator
import numpy as np
from gerrychain import Partition
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as LSA

def adjacency_graph_cut_edges(part):
    """
    Returns the dual graph of a districting plan
    """
    edges = set([(part.assignment[x], part.assignment[y]) for x, y in part['cut_edges']])
    adjacency_graph = nx.Graph()
    adjacency_graph.add_nodes_from(list(part.parts.keys()))
    adjacency_graph.add_edges_from(list(edges))
    return adjacency_graph

def relabel_by_dem_vote_share(part, election):
    """
    Renumbers districts by DEM vote share, 0-indexed
    """
    dem_percent = election.percents('Democratic')
    unranked_to_ranked = sorted([(list(part.parts.keys())[x], dem_percent[x])
                                  for x in range(0, len(part))],
                                  key=operator.itemgetter(1))
    unranked_to_ranked_list = [x[0] for x in unranked_to_ranked]
    unranked_to_ranked = {unranked_to_ranked[x][0]:x for x in range(0, len(part))}
    newpart = Partition(part.graph, {x:unranked_to_ranked[part.assignment[x]] for x in part.graph.nodes}, part.updaters)
    return newpart

def persistence_diagram(partition0, election0, down=True, shift=False):
    """
    Makes a persistence diagram for a part with an election
    """
    part0 = relabel_by_dem_vote_share(partition0, election0)
    adjacency_graph0 = adjacency_graph_cut_edges(part0)
    weights0 = sorted(election0.percents("Democratic"))
    #generate filtered complex for partition0
    return persistence_diagram_from_graph(adjacency_graph0, weights0, down=down, shift=shift)

def persistence_diagram_from_graph(graph0, weights0, down=True, shift=False):
    """
    Plots a persistence diagram for a 0-indexed graph with weights
    """
    adjacency_graph0 = graph0
    if min(list(graph0.nodes)) != 0:
        raise ValueError("Graph must be 0-indexed!")
    #get a shift value if necessary
    if down and shift:
        shift0 = 1-max(weights0)
    elif shift:
        shift0 = 0-min(weights0)
    if not shift:
        shift0=0
    #realign and shift
    new_weights0 = np.zeros(len(weights0))
    if down:
        for i in range(len(weights0)):
            new_weights0[i] = 1-weights0[i]+shift0
    else:
        for i in range(len(weights0)):
            new_weights0[i] = weights0[i]+shift0
    #generate filtered complex for partition0
    spCpx0 = gd.SimplexTree()
    for node in adjacency_graph0.nodes:
        spCpx0.insert([node])
    for edge in adjacency_graph0.edges:
        spCpx0.insert(list(edge))
    zero_skeleton = spCpx0.get_skeleton(0)
    for j in zero_skeleton:
        spCpx0.assign_filtration(
            j[0], filtration=new_weights0[j[0][0]])
    spCpx0.make_filtration_non_decreasing()
    #compute persistent homology
    barcodes0 = spCpx0.persistence()
    I0 = spCpx0.persistence_intervals_in_dimension(0)
    return I0

def bottleneck_distance(partition0, partition1, election0, election1, down=True, shift=False):
    """
    Computes the TDA-inspired distance between two plans

    :param partition0: The first partition
    :param partition1: The second partition
    :param election0: An election updater with a 'Democratic' alias
    :param election1: An election updater with a 'Democratic' alias
    :param down: Whether to filter downwards by DEM vote share
    :param shift: Shift the barcodes so that the first class appears at 0
    """
    #renumber district by democratic vote share ranked order
    part0 = relabel_by_dem_vote_share(partition0, election0)
    part1 = relabel_by_dem_vote_share(partition1, election1)
    #construct a dual graph for each partition
    adjacency_graph0 = adjacency_graph_cut_edges(part0)
    adjacency_graph1 = adjacency_graph_cut_edges(part1)

    return bottleneck_distance_from_graph(
        adjacency_graph0, adjacency_graph1,
        sorted(election0.percents("Democratic")),
        sorted(election1.percents("Democratic")),
        down=down,
        shift=shift
        )

def bottleneck_distance_from_graph(graph0, graph1, weights0, weights1, down=True, shift=False):
    """
    Computes the bottleneck distance between graphs with node weights

    :param graph0, graph1: networkx Graph objects, nodes must be 0-indexed
    :param weights0, weights1: filtration values for nodes
    :param down: Whether to filter downwards from 1 by weight
    :param shift: Shift the barcodes so that the first class appears at 0
    """
    adjacency_graph0 = graph0
    adjacency_graph1 = graph1
    if min(list(graph0.nodes)+list(graph1.nodes)) != 0:
        raise ValueError("Graph must be 0-indexed!")
    #get a shift value if necessary
    if down and shift:
        shift0 = 1-max(weights0)
        shift1 = 1-max(weights1)
    elif shift:
        shift0 = 0-min(weights0)
        shift1 = 0-min(weights1)
    if not shift:
        shift0=0
        shift1=0
    #get weights in order and shifted
    new_weights0 = np.zeros(len(weights0))
    new_weights1 = np.zeros(len(weights1))
    if down:
        for i in range(len(weights0)):
            new_weights0[i] = 1-weights0[i]+shift0
        for i in range(len(weights1)):
            new_weights1[i] = 1-weights1[i]+shift1
    else:
        for i in range(len(weights0)):
            new_weights0[i] = weights0[i]+shift0
        for i in range(len(weights1)):
            new_weights1[i] = weights1[i]+shift1
    #generate filtered complex for partition0
    spCpx0 = gd.SimplexTree()
    for node in adjacency_graph0.nodes:
        spCpx0.insert([node])
    for edge in adjacency_graph0.edges:
        spCpx0.insert(list(edge))
    zero_skeleton = spCpx0.get_skeleton(0)
    for j in zero_skeleton:
        spCpx0.assign_filtration(
            j[0], filtration=new_weights0[j[0][0]])
    spCpx0.make_filtration_non_decreasing()
    #generate filtered complex for partition1
    spCpx1 = gd.SimplexTree()
    for node in adjacency_graph1.nodes:
        spCpx1.insert([node])
    for edge in adjacency_graph1.edges:
        spCpx1.insert(list(edge))
    zero_skeleton = spCpx1.get_skeleton(0)
    for j in zero_skeleton:
        spCpx1.assign_filtration(
            j[0], filtration=new_weights1[j[0][0]])
    spCpx1.make_filtration_non_decreasing()
    #compute persistent homology
    barcodes0 = spCpx0.persistence()
    barcodes1 = spCpx1.persistence()
    #compute bottleneck distance
    spCpx0.persistence()
    spCpx1.persistence()
    I0 = spCpx0.persistence_intervals_in_dimension(0)
    I1 = spCpx1.persistence_intervals_in_dimension(0)

    return gd.bottleneck_distance(I0,I1)

def plot_districts_and_labels(part, gdf, labels, cmap="tab20c"):
    """
    Plots districts with labels on them

    :param part: a partition
    :param gdf: a geodataframe matching part
    :param labels: a dictionary matching districts to strings
    """
    gdf["assignment"] = [part.assignment[x] for x in part.graph.nodes]
    districts = gdf.dissolve(by="assignment")
    centroids = districts.geometry.representative_point()
    districts["centroid"] = centroids
    fig, ax = plt.subplots(figsize=(20,20))
    part.plot(gdf, cmap=cmap, ax=ax)
    districts.boundary.plot(ax=ax, edgecolor='black')
    for idx, row in districts.iterrows():
        ax.annotate(s=str(labels[row.name]), xy=row['centroid'].coords[0],
                 horizontalalignment='center')
    plt.show()
    del gdf["assignment"]

def pd_point_mean(V):
    """
    Computes the mean of a vector of points, some of which can be 'd' for diagonal.
    """
    num_diag = len([v for v in V if v == 'd'])
    if num_diag == len(V):
        return 'd' #only diagonal points

    nondiag_X = [v[0] for v in V if v != 'd']
    nondiag_Y = [v[1] for v in V if v != 'd']
    w = (np.mean(nondiag_X), np.mean(nondiag_Y))
    wdelta = ((w[1]+w[0])/2, (w[1]+w[0])/2) #closeset point on diagonal
    k = len(nondiag_X)
    m = len(V)

    return ((k*w[0]+(m-k)*wdelta[0])/m, (k*w[1]+(m-k)*wdelta[1])/m)

def dev_from_mean(PDs, Y):
    """
    Determines the total L2 Wasserstein distance from Y to the elements of PD
    """
    cost = sum([match_Hungarian_and_cost(pd, Y)[1] for pd in PDs])
    return cost

def match_Hungarian_and_cost(pd, Y):
    """
    Matches points in pd to points in Y or the diagonal in Y.

    Returns: list of lists containing the points matched to each y in Y.
    """
    longest_length = len(pd) + len(Y)
    M = np.zeros((longest_length, longest_length)) #cost matrix
    for i in range(longest_length): #pd
        for j in range(longest_length): #Y
            if i < len(pd) and j < len(Y):
                M[i,j] = (pd[i][0]-Y[j][0])**2+(pd[i][1]-Y[j][1])**2
            elif i < len(pd):
                M[i,j] = ((pd[i][0]-pd[i][1])**2)/2 #match to diagonal
            elif j < len(Y):
                M[i,j] = ((Y[j][0]-Y[j][1])**2)/2 #match to diagonal

    row_indices, col_indices = LSA(M)
    matched_to_Y = [None for y in Y]
    cost = 0


    for c, r in zip(col_indices, row_indices):
        cost += M[r,c]
        if c < len(Y): #not paired to diagonal
            if r < len(pd):
                matched_to_Y[c] = pd[r] #point
            else:
                matched_to_Y[c] = 'd' #diagonal point

    return matched_to_Y, cost

def match_Hungarian(pd, Y):
    """
    Matches points in pd to points in Y or the diagonal in Y.

    Returns: list of lists containing the points matched to each y in Y.
    """
    longest_length = len(pd) + len(Y)
    M = np.zeros((longest_length, longest_length)) #cost matrix
    for i in range(longest_length): #pd
        for j in range(longest_length): #Y
            if i < len(pd) and j < len(Y):
                M[i,j] = (pd[i][0]-Y[j][0])**2+(pd[i][1]-Y[j][1])**2
            elif i < len(pd):
                M[i,j] = ((pd[i][0]-pd[i][1])**2)/2 #match to diagonal
            elif j < len(Y):
                M[i,j] = ((Y[j][0]-Y[j][1])**2)/2 #match to diagonal

    row_indices, col_indices = LSA(M)
    matched_to_Y = [None for y in Y]
    for c, r in zip(col_indices, row_indices):
        if c < len(Y): #not paired to diagonal
            if r < len(pd):
                matched_to_Y[c] = pd[r] #point
            else:
                matched_to_Y[c] = 'd' #diagonal point
    return matched_to_Y

def Frechet_mean(PDs, seed=None):
    """
    Function for finding Frechet means ala Turner et al.

    PDs: list of persistence diagrams (each is a list of pairs)

    Convention: we only list the non-diagonal elements in the diagram.
    """
    if seed == None:
        Y_new = PDs[0].copy() #initialize
    else:
        Y_new = PDs[seed].copy()
    MAXITER = 100
    for iteration in range(MAXITER):
        Y_old = Y_new.copy()
        x_paired_to_y = [[] for y in Y_new]

        #pair up points in X_i to points in Y
        for i, pd in enumerate(PDs):
            #get point matched to each y from pd
            paired_to_y = match_Hungarian(pd, Y_new)
            for i, l in enumerate(paired_to_y):
                if l is not None:
                #add to list of all x matched to this y
                    x_paired_to_y[i].append(l)

        #calculate means and update Y
        for i, pd in enumerate(Y_new):
            if len(x_paired_to_y[i]) == 0: #no matches => drop
                Y_new[i] = 'd'
            else:
                Y_new[i] = pd_point_mean(x_paired_to_y[i]) #extended mean

        #remove diagonal points
        eps = 0
        for i in range(len(Y_new)):
            if Y_new[i] != 'd':
                eps += np.abs(Y_old[i][0]-Y_new[i][0])+np.abs(Y_old[i][1]-Y_new[i][1])
            else:
                eps += np.abs(Y_old[i][0]-Y_old[i][1]) #unmatch
        Y_new = [y for y in Y_new if y != 'd']
        if eps < 1e-3:
            return Y_new #converged
    return Y_new

def Frechet_mean_reseed(PDs):
    """
    Starts the greedy algorithm at every possible seed
    and returns the result with least distortion.
    """
    best = math.inf
    for j in range(len(PDs)):
        mean = Frechet_mean(PDs, j)
        dev = dev_from_mean(PDs, mean)
        if dev < best:
            print("({}, {:.2f})".format(j, dev), end=" ")
            bestmean = mean
            bestindex = j
            best = dev
    print("Best seed: {}".format(bestindex))
    return bestmean
