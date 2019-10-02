import gudhi as gd
import operator
import numpy as np
from gerrychain import Partition
import networkx as nx

def adjacency_graph_cut_edges(part):
    """Returns the dual graph of a districting plan
    """
    edges = set([(part.assignment[x], part.assignment[y]) for x, y in part['cut_edges']])
    adjacency_graph = nx.Graph()
    adjacency_graph.add_nodes_from(list({x for (x,y) in edges}))
    adjacency_graph.add_edges_from(list(edges))
    return adjacency_graph

def relabel_by_dem_vote_share(part, election):
    """Renumbers districts by DEM vote share, 1-indexed
    """
    dem_percent = election.percents('Democratic')
    unranked_to_ranked = sorted([(list(part.parts.keys())[x], dem_percent[x])
                                  for x in range(0, len(part))],
                                  key=operator.itemgetter(1))
    unranked_to_ranked_list = [x[0] for x in unranked_to_ranked]
    unranked_to_ranked = {unranked_to_ranked[x][0]:x for x in range(0, len(part))}
    newpart = Partition(part.graph, {x:unranked_to_ranked[part.assignment[x]] for x in part.graph.nodes}, part.updaters)
    return newpart

def bottleneck_distance(partition0, partition1, election0, election1, down=True, shift=False):
    """Computes the TDA-inspired distance between two plans

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
    #get a shift value if necessary
    if down and shift:
        shift0 = 1-max(election0.percents('Democratic'))
        shift1 = 1-max(election1.percents('Democratic'))
    elif shift:
        shift0 = 0-min(election0.percents('Democratic'))
        shift1 = 0-min(election1.percents('Democratic'))
    if not shift:
        shift0=0
        shift1=0
    #get weights in order
    weights0 = np.zeros(len(part0))
    weights1 = np.zeros(len(part1))
    if down:
        for i in range(len(weights0)):
            weights0[i] = 1-sorted(election0.percents('Democratic'))[i]+shift0
        for i in range(len(weights1)):
            weights1[i] = 1-sorted(election1.percents('Democratic'))[i]+shift1
    else:
        for i in range(len(weights0)):
            weights0[i] = sorted(election0.percents('Democratic'))[i]+shift0
        for i in range(len(weights1)):
            weights1[i] = sorted(election1.percents('Democratic'))[i]+shift1

    return bottleneck_distance_from_graph(
        adjacency_graph0, adjacency_graph1,
        weights0, weights1
        )

def bottleneck_distance_from_graph(graph0, graph1, weights0, weights1):
    """
    Computes the bottleneck distance between graphs

    :param graph0, graph1: networkx Graph objects, nodes must be 0-indexed
    :param weights0, weights1: filtration values for nodes

    """
    adjacency_graph0 = graph0
    adjacency_graph1 = graph1
    #generate filtered complex for partition0
    spCpx0 = gd.SimplexTree()
    for node in adjacency_graph0.nodes:
        spCpx0.insert([node])
    for edge in adjacency_graph0.edges:
        spCpx0.insert(list(edge))
    zero_skeleton = spCpx0.get_skeleton(0)
    for j in range(len(zero_skeleton)):
        spCpx0.assign_filtration(
            zero_skeleton[j][0], filtration=weights0[j])
    spCpx0.make_filtration_non_decreasing()
    #generate filtered complex for partition1
    spCpx1 = gd.SimplexTree()
    for node in adjacency_graph1.nodes:
        spCpx1.insert([node])
    for edge in adjacency_graph1.edges:
        spCpx1.insert(list(edge))
    zero_skeleton = spCpx1.get_skeleton(0)
    for j in range(len(zero_skeleton)):
        spCpx1.assign_filtration(
            zero_skeleton[j][0], filtration=weights1[j])
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
