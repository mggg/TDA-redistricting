import gudhi as gd
import operator
import numpy as np
from gerrychain import Partition
import networkx as nx
import matplotlib.pyplot as plt

def adjacency_graph_cut_edges(part):
    """Returns the dual graph of a districting plan
    """
    edges = set([(part.assignment[x], part.assignment[y]) for x, y in part['cut_edges']])
    adjacency_graph = nx.Graph()
    adjacency_graph.add_nodes_from(list(part.parts.keys()))
    adjacency_graph.add_edges_from(list(edges))
    return adjacency_graph

def relabel_by_dem_vote_share(part, election):
    """Renumbers districts by DEM vote share, 0-indexed
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
    """Makes a persistence diagram for a part with an election
    """
    part0 = relabel_by_dem_vote_share(partition0, election0)
    adjacency_graph0 = adjacency_graph_cut_edges(part0)
    weights0 = sorted(election0.percents("Democratic"))
    #generate filtered complex for partition0
    return persistence_diagram_from_graph(adjacency_graph0, weights0, down=down, shift=shift)

def persistence_diagram_from_graph(graph0, weights0, down=True, shift=False):
    """Plots a persistence diagram for a 0-indexed graph with weights
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
        spCpx0.assign_filtration(
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
    """Plots districts with labels on them

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
