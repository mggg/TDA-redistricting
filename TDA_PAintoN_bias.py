'''
Generates a biased ensemble using PRES16 results.

Inputs: 
- outputfolder: an output folder to dump the results to (code adds the suffix "biased")
- partytofavor: either Democratic or Republican

Example usage:

python -u TDA_PAintoN_bias.py testrun DEM


'''

from gerrychain import Graph, Election, updaters, Partition, constraints, MarkovChain
from gerrychain.updaters import cut_edges
from gerrychain.proposals import recom
from gerrychain.tree import recursive_seed_part
from gerrychain.accept import always_accept
import numpy as np
import operator
from functools import partial
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import math, os
import pickle
import sys
import networkx as nx

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

outputfolder = sys.argv[1]+"biased"
partytofavor = sys.argv[2]
os.makedirs(outputfolder, exist_ok=True)
num_districts = 50
blocks_or_precincts = 'precincts'
INTERVAL = 100
steps = 101000 #recom steps

election_names = ["PRES16", "SEN16", "ATG12", "GOV14", "GOV10", "PRES12", "SEN10", "ATG16", "SEN12"]
election_columns = [
    ["T16PRESD", "T16PRESR"],
    ["T16SEND", "T16SENR"],
    ["ATG12D", "ATG12R"],
    ["F2014GOVD", "F2014GOVR"],
    ["GOV10D", "GOV10R"],
    ["PRES12D", "PRES12R"],
    ["SEN10D", "SEN10R"],
    ["T16ATGD", "T16ATGR"],
    ["USS12D", "USS12R"]
] #DEM, REP
# election_names = ["PRES16"]
# election_columns = [["T16PRESD", "T16PRESR"]]
pop_tol = 0.02
pop_col = "TOTPOP"

if blocks_or_precincts == "blocks":
    blocks_file_json = "/cluster/tufts/mggg/tweigh01/scale-rodden-cluster/PA_blocks_all_e.json"
elif blocks_or_precincts == "precincts":
    blocks_file_json = "./PA_VTD.json"
else:
    print("Please specify blocks or precincts in third argument.")

print("# districts for ReCom:", num_districts)

graph = Graph.from_json(blocks_file_json)

for n in graph.nodes:
    for i in range(len(election_columns)):
        for j in [0,1]:
            if math.isnan(graph.nodes[n][election_columns[i][j]]):
                graph.nodes[n][election_columns[i][j]] = 0
                print("Fixed NaN in ", election_columns[i][j])

total_population = sum([graph.nodes[n][pop_col] for n in graph.nodes()])

print("Shapefiles loaded and ready to run ReCom...")

for k in [num_districts]:
    parts = []
    pop_target = total_population/k
    myproposal = partial(recom, pop_col=pop_col, pop_target=pop_target, epsilon=pop_tol, node_repeats=2)

    #updaters
    myupdaters = {
        "population": updaters.Tally(pop_col, alias="population"),
        "cut_edges": cut_edges,
    }
    elections = [
        Election(
            election_names[i],
            {"Democratic": election_columns[i][0], "Republican": election_columns[i][1]},
        )
        for i in range(len(election_names))
    ]
    election_updaters = {election.name: election for election in elections}
    myupdaters.update(election_updaters)

    #initial partition
    ass = recursive_seed_part(graph, range(k), pop_target, pop_col, pop_tol)
    initial_partition = Partition(graph, ass, myupdaters)
    dev = max([np.abs(initial_partition["population"][d] - pop_target) for d in initial_partition.parts])
    print(" Using initial", k, "districts with population deviation = ", 100*dev/pop_target, "% of ideal.")

    #chain
    myconstraints = [
        constraints.within_percent_of_ideal_population(initial_partition, pop_tol)
    ]
    def myaccept(part):
        parent = part.parent
        DEMseats = len(
            [x for x in part["PRES16"].percents(partytofavor) if x > 0.53]
        )
        DEMseatsparent = len(
            [x for x in parent["PRES16"].percents(partytofavor) if x > 0.53]
        )
        alpha = np.exp(2*(DEMseats-DEMseatsparent))
        doaccept = (np.random.random() < alpha)
        return doaccept
    chain = MarkovChain(
        proposal=myproposal,
        constraints=myconstraints,
        accept=myaccept,
        initial_state=initial_partition,
        total_steps=steps
    )

    #run ReCom
    graphs = {e: [] for e in election_names}
    for index, step in enumerate(chain):
        if index%INTERVAL == 0 and index >= steps/101:
            print(index)
            for e in ["PRES16"]:
                newp = relabel_by_dem_vote_share(step, step[e])
                graphs[e].append((adjacency_graph_cut_edges(newp),sorted(step[e].percents("Democratic"))))
            #dump all plans
            parts.append(step.assignment)
        #dump ten times during run
        if index%int(steps/10) == 0:
            pickle.dump(parts, open(outputfolder+"/parts"+str(num_districts)+"_"+partytofavor+".p", "wb"))
            pickle.dump(graphs, open(outputfolder+"/graphs"+str(num_districts)+"_"+partytofavor+".p", "wb"))

print("Done with ReCom!")
pickle.dump(parts, open(outputfolder+"/parts"+str(num_districts)+"_"+partytofavor+".p", "wb"))
pickle.dump(graphs, open(outputfolder+"/graphs"+str(num_districts)+"_"+partytofavor+".p", "wb"))
