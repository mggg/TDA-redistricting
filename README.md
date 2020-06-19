# Persistence of Gerrymandering example code

This is some of the code used for the paper "The (Homological) Persistence of Gerrymandering"
by Duchin, Needham and Weighill. 

- `tdaredistricting.py`: the main functions needed to convert districting plans into persistence diagrams.
- `generate_ensemble.py`: an example script for generating an ensemble of Congressional plans for Pennsylvania.
- `make_graphics.ipynb`: a jupyter notebook for generating some plots from the output of `generate_ensemble.py`.

All of the code requires the open-source ensemble generation code [GerryChain](github.com/mggg/GerryChain) and the [gudhi](https://gudhi.inria.fr/) library. The PA shapefile is from the [mggg-states](github.com/mggg-states) github.
