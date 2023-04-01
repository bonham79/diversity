import os
import re
import random
import logging

import numpy as np

from graph import Graph
from misp import misp_factory
from solver import binary_max_min_dispersion, max_min_dispersion_max_min_sum, get_dispersion_min, get_dispersion_sum

from typing import List

logging.basicConfig(level = logging.INFO)

MISP_PATH="benchmarks/max_independent"
MMDP_PATH="benchmarks/max_min_diversity"

def test_misp(method: 'str'):
    files = os.listdir(MISP_PATH)
    for file in files:
        with open(os.path.join(MISP_PATH, file)) as source:
            lines = source.readlines()
            header = lines[0].split()
            v, e = int(header[-2]), int(header[-1])
            d_matrix = np.ones((v, v))
            for line in lines[1:]:
                line = line.split()[1:]
                i, j = int(line[0]) - 1, int(line[1]) - 1
                d_matrix[i][j] = 0
                d_matrix[j][i] = 0
            assert (d_matrix == 0).sum() == 2*e # checking sum
            for i in range(v): # we need to preserve index being attached to self
                d_matrix[i][i] = 0 

            logging.info(f"Finding maximum independent subset for: {file}. Method: {method}")
            graph = Graph(d_matrix < 1)
            subset = misp_factory[method](graph)
            logging.info(f"Found maximum independent subset of size: {len(subset)}. Members: {subset}")


def test_mmdp(method: str):
    files = os.listdir(MMDP_PATH)
    for file in files:
        p = int(file.split("_m")[-1].split(".")[0])
        with open(os.path.join(MMDP_PATH, file)) as source:
            lines = source.readlines()
            header = lines[0].split()
            m, n = int(header[0]), int(header[1])
            d_matrix = np.zeros((m, m))
            for line in lines[1:]:
                line = line.split()
                i, j, d = int(line[0]), int(line[1]), float(line[2])
                if i == j:
                    logging.warning(f"Dataset has non-trivial distance from node {i} to itself.") 
                d_matrix[i][j] = d
                d_matrix[j][i] = d
            for i in range(m):
                d_matrix[i][i] = 0 
            logging.info(f"Finding maximum-minimum dispersed subset for: {file}. Method: {method}")
            subset, L = binary_max_min_dispersion(d_matrix, p=p, method=method)
            assert p == len(subset) or len(subset) == 0

            logging.info(f"Calculating dispersion sum for subset of size {len(subset)} with bound {L} and members {subset}")
            logging.info(f"Min-dispersion: {get_dispersion_min(list(subset), d_matrix)[1]}")
            logging.info(f"Total: {get_dispersion_sum(list(subset), d_matrix)}")


def test_mmds(method: str):
    files = os.listdir(MMDP_PATH)
    for file in files:
        p = int(file.split("_m")[-1].split(".")[0])
        with open(os.path.join(MMDP_PATH, file)) as source:
            lines = source.readlines()
            header = lines[0].split()
            m, n = int(header[0]), int(header[1])
            d_matrix = np.zeros((m, m))
            for line in lines[1:]:
                line = line.split()
                i, j, d = int(line[0]), int(line[1]), float(line[2])
                if i == j:
                    logging.warning(f"Dataset has non-trivial distance from node {i} to itself.") 
                d_matrix[i][j] = d
                d_matrix[j][i] = d
            for i in range(m):
                d_matrix[i][i] = 0 
            logging.info(f"Finding max minimum sum subset for: {file}. Method: {method}")
            _, L = binary_max_min_dispersion(d_matrix, p=p, method=method)
            subset, D = max_min_dispersion_max_min_sum(d_matrix, p=p, L=L, method=method)
            assert p == len(subset) or len(subset) == 0

            logging.info(f"Calculating dispersion sum for subset of size {len(subset)} with bound {L} and members {subset}")
            logging.info(f"Min-dispersion: {D}")
            logging.info(f"Total: {get_dispersion_sum(list(subset), d_matrix)}")


if __name__=="__main__":
    test_mmdp("lin")
    test_mmds('lin')