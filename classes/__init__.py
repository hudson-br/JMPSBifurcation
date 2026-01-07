print("Loading symbolic packages.")

import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import json
import time
from datetime import date
from IPython.display import clear_output

import os, sys

sys.path.append('../')

from IPython.display import display, Math

# from classes import *
from classes.symbolic_problem import SymbolicProblem
from classes.bifurcation_numeric import BifurcationNumeric
from classes.equilibrium import EquilibriumProblem
from classes.utils import *

savedir = "../data/"
if not os.path.exists(savedir):
    os.makedirs(savedir)