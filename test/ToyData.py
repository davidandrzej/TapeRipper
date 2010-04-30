"""
Generate synthetic data for testing audio segmentation algorithms
"""
import functools as FT
import operator as OP

import numpy as NP
import numpy.random as NR

#
# These functions take a List of (runtype,runlength) Tuples
#
# EX: [('gap',10),('track',50),('gap',20)] gives a gap of 
# length 10, a track of length 50, and a gap of length 20
# 

def generateData(runs):
    """ Return the raw NumPy array of all runs """
    return NP.concatenate([generateRun(run) for run in runs])

def generateRun(run):
    """ Generate the NumPy array for a single run """
    if(run[0] == 'gap'):
        return NR.normal(0,1,(run[1],))
    elif(run[0] == 'track'):
        return NR.normal(100,1,(run[1],))

def generateLabels(runs):
    """ 
    Get a NumPy array of ground-truth labels for the run 
    0 = gap
    1 = track
    """
    return NP.array(FT.reduce(OP.concat,
                              [[1 if run[0]=='track' else 0
                                for i in range(run[1])]
                               for run in runs]))                              
