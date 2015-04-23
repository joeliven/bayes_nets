import scipy as sc
import scipy.io as scio
import scipy.io.wavfile as wav
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.linalg as lin
import scipy.stats as st
import math
import os

def processQuery(query):
    nodesOfInterest_temp = ((query.strip()).split("|"))[0]
    evidenceNodes_temp = ((query.strip()).split("|"))[1]
    nodesOfInterest_ = nodesOfInterest_temp.split(",")
    evidenceNodes_ = evidenceNodes_temp.split(",")
    nodesOfInterest = list()
    nodesOfInterestVals = list()
    evidenceNodes = list()
    evidenceNodesVals = list()
    for i in xrange(0, len(nodesOfInterest_)):
        tmp = nodesOfInterest_[i].split("=")
        nodesOfInterest.append(int(tmp[0]))
        if str(tmp[1]) == 'T':
            nodesOfInterestVals.append(1)
        else:
            nodesOfInterestVals.append(0)
    for i in xrange(0, len(evidenceNodes_)):
        tmp = evidenceNodes_[i].split("=")
        evidenceNodes.append(int(tmp[0]))
        if str(tmp[1]) == 'T':
            evidenceNodesVals.append(1)
        else:
            evidenceNodesVals.append(0)
            
    evidenceNodes = np.asarray(evidenceNodes)
    evidenceNodesVals = np.asarray(evidenceNodesVals)
    sort_order = np.argsort(evidenceNodes)
    evidenceNodes = evidenceNodes[sort_order]
    evidenceNodesVals = evidenceNodesVals[sort_order]
    
    nodesOfInterest = np.asarray(nodesOfInterest)
    nodesOfInterestVals = np.asarray(nodesOfInterestVals)
    sort_order = np.argsort(nodesOfInterest)
    nodesOfInterest = nodesOfInterest[sort_order]
    nodesOfInterestVals= nodesOfInterestVals[sort_order]
    
    return nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals

def generateRandomQuery(numNodes, numClasses):
    nodesOfInterest = list()
    nodesOfInterestVals = list()
    evidenceNodes = list()
    evidenceNodesVals = list()
    
    numNodesOfInterest = np.random.randint(1, numNodes - 1)
    numEvidenceNodes = np.random.randint(1, (numNodes - numNodesOfInterest))
    nodes = list()
    for i in xrange(0, numNodes):
        nodes.append(i)
    
    if( numClasses == 2): # note, note handling the case for numClasses > 2 at this point...
        for i in xrange(0, numNodesOfInterest):
            nodeNum = nodes.pop(np.random.randint(0, len(nodes) - 1))
            nodesOfInterest.append(nodeNum)
        for i in xrange(0, numEvidenceNodes):
            nodeNum = nodes.pop(np.random.randint(0, len(nodes) - 1))
            evidenceNodes.append(nodeNum) 
        
        nodesOfInterest.sort()
        evidenceNodes.sort()
        for i in xrange(0, numNodesOfInterest):
            if (np.random.random_sample() < 0.5):
                nodesOfInterestVals.append(0)
            else:
                nodesOfInterestVals.append(1)
        for i in xrange(0, numEvidenceNodes):
            if (np.random.random_sample() < 0.5):
                evidenceNodesVals.append(0)
            else:
                evidenceNodesVals.append(1)
    return nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals

def getQueryString(nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals):
    print(nodesOfInterestVals)
    string = "p("
    for i in xrange(0, len(nodesOfInterest)):
        string += "x" + str(nodesOfInterest[i]) + "="
        if nodesOfInterestVals[i] == 0:
            string += "F"
        else:
            string += "T"
        if i < len(nodesOfInterest) - 1:
            string += ";"
    string += "|"
    for i in xrange(0, len(evidenceNodes)):
        string += "x" + str(evidenceNodes[i]) + "="
        if evidenceNodesVals[i] == 0:
            string += "F"
        else:
            string += "T"
        if i < len(evidenceNodes) - 1:
            string += ";"
    string += ")"
    return string

def randomBinaryVals(num):
    vals = list()
    for i in xrange(0, num):
        if(np.random.random_sample() < .5):
            vals.append(0)
        else:
            vals.append(1)
    return vals



