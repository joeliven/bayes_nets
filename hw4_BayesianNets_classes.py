import scipy as sc
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin
import scipy.stats as st
import math
import os
import sys


class BayesianNetwork:
    def __init__(self, name):
        self.name = name
        self.nodes = list()
        self.numNodes = 0
        self.numLinks = 0
                
    def addNode(self, label, dim, discrete_continuous, classes):
        index = len(self.nodes)
        node = Node(self, index, label, dim, discrete_continuous, classes)
        self.nodes.append(node)
        self.numNodes += 1
        
    def addLink(self, parentNodeIndex, childNodeIndex):
        (self.nodes[parentNodeIndex]).addChild(childNodeIndex)
        (self.nodes[childNodeIndex]).addParent(parentNodeIndex)
        self.numLinks += 1
        
    def prepCondProbTables(self):
        for i in xrange(0, self.numNodes):
            (self.nodes[i]).prepCondProbTable()
    
    def saveToFileCondProbTables(self, extension):
        cur_dir = os.getcwd()
        path_name =  str(self.name) + "_condProbTables"
        path = os.path.join(cur_dir, path_name)       
        if os.access(path, os.F_OK):
                os.chdir(path)
        else:
            os.mkdir(path)
            os.chdir(path)
            
        for i in xrange(0, self.numNodes):
            (self.nodes[i]).saveToFileCondProbTable(extension)
    
    def getCondProbTablesFromFiles(self, extension):
        cur_dir = os.getcwd()
        path_name =  str(self.name) + "_condProbTables"
        path = os.path.join(cur_dir, path_name)       
        if os.access(path, os.F_OK):
                os.chdir(path)
        else:
            print("Error! Subdirectory does not exist.")
            sys.exit()
            
        for i in xrange(0, self.numNodes):
            (self.nodes[i]).getCondProbTableFromFile(extension)
        
    def toString(self):
        string = "************************************************\n"
        string += str("BayesianNetwork.name:\t\t") + str(self.name) + str("\n")
        string += str("BayesianNetwork.numNodes:\t") + str(self.numNodes) + str("\n")
        string += str("BayesianNetwork.numLinks:\t") + str(self.numLinks) + str("\n")
        string += str("Nodes:\n---------------------------------------------\n")
        for i in xrange(0, self.numNodes):
            string += str((self.nodes[i]).toString())  + "\n"
        string += str("************************************************\n")
        return string        
            
    def sampleFromJoint(self):
        jointSample = list()
        for i in xrange(0, self.numNodes):
            distr = list()
            parents = (self.nodes[i]).parents
            parentsVals = (np.asarray(jointSample))[parents]
            for classNum in xrange(0, (self.nodes[i]).numClasses):
                distr.append((self.nodes[i]).lookUpCondProb((self.nodes[i]).classes[classNum], parents, parentsVals, all_parents_needed=True))
            uniform_sample = np.random.random()
            if (self.nodes[i]).numClasses == 2: # note, not dealing with case where numClasses > 2 at this point...
                if uniform_sample < distr[0]:
                    jointSample.append(0)
                else:
                    jointSample.append(1)
        return jointSample
    
    def sampleFromJoint_Rejection(self, evidenceNodes, evidenceNodesVals):
        jointSample = list()
        for i in xrange(0, self.numNodes):
            distr = list()
            parents = (self.nodes[i]).parents
            parentsVals = (np.asarray(jointSample))[parents]
            for classNum in xrange(0, (self.nodes[i]).numClasses):
                distr.append((self.nodes[i]).lookUpCondProb((self.nodes[i]).classes[classNum], parents, parentsVals, all_parents_needed=True))
            uniform_sample = np.random.random()
            if (self.nodes[i]).numClasses == 2: # note, not dealing with case where numClasses > 2 at this point...
                if uniform_sample < distr[0]:
                    sample = 0
                else:
                    sample = 1
            jointSample.append(sample)
            idx = np.argwhere(evidenceNodes == i)
            if len(idx) == 1:
                idx = idx[0,0]
                if evidenceNodesVals[idx] != sample:
#                     print("***rejecting this sample***")
                    return -1
#         print("jointSample: " + str(jointSample))
        return jointSample
    
    
    def sampleFromJoint_Likelihood(self, evidenceNodes, evidenceNodesVals):
        jointSample = list()
        weight = 1.0
        for i in xrange(0, self.numNodes):
            distr = list()
            parents = (self.nodes[i]).parents
            parentsVals = (np.asarray(jointSample))[parents]
            
            idx = np.argwhere(evidenceNodes == i)
            if len(idx) == 1:
                idx = idx[0,0]
                jointSample.append(evidenceNodesVals[idx])
                condProb_temp = (self.nodes[i]).lookUpCondProb((self.nodes[i]).classes[(evidenceNodesVals[idx])], parents, parentsVals, all_parents_needed=True)
                weight *= condProb_temp
                continue
            for classNum in xrange(0, (self.nodes[i]).numClasses):
                distr.append((self.nodes[i]).lookUpCondProb((self.nodes[i]).classes[classNum], parents, parentsVals, all_parents_needed=True))
            uniform_sample = np.random.random()
            if (self.nodes[i]).numClasses == 2: # note, not dealing with case where numClasses > 2 at this point...
                if uniform_sample < distr[0]:
                    sample = 0
                else:
                    sample = 1
            jointSample.append(sample)
        return jointSample, weight
  
      
    def rejectionSampling_fixed(self, max_its, numSamples, nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals):      
        denominator = 0.0
        numerator = 0.0
        overall_ct = 0
        sample_ct = 0
        while denominator < numSamples:
            sample = self.sampleFromJoint_Rejection(evidenceNodes, evidenceNodesVals)
            if sample != -1:
                sample_ct += 1
                denominator += 1
                sample = np.asarray(sample)
                comp_of_interest = sample[nodesOfInterest]
                if np.array_equal(comp_of_interest, nodesOfInterestVals):
                    numerator += 1
                if sample_ct % 100 == 0:
                    print("o_ct: " + str(overall_ct) + "  smpl_ct: "+ str(sample_ct) + "\trej_prob: " + str(numerator/denominator))
            overall_ct += 1
            if overall_ct > max_its:
                return (float(numerator)/float(denominator)), sample_ct, overall_ct
        return (float(numerator)/float(denominator)), sample_ct, overall_ct
    
    def rejectionSampling_convg(self, delta, interval, max_its, nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals):      
        denominator = 0.0
        numerator = 0.0
        ct = 0
        dif = 100.0
        prev_result = 0.0
        while dif > delta:
            sample = self.sampleFromJoint_Rejection(evidenceNodes, evidenceNodesVals)
            if sample != -1:
                denominator += 1
                sample = np.asarray(sample)
                comp_of_interest = sample[nodesOfInterest]
                if np.array_equal(comp_of_interest, nodesOfInterestVals):
                    numerator += 1
            if ct % interval == 0 and numerator != 0 and denominator != 0:
                dif = abs(prev_result - ((float(numerator)/float(denominator))))
                prev_result = ((float(numerator)/float(denominator)))
            if ct > max_its:
                return (float(numerator)/float(denominator)), ct
            ct += 1
        return (float(numerator)/float(denominator)), ct
                    
    def likelihoodSampling_convg(self, delta, interval, max_its, nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals):      
        denominator = 0.0
        numerator = 0.0
        ct = 0
        dif = 100.0
        prev_result = 0.0
        while dif > delta:
            sample, weight = self.sampleFromJoint_Likelihood(evidenceNodes, evidenceNodesVals)
            denominator += 1
            sample = np.asarray(sample)
            comp_of_interest = sample[nodesOfInterest]
            if np.array_equal(comp_of_interest, nodesOfInterestVals):
                numerator += (1.0 * weight)
            if ct % interval == 0 and numerator != 0 and denominator != 0:
                dif = abs(prev_result - ((float(numerator)/float(denominator))))
                prev_result = ((float(numerator)/float(denominator)))
            if ct > max_its:
                return (float(numerator)/float(denominator)), ct
            ct += 1
        return (float(numerator)/float(denominator)), ct
    
    def likelihoodSampling_noW(self, delta, interval, max_its, nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals):      
        denominator = 0.0
        numerator = 0.0
        ct = 0
        dif = 100.0
        prev_result = 0.0
        while dif > delta:
            sample, weight = self.sampleFromJoint_Likelihood(evidenceNodes, evidenceNodesVals)
            denominator += 1
            sample = np.asarray(sample)
            comp_of_interest = sample[nodesOfInterest]
            if np.array_equal(comp_of_interest, nodesOfInterestVals):
                numerator += 1.0
            if ct % interval == 0 and numerator != 0 and denominator != 0:
                dif = abs(prev_result - ((float(numerator)/float(denominator))))
                prev_result = ((float(numerator)/float(denominator)))
            if ct > max_its:
                return (float(numerator)/float(denominator)), ct
            ct += 1
        return (float(numerator)/float(denominator)), ct
        
        
    def gibbsSampling(self, numSamples, burn_in, keep_every, delta, interval, max_its, nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals):      
        denominator = 0.0
        numerator = 0.0
        overall_ct = 0
        sample_ct = 0
        dif = 100.0
        prev_result = 0.0
        # initialize the first sample randomly, except with evidenceNodes fixed:
        sample = np.zeros(self.numNodes, dtype=np.int)
        nonEvidenceNodes = list()
        for i in xrange(0, self.numNodes):
            if np.random.random() > .5:
                sample[i] = (1)
            if not np.in1d([i], evidenceNodes):
                nonEvidenceNodes.append(i)
        nonEvidenceNodes = np.asarray(nonEvidenceNodes)
        for i in xrange(0, len(evidenceNodes)):
            sample[evidenceNodes[i]] = evidenceNodesVals[i]

        # burn in phase:
        for i in xrange(0, burn_in):
            for i in xrange(0, len(nonEvidenceNodes)):
#                 updateNode = nonEvidenceNodes[np.random.randint(0, len(nonEvidenceNodes))]
                updateNode = nonEvidenceNodes[i]
                updateNode_distr_givenMB = self.bruteForceUpdate(sample, updateNode)
                if np.random.random() < updateNode_distr_givenMB:
                    sample[updateNode] = 1
                else:
                    sample[updateNode] = 0
        # burn in phase over...inference phase beginning:
        while(sample_ct < numSamples):
            updateNode = nonEvidenceNodes[np.random.randint(0, len(nonEvidenceNodes))]
            updateNode_distr_givenMB = self.bruteForceUpdate(sample, updateNode)
            if np.random.random() < updateNode_distr_givenMB:
                sample[updateNode] = 1
            else:
                sample[updateNode] = 0
            if overall_ct % keep_every == 0:
                denominator += 1
                comp_of_interest = sample[nodesOfInterest]
                if np.array_equal(comp_of_interest, nodesOfInterestVals):
                    numerator += 1.0
                if sample_ct % 100 == 0:
                    print("o_ct: " + str(overall_ct) + "  smpl_ct: "+ str(sample_ct) + "\tgibbs_prob: " + str(float(numerator)/float(denominator)))
                sample_ct += 1
            overall_ct += 1
        return (numerator/denominator), sample_ct, overall_ct

    
    def computeFullJoint(self, allNodeVals):
        fullJointProb = 1.0
        for i in xrange(0, self.numNodes):
            parents = (self.nodes[i]).parents
            parentsVals = (np.asarray(allNodeVals))[parents]
            fullJointProb *= (self.nodes[i]).lookUpCondProb((self.nodes[i]).classes[allNodeVals[i]], parents, parentsVals, all_parents_needed=True)
        return fullJointProb
    
    def bruteForceUpdate(self, allNodeVals, updateNode):
        allNodeVals[updateNode] = 1
        probUpdateNodeTrue = 1.0
        for i in xrange(0, self.numNodes):
            parents = (self.nodes[i]).parents
            parentsVals = (np.asarray(allNodeVals))[parents]
            probUpdateNodeTrue *= (self.nodes[i]).lookUpCondProb((self.nodes[i]).classes[allNodeVals[i]], parents, parentsVals, all_parents_needed=True)
        allNodeVals[updateNode] = 0
        probUpdateNodeFalse = 1.0
        for i in xrange(0, self.numNodes):
            parents = (self.nodes[i]).parents
            parentsVals = (np.asarray(allNodeVals))[parents]
            probUpdateNodeFalse *= (self.nodes[i]).lookUpCondProb((self.nodes[i]).classes[allNodeVals[i]], parents, parentsVals, all_parents_needed=True)
        return probUpdateNodeTrue/(probUpdateNodeTrue + probUpdateNodeFalse)
           
    def getTrueProb(self, queryNum):
        trueProbs = [0.299905, 0.39523, 0.049788, 0.129603, 0.70044]
        return trueProbs[queryNum]




class Node:
    def __init__(self, BN, index, label, dim, discrete_continuous, classes):
        self.BN = BN
        self.index = index
        self.label = label
        self.dim = dim
        self.discrete_continuous = discrete_continuous
        self.classes = classes
        self.numClasses = len(classes)
        self.parents = []
        self.children = []
        self.numParents = 0
        self.numChildren = 0      
    
    def addParent(self, parentNodeIndex):
        self.parents.append(parentNodeIndex)
        self.numParents += 1
    
    def addChild(self, childNodeIndex):
        self.children.append(childNodeIndex)
        self.numChildren += 1
    
    def lookUpCondProb(self, self_val, evidenceNodes, evidenceNodesVals, all_parents_needed=False):
        condProb = 0.0
        if self.numParents == 0:
            condProb = self.condProbTable[1, self_val]
            return condProb
        evidenceNodes = np.asarray(evidenceNodes)
        evidenceNodesVals = np.asarray(evidenceNodesVals)
        sort_order = np.argsort(evidenceNodes)
        evidenceNodes = evidenceNodes[sort_order]
        evidenceNodesVals = evidenceNodesVals[sort_order]
        if(all_parents_needed):
            if not np.array_equal(self.condProbTable[0, 0:self.numParents], evidenceNodes):
                print("Error! Not all parents are specified")
                print(self.condProbTable[0, 0:self.numParents])
                print(evidenceNodes)
                sys.exit()
            else:
                found = False
                for row in xrange(1, len(self.condProbTable[:,0])):
                    if(np.array_equal(self.condProbTable[row, 0:self.numParents], evidenceNodesVals)):
                        condProb = self.condProbTable[row, self.numParents + self_val]
                        found = True
                if not found: condProb = -1.0
        else:
            for row in xrange(1, len(self.condProbTable[:,0])):
                correct_row = True
                for col in xrange(0, len(self.condProbTable[0,0:self.numParents-1])):
                    if(self.condProbTable[0, col] not in evidenceNodes):
                        continue
                    else:
                        idx = np.where(evidenceNodes == self.condProbTable[0, col])
                        if(self.condProbTable[row, col]) != evidenceNodesVals[idx]:
                            correct_row = False
                if correct_row:
                    condProb += self.condProbTable[row, self.numParents + self_val]
        return condProb
    
    
    def toString(self):
        string = str("\t") + str(self.label) + str(".index:\t\t") + str(self.index) + str("\n")
        string += str("\t") + str(self.label) + str(".label:\t\t") + str(self.label) + str("\n")
        string += str("\t") + str(self.label) + str(".dim:\t\t") + str(self.dim) + str("\n")
        string += str("\t") + str(self.label) + str(".disc_cont:\t") + str(self.discrete_continuous) + str("\n")
        string += str("\t") + str(self.label) + str(".numClasses:\t") + str(self.numClasses) + str("\n")
        string += str("\t") + str(self.label) + str(".classes:\t\t") + str(self.classes) + str("\n")
        string += str("\t") + str(self.label) + str(".numParents:\t") + str(self.numParents) + str("\n")
        string += str("\t") + str(self.label) + str(".parents:\t\t") + str(self.parents) + str("\n")
        string += str("\t") + str(self.label) + str(".numChildren:\t") + str(self.numChildren) + str("\n")
        string += str("\t") + str(self.label) + str(".children:\t\t") + str(self.children) + str("\n")
        string += str("\t") + str(self.label) + str(".condProbTable dm:\t") + str(np.shape(self.condProbTable)) + str("\n")
        string += str("\t") + str(self.label) + str(".condProbTable:\n")
        string += str(self.condProbTable)
        return string
    
    def prepCondProbTable(self):
        numCols = int(self.numParents + self.numClasses)
        numRows = int((math.pow(self.numClasses, self.numParents)) + 1)
        
        self.condProbTable = np.zeros((numRows, numCols), dtype=np.float)
        for col in xrange(0, numCols):
            if col < (numCols - self.numClasses):
                classValIndex = -1
                for row in xrange(0, numRows):
                    if row == 0:
                        self.condProbTable[row, col] = self.parents[col]
                    else:
                        if((row-1) % math.pow(self.numClasses, col)) == 0:
                            classValIndex += 1
                        self.condProbTable[row, col] = self.classes[(classValIndex % self.numClasses)]
            else:
                for row in xrange(0, numRows):
                    if row == 0:
                        self.condProbTable[row, col] = self.index
                    else:
                        self.condProbTable[row, col] = -1

                    
    def saveToFileCondProbTable(self, extension):       
        fname = str(self.BN.name) + "_" + str(self.label) + str(extension)
        with open(fname, 'w') as cpt_table:
            for row in xrange(0, len(self.condProbTable[:,0])):
                for col in xrange(0, len(self.condProbTable[0,:])):
                    cpt_table.write(str(self.condProbTable[row, col]))
                    if (col < len(self.condProbTable[0,:]) - 1):
                        cpt_table.write(",")
                    else:
                        cpt_table.write("\n")
                        
    def getCondProbTableFromFile(self, extension):
        numCols = int(self.numParents + self.numClasses)
        numRows = int((math.pow(self.numClasses, self.numParents)) + 1)
        
        self.condProbTable = np.zeros((numRows, numCols), dtype=np.float)

        fname = str(self.BN.name) + "_" + str(self.label) + str(extension)
        with open(fname, 'r') as cpt_table:
            for row in xrange(0, len(self.condProbTable[:,0])):
                line = cpt_table.readline()
                line = line.strip()
                cells = line.split(",")
                for col in xrange(0, len(self.condProbTable[0,:])):
                    self.condProbTable[row, col] = float(cells[col])
        
    def computeMarkovBlanket(self, prevSample): # note, this function is not working properly
        prevSample = np.asarray(prevSample)
        parentsVals = prevSample[self.parents]
        
        # calculate p(me=T|MB(me)) (me = True)
        probMeTrueGivenMB = 1.0
        probMeTrueGivenMyParents = self.lookUpCondProb(1, self.parents, parentsVals, True)
        probMeTrueGivenMB *= probMeTrueGivenMyParents
        for i in xrange(0, self.numChildren):
            myChildsParents = np.asarray((self.BN.nodes[(self.children[i])]).parents)
            myChildsParentsVals = prevSample[myChildsParents]
            my_idx_in_my_childs_parents = np.where(myChildsParents == self.index)
            tmp = myChildsParentsVals[my_idx_in_my_childs_parents]
            myChildsParentsVals[my_idx_in_my_childs_parents] = 1
            probMeTrueGivenMB *= (self.BN.nodes[(self.children[i])]).lookUpCondProb(1, myChildsParents, myChildsParentsVals, True)
    
        # calculate p(me=F|MB(me)) (me = False)
        probMeFalseGivenMB = 1.0
        probMeFalseGivenMyParents = self.lookUpCondProb(0, self.parents, parentsVals, True)
        probMeFalseGivenMB *= probMeFalseGivenMyParents
        for i in xrange(0, self.numChildren):
            myChildsParents = np.asarray((self.BN.nodes[(self.children[i])]).parents)
            myChildsParentsVals = prevSample[myChildsParents]
            my_idx_in_my_childs_parents = np.where(myChildsParents == self.index)
            probMeFalseGivenMB *= (self.BN.nodes[(self.children[i])]).lookUpCondProb(0, myChildsParents, myChildsParentsVals, True)
        
        probMeTrueGivenMB_normalized = probMeTrueGivenMB / (probMeTrueGivenMB + probMeFalseGivenMB)
        return probMeTrueGivenMB_normalized
    
                     
                        