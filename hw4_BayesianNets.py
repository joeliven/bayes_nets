import scipy as sc
import scipy.stats as st
import scipy.io as scio
import sys
import numpy as np
import matplotlib.pyplot as plt
import hw4_BayesianNets_functions as fn
import hw4_BayesianNets_classes as cl
import scipy.linalg as lin
import os
import math
import time as tm
# CONSTANTS:
extension = ".csv"
test = False
generate_raw_condProbTables = False

burn_in = 1000
keep_every = 1
delta = .0001
interval = 100
max_its = 30000
fixed_num_samples = 1000

# EXPERIMENTAL BN:
if(not test):
    # Experimental Constants:
    BN_name = "DadBN"
    labels = list()
    labels_raw = list(["Weekend", 
                       "EOQ", 
                       "CocktailHr", 
                       "BusinessTraveling", 
                       "Working", 
                       "AtMntHouse", 
                       "WithWife", 
                       "WorkGoingWell", 
                       "GoodWorkoutToday", 
                       "Mood"])
    numNodes = len(labels_raw)
    numClasses = 2
    for i in xrange(0, numNodes):
        labels.append(str(i) + "_" + str(labels_raw[i]))
    dim = 1 # note, not currently handling the multi-diminsional variable/node case
    discrete_continuous = "discrete" # note, not currently handling the continuous variable case
    classes = list()
    for i in xrange(0, numClasses):
        classes.append(i)
        
    # Create Experimental BN (note, this is a hand-designed BN for a DAG that models various aspects of my Dad's life:
    BN = cl.BayesianNetwork(BN_name)
    for i in xrange(0, numNodes):
        BN.addNode(labels[i], dim, discrete_continuous, classes)
    # Add links to Experimental BN:
    BN.addLink(0, 3)
    BN.addLink(0, 4)
    BN.addLink(0, 5)
    BN.addLink(0, 8)
    BN.addLink(1, 3)
    BN.addLink(1, 7)
    BN.addLink(2, 4)
    BN.addLink(2, 6)
    BN.addLink(2, 9)
    BN.addLink(3, 4)
    BN.addLink(3, 5)
    BN.addLink(3, 8)
    BN.addLink(3, 7)
    BN.addLink(3, 6)
    BN.addLink(3, 9)
    BN.addLink(4, 5)
    BN.addLink(5, 8)
    BN.addLink(5, 7)
    BN.addLink(5, 6)
    BN.addLink(5, 9)
    BN.addLink(8, 9)
    BN.addLink(7, 9)
    BN.addLink(6, 8)
    BN.addLink(6, 9)

# FOR EITHER TEST OR EXPERIMENTAL...
raw_completed = ""
if (generate_raw_condProbTables):
    # Prep the conditional probability table, then save it so the probabilities can be filled in by hand in excel:
    BN.prepCondProbTables()
    BN.saveToFileCondProbTables(extension)
    raw_completed = "raw"
else:
    BN.getCondProbTablesFromFiles(extension)
    raw_completed = "completed"

BN_string = BN.toString()

print("*******************************************")

add_header = raw_input("add header? (y/n) " )
if add_header == 'y':
    add_header = True
else:
    add_header = False
for i in xrange(0, len(BN.nodes)):
    print("X" + str(BN.nodes[i].index) + ": " + str(BN.nodes[i].label))
print("\nEnter query in form of (or \"r\" to generate random query):")
query = raw_input("[X]i=T/F,...,[X]9=T/F|[X]i=T/F,...,[X]9=T/F\t(don't include \"X\"):\n")
if query == 'r':
    word = "Random"
    nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals = fn.generateRandomQuery(numNodes, numClasses)
else:
    word = "Your"
    nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals = fn.processQuery(query)

fixed_num_samples = [2000]
delta = [.0001]
interval = [100]
burn_in = [1000]
keep_every = [2]
max_its = 100000
numQueries = 5
expCt = 0
queries = ["9=T|2=F,3=F,5=F,6=F,7=F,8=F","6=T|0=F,1=F,2=F","3=F,6=F,7=F,8=T|0=F,1=F,2=F,4=T","0=F,4=T,7=F|2=F,3=T,6=F,8=F","0=T,8=T|5=T,6=T,4=F"]

par_dir = os.path.pardir      
if os.access(par_dir, os.F_OK):
    os.chdir(par_dir)
       
with open("hw4_results.csv", 'a') as results:
    # write column headers to file:
    header = "expCt,method,qNum,query,numIntsNds,numEvNds,result,err,rej_gibbs_dif,sampleCt,overallCt,time,fixed_num,delta,intvl,burn,kp,max_its\n"
    if add_header:
        results.write(header)

    for queryNum in xrange(0, numQueries):
        print("*******************************************************************")
#         nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals = fn.generateRandomQuery(numNodes, numClasses)
        nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals = fn.processQuery(queries[queryNum])
        numIntsNds = len(nodesOfInterest)
        numEvNds = len(evidenceNodes)
        nodesOfInterest = np.asarray(nodesOfInterest)
        nodesOfInterestVals = np.asarray(nodesOfInterestVals)
        evidenceNodes = np.asarray(evidenceNodes)
        evidenceNodesVals = np.asarray(evidenceNodesVals)
        trueProb = BN.getTrueProb(queryNum)
        string = fn.getQueryString(nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals)
        print("\n" + str(word) + " query is:\n" + str(string))
        
        ##################################################################################
        print("RESULTS FOR QUERY# " + str(queryNum))
        #---------------------------------------------------------------------------------
        # Rejection Sampling (fixed):
        expCt += 1
        start_time = tm.clock()
        rej_fixed_result, sample_ct, overall_ct = BN.rejectionSampling_fixed(max_its, fixed_num_samples[0], nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals)   
        end_time = tm.clock()
        tot_time = end_time - start_time
        err = abs(trueProb - rej_fixed_result)    
        print("Rejection (Fixed): " + str(round(rej_fixed_result,5)) + "\tsmpl_ct = " + str(sample_ct) + "\to_ct = " + str(overall_ct) + "\ttime = " + str(round(tot_time, 4)))        
        output = (str(expCt)+",reject_fixed,"+str(queryNum)+","+str(string)+","+str(numIntsNds)+","+str(numEvNds)+","+str(rej_fixed_result)+","+str(err)+","+""+","+str(sample_ct)+","+str(overall_ct)+","+str(tot_time)+","+str(fixed_num_samples[0])+","+"na"+","+"na"+","+"na"+","+"na"+","+str(max_its)+"\n")
        results.write(output)

        #---------------------------------------------------------------------------------
        # Gibbs Sampling:
        expCt += 1
        start_time = tm.clock()
        gibbs_result, sample_ct, overall_ct = BN.gibbsSampling(fixed_num_samples[0], burn_in[0], keep_every[0], delta[0], interval[0], max_its, nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals)  
        end_time = tm.clock()
        tot_time = end_time - start_time
        err = abs(trueProb - gibbs_result)
        rej_gibbs_dif = abs(rej_fixed_result - gibbs_result)
        print("Gibbs (Convg): " + str(round(gibbs_result,5)) + "\tsmpl_ct = " + str(sample_ct) + "\to_ct = " + str(overall_ct) + "\ttime = " + str(round(tot_time, 4)))
        output = (str(expCt)+",gibbs,"+str(queryNum)+","+str(string)+","+str(numIntsNds)+","+str(numEvNds)+","+str(gibbs_result)+","+str(err)+","+str(rej_gibbs_dif)+","+str(sample_ct)+","+str(overall_ct)+","+str(tot_time)+","+str(fixed_num_samples[0])+","+str(delta[0])+","+str(interval[0])+","+str(burn_in[0])+","+str(keep_every[0])+","+str(max_its)+"\n")
        results.write(output)








#        for fixed_num in xrange(0, len(fixed_num_samples)):
#              #---------------------------------------------------------------------------------
#              # Rejection Sampling (fixed):
#             print("fixed_num_samples: " + str(fixed_num_samples[fixed_num]))
#             expCt += 1
#             print("\nexpCt: " + str(expCt))
#             start_time = tm.clock()
#             prob_result, ct = BN.rejectionSampling_fixed(max_its, fixed_num_samples[fixed_num], nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals)   
#             end_time = tm.clock()
#             tot_time = end_time - start_time
#             err = abs(trueProb - prob_result)
# #             rej_gibbs_dif = abs(trueProb - prob_result)
#             
#             print("Rejection (Fixed): " + str(round(prob_result,5)) + "\tct = " + str(ct) + "\ttime = " + str(round(tot_time, 4)))
#             output = (str(expCt)+",reject_fixed,"+str(queryNum)+","+str(string)+","+str(numIntsNds)+","+str(numEvNds)+","+str(prob_result)+","+str(ct)+","+str(tot_time)+","+str(fixed_num_samples[fixed_num])+","+"na"+","+"na"+","+"na"+","+"na"+","+str(max_its)+"\n")
#             results.write(output)
#             
#         for dlt in xrange(0, len(delta)):
#             print("delta: " + str(delta[dlt]))
#             for intvl in xrange(0, len(interval)):
#                 print("interval: " + str(interval[intvl]))
#                 #---------------------------------------------------------------------------------
#                 # Rejection Sampling (convg):
#                 expCt += 1
#                 print("\nexpCt: " + str(expCt))
#                 start_time = tm.clock()
#                 prob_result, ct = BN.rejectionSampling_convg(delta[dlt], interval[intvl], max_its, nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals)   
#                 end_time = tm.clock()
#                 tot_time = end_time - start_time
#                 print("Rejection (Convg): " + str(round(prob_result,5)) + "\tct = " + str(ct) + "\ttime = " + str(round(tot_time, 4)))
#                 output = (str(expCt)+",reject_convg,"+str(queryNum)+","+str(string)+","+str(numIntsNds)+","+str(numEvNds)+","+str(prob_result)+","+str(ct)+","+str(tot_time)+","+str(fixed_num_samples[fixed_num])+","+str(delta[dlt])+","+str(interval[intvl])+","+"na"+","+"na"+","+str(max_its)+"\n")
#                 results.write(output)
#                 #---------------------------------------------------------------------------------
#                 # Likelihood Sampling (wW):
#                 expCt += 1
#                 print("\nexpCt: " + str(expCt))
#                 start_time = tm.clock()
#                 prob_result, ct = BN.likelihoodSampling_convg(delta[dlt], interval[intvl], max_its, nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals)   
#                 end_time = tm.clock()
#                 tot_time = end_time - start_time
#                 print("Likelihood (wght): " + str(round(prob_result,5)) + "\tct = " + str(ct) + "\ttime = " + str(round(tot_time, 4)))
#                 output = (str(expCt)+",likelihood_wW,"+str(queryNum)+","+str(string)+","+str(numIntsNds)+","+str(numEvNds)+","+str(prob_result)+","+str(ct)+","+str(tot_time)+","+str(fixed_num_samples[fixed_num])+","+str(delta[dlt])+","+str(interval[intvl])+","+"na"+","+"na"+","+str(max_its)+"\n")
#                 results.write(output)
#                 #---------------------------------------------------------------------------------
#                 # Likelihood Sampling (nW):
#                 expCt += 1
#                 print("\nexpCt: " + str(expCt))
#                 start_time = tm.clock()
#                 prob_result, ct = BN.likelihoodSampling_noW(delta[dlt], interval[intvl], max_its, nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals)   
#                 end_time = tm.clock()
#                 tot_time = end_time - start_time 
#                 print("Likelihood (no w): " + str(round(prob_result,5)) + "\tct = " + str(ct) + "\ttime = " + str(round(tot_time, 4)))
#                 output = (str(expCt)+",likelihood_nW,"+str(queryNum)+","+str(string)+","+str(numIntsNds)+","+str(numEvNds)+","+str(prob_result)+","+str(ct)+","+str(tot_time)+","+str(fixed_num_samples[fixed_num])+","+str(delta[dlt])+","+str(interval[intvl])+","+"na"+","+"na"+","+str(max_its)+"\n")
#                 results.write(output)
# 
#                 for burn in xrange(0, len(burn_in)):
#                     print("burn_in: " + str(burn_in[burn]))
#                     for kp in xrange(0, len(keep_every)):
#                         print("keep_every: " + str(keep_every[kp]))
#                         #---------------------------------------------------------------------------------
#                         # Gibbs Sampling:
#                         expCt += 1
#                         print("\nexpCt: " + str(expCt))
#                         start_time = tm.clock()
#                         prob_result, ct = BN.gibbsSampling_convg(burn_in[burn], keep_every[kp], delta[dlt], interval[intvl], max_its, nodesOfInterest, nodesOfInterestVals, evidenceNodes, evidenceNodesVals)  
#                         end_time = tm.clock()
#                         tot_time = end_time - start_time
#                         print("Gibbs (Convg): " + str(round(prob_result,5)) + "\t\tct = " + str(ct) + "\ttime = " + str(round(tot_time, 4)))
#                         output = (str(expCt)+",gibbs,"+str(queryNum)+","+str(string)+","+str(numIntsNds)+","+str(numEvNds)+","+str(prob_result)+","+str(ct)+","+str(tot_time)+","+"na"+","+str(delta[dlt])+","+str(interval[intvl])+","+str(burn_in[burn])+","+str(keep_every[kp])+","+str(max_its)+"\n")
#                         results.write(output)
