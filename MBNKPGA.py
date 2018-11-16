#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import itertools as itt
import random
from math import radians, cos, sin, asin, sqrt
import copy
# import matplotlib.pyplot as plt

def MBNKPGA(GAParam, Data):
    '''
    GA params settings:
        popSize: number of population in each iteration
        chromoSize: length of chromo
        generationSize:
        crossRate: rate of crossover
        mutateRate: rate of mutation
        elitism: whether elitism or not
    '''
    GAParam['elitism'] = True
    #GAParam['popSize'] = 20
    GAParam['popSize'] = GAParam['taskLen']*6
    GAParam['chromoSize'] = GAParam['taskLen']
    GAParam['generationSize'] = 100
    Var = Variables(GAParam)
    result = dict()
    result = GeneticAlgorithm(Var, GAParam,Data)
    return result

class Variables(object):
    def __init__(self, GAParam):
        self.fitness_value = np.zeros(GAParam['generationSize']) # different with MATLAB code
        self.fitness_avg = np.zeros(GAParam['generationSize'])
        self.fitness_table = np.zeros(GAParam['popSize'])
        self.best_fitness = 0.
        self.best_individual = [[] for _ in range(GAParam['chromoSize'])]
        self.best_generation = 0
        self.bestStartTime = np.zeros(GAParam['chromoSize'])
        self.bestDepartTime = np.zeros(GAParam['chromoSize'])
        self.best_BInd = np.zeros(GAParam['MB'],)
        self.best_WInd = [[] for _ in range(GAParam['chromoSize'])]
        self.best_BInventFinal = [[] for _ in range(GAParam['MB'])] # 0723
        self.best_BEndFinal = [[] for _ in range(GAParam['MB'])]# 0723   
        self.best_pickupTurbineOrd = [[] for _ in range(GAParam['chromoSize'])]
        self.best_pickupTime = [[] for _ in range(GAParam['chromoSize'])]
        self.best_pickupBoat = [[] for _ in range(GAParam['chromoSize'])]
        self.best_totalCostPL = 0.
        self.best_totalCostOT = 0.
        self.best_totalCostBoat = 0.
        self.best_individualCost = 0.
        self.best_individualCostPL = 0.
        self.best_individualCostOT = 0.
        self.best_individualCostBoat = 0.
        self.best_individualWW = np.zeros(GAParam['chromoSize'])
        
        # self.popu, self.popuWW = self.initialize(GAParam['popSize'], GAParam['chromoSize'])
        self.popu, self.popuWW = self.initialize(GAParam) # modified 1101
        self.totalCost = np.zeros(GAParam['popSize'])
        self.totalCostPL = np.zeros(GAParam['popSize'])
        self.totalCostOT = np.zeros(GAParam['popSize'])
        self.totalCostBoat = np.zeros(GAParam['popSize'])
        self.individualCost = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
        self.individualCostPL = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
        self.individualCostOT = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
        self.individualCostBoat = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
        self.startTime = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
        self.departTime = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
        self.BInd = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
        self.WInd = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
        self.BInventFinal = [[[] for _ in range(GAParam['MB'])] for _ in range(GAParam['popSize'])] #0723
        self.BEndFinal = [[[] for _ in range(GAParam['MB'])] for _ in range(GAParam['popSize'])] #0723
        self.pickupTurbineOrdAll = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
        self.pickupTimeAll = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
        self.pickupBoatAll = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
    
    def initialize(self, GAParam):
        popSize = GAParam['popSize']
        chromoSize = GAParam['chromoSize']
        popu = np.zeros((popSize,chromoSize), dtype = int)
        if GAParam['Priority_condition'] == True:
            task_set = np.setdiff1d(range(chromoSize),GAParam['Priority_list'])
            for i in np.arange(0,popSize).reshape(-1):
                popu[i,range(len(GAParam['Priority_list']),chromoSize)]=np.random.permutation(task_set)
                del i   
            popu[:,range(len(GAParam['Priority_list']))] = GAParam['Priority_list']
            
        else: 
            for i in np.arange(0,popSize).reshape(-1):
                popu[i,:]=np.random.permutation(range(chromoSize))
                del i
                
        popuWW=np.zeros((popSize,chromoSize))
        for i in np.arange(0,popSize).reshape(-1):
            popuWW[i,:]=np.random.choice(np.arange(0,4),chromoSize,replace=True)
            del i
        return popu, popuWW



def GeneticAlgorithm(Var, GAParam, Data):
    '''
    This function is to find out optimal sequence series based on GA
    '''
    #import numpy as np
    
# =============================================================================
#     global G  # of generations
#     global fitness_value
#     global best_fitness
#     global fitness_avg
#     global best_individual
#     global best_generation
#     global bestStartTime
#     global bestDepartTime
#     global best_BInd
#     global best_WInd
#     global best_BInventFinal # 0723
#     global best_BEndFinal # 0723   
#     global best_pickupTurbineOrd
#     global best_pickupTime
#     global best_pickupBoat
#     global best_totalCostPL
#     global best_totalCostOT
#     global best_totalCostBoat
#     global best_individualCost
#     global best_individualCostPL
#     global best_individualCostOT
#     global best_individualCostBoat
#     
#     fitness_avg = np.zeros(GAParam['generationSize'])
#     fitness_value = np.zeros(GAParam['generationSize']) # different with MATLAB code
#     best_fitness = 0.
#     best_generation = 0
# =============================================================================

    #initilize(GAParam['popSize'], GAParam['chromoSize']) # initialization
    allbestresult = np.zeros(shape=[GAParam['generationSize'],])
    for G in range(GAParam['generationSize']):
        Var = fitnessMBNKP(Var, GAParam,Data)
        Var = rank(Var, GAParam, G)  # ranking of all schedulings
        Var = selection(Var, GAParam) # samples selection
        Var = crossover(Var, GAParam) # crossover
        Var = mutation(Var, GAParam) # mutation
        
        allbestresult[G] = 1./Var.best_fitness
        '''
        if G%5 == 0:
            print (1./Var.best_fitness)
        plt.plot(np.arange(GAParam['generationSize'])+1, allbestresult)
        '''
    
    results = dict()
    results['bestSequence'] = Var.best_individual
    results['bestWW'] = Var.best_individualWW 
    results['objValue'] = 1./Var.best_fitness
    results['bestSTime'] = Var.bestStartTime
    results['bestDTime'] = Var.bestDepartTime
    results['bestGeneration'] = Var.best_generation
    results['bestBInd'] = Var.best_BInd
    results['bestWInd'] = Var.best_WInd
    results['bestBInvent'] = Var.best_BInventFinal # 0723
    results['bestBEnd'] = Var.best_BEndFinal # 0723
    results['bestPickupTurbineOrd'] = Var.best_pickupTurbineOrd
    results['bestPickupTime'] = Var.best_pickupTime
    results['bestPickupBoat'] = Var.best_pickupBoat
    results['bestTotalCostPL'] = Var.best_totalCostPL
    results['bestTotalCostOT'] = Var.best_totalCostOT
    results['bestTotalCostBoat'] = Var.best_totalCostBoat
    results['bestIndividualCost'] = Var.best_individualCost
    results['bestIndividualCostPL'] = Var.best_individualCostPL
    results['bestIndividualCostOT'] = Var.best_individualCostOT
    results['bestIndividualCostBoat'] = Var.best_individualCostBoat

    return results

# =============================================================================
# def initilize(popSize,chromoSize):
#     import numpy as np
#     
#     global popu
#     popu = np.zeros((popSize,chromoSize), dtype = int)
#     for i in np.arange(0,popSize).reshape(-1):
#         popu[i,:]=np.random.permutation(range(chromoSize))
#     del i
#     
#     global popuWW
#     popuWW=np.zeros((popSize,chromoSize))
#     for i in np.arange(0,popSize).reshape(-1):
#         popuWW[i,:]=np.random.choice(np.arange(0,4),chromoSize,replace=True)
#     del i
#     
#     return  
# =============================================================================
    

def fitnessMBNKP(Var, GAParam,Data):
# =============================================================================
#     import numpy as np
#     
#     global fitness_value
#     global popu
#     global popuWW
#     global totalCost
#     global totalCostPL
#     global totalCostOT
#     global totalCostBoat
#     global individualCost
#     global individualCostPL
#     global individualCostOT
#     global individualCostBoat
#     global startTime
#     global departTime
#     global BInd
#     global WInd
#     global BInventFinal # 0723
#     global BEndFinal # 0723
#     global pickupTurbineOrdAll
#     global pickupTimeAll
#     global pickupBoatAll
# 
# 
#     # parameters Type I initialization
#     startTime = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
#     departTime = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
#     #BInd = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
#     BInd = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
#     WInd = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
#     BEndFinal = [[[] for _ in range(GAParam['MB'])] for _ in range(GAParam['popSize'])] #0723
#     BInventFinal = [[[] for _ in range(GAParam['MB'])] for _ in range(GAParam['popSize'])] #0723
#     pickupTurbineOrdAll = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
#     pickupTimeAll = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
#     pickupBoatAll = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
#     
#     individualCost = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
#     individualCostPL = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
#     individualCostOT = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
#     individualCostBoat = np.zeros((GAParam['popSize'],GAParam['chromoSize']))
#     
#     totalCost = np.zeros(GAParam['popSize'])
#     totalCostPL = np.zeros(GAParam['popSize'])
#     totalCostOT = np.zeros(GAParam['popSize'])
#     totalCostBoat = np.zeros(GAParam['popSize'])
# =============================================================================

    for ps in range(GAParam['popSize']):
        
        # parameters type II initialization
        # t, tB, costIndividualTask, costIndividualPL, costIndividualOT, costIndividualBoat
        t = np.zeros(GAParam['chromoSize'], dtype = np.int)
        tB = np.zeros(GAParam['chromoSize'], dtype = np.int)
        #BIndindi = np.zeros(GAParam['chromoSize'], dtype = np.int)
        BIndindi = [[] for _ in range(GAParam['chromoSize'])]
        costIndividualTask = np.zeros(GAParam['chromoSize'])
        costIndividualPL = np.zeros(GAParam['chromoSize'])
        costIndividualOT = np.zeros(GAParam['chromoSize'])
        costIndividualBoat = np.zeros(GAParam['chromoSize'])
        
        # timeIni = 36
        timeIni = GAParam['timeIni']
        
        # workers with PTypes different skills (such as: mechanical & electrical)
        PTypes = GAParam['PType']
        
        # workers requirements for different tasks
        # TaskWorkers should be changed based on the order of popu(ps,j)
        TaskWorkers = GAParam['TaskWorkers'][:, Var.popu[ps]]
        pickupTurbineOrd = [[] for _ in range(GAParam['chromoSize'])] 
        pickupTime = [[] for _ in range(GAParam['chromoSize'])]
        
        
        # Task initialization
        # Worker assignment for first MB tasks
        # Assume that # of every type workers is larger than MB
        # WIndindi = [[] for _ in range(GAParam['chromoSize'])]
        WIndindi = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(PTypes)]
        
        for j in range(GAParam['MB']):
            #t[j] = timeIni + popuWW[ps,j] + np.ceil(Data['TransportTime'][0,popu[ps,j]])
            # after check initialization, popu ranges from 0 to chromoSize-1, incorrect, 0906
            t[j] = timeIni + Var.popuWW[ps,j] + np.ceil(Data['TransportTime'][0,Var.popu[ps,j]+1])
            tB[j] = timeIni + Var.popuWW[ps,j]
            
            costIndividualPL[j] = sum(Data['PL0'][range(0,int(t[j])),Var.popu[ps,j]])*GAParam['ElecPrice'] + sum(Data['PL1'][range(int(t[j]),int(t[j]+Data['TTR'][Var.popu[ps,j]])),Var.popu[ps,j]])*GAParam['ElecPrice']
            costIndividualOT[j] = 0
            costIndividualBoat[j] = copy.deepcopy(Data['CostTransport'][0,Var.popu[ps,j]+1])
            
            costIndividualTask[j] = costIndividualPL[j] + costIndividualOT[j] + costIndividualBoat[j]

            for i in range(PTypes):
                if j == 0:
                    WIndindi[i][j] = list(range(TaskWorkers[i,j]))
                else:
                    WIndindi[i][j] = list(range(sum(TaskWorkers[i,range(j)]), sum(TaskWorkers[i,range(j+1)])))
            
            BIndindi[j] = j
            
            Var.startTime[ps,j] = copy.deepcopy(t[j])
            Var.departTime[ps,j] = copy.deepcopy(tB[j])
            Var.individualCost[ps,j] = copy.deepcopy(costIndividualTask[j])
            Var.individualCostPL[ps,j] = copy.deepcopy(costIndividualPL[j])
            Var.individualCostOT[ps,j] = copy.deepcopy(costIndividualOT[j])
            Var.individualCostBoat[ps,j] = copy.deepcopy(costIndividualBoat[j])
        
        DPFlag = True
        for i in range(PTypes): 
            WIndindiiRavel = [item for sublist in WIndindi[i] for item in sublist]
            if len(WIndindiiRavel) > 0:
                if GAParam['NP'][i] <= max(WIndindiiRavel):
                    DPFlag = False
                    
                    
        if DPFlag:
            # Allocation of rest workers on boarding
            #from BInventInitial import BInventInitial
            WIndindiIni = [[]  for _ in range(PTypes)]
            for pp in range(PTypes):
                WIndindiIni[pp] = copy.deepcopy(WIndindi[pp][:GAParam['MB']])
    
            BInvent = copy.deepcopy(BInventInitial(WIndindiIni,GAParam['NP']))
               
            # record the workers on the boat
            # from WorkersonBoats import WorkersonBoats
            WonB = WorkersonBoats(BInvent)
            
            # task assignment
            for j in range(GAParam['MB'],GAParam['chromoSize']):
                BAvail = np.zeros(GAParam['MB'])
                # find out if any boat can meet the workers requirement of task j
                for jj in range(GAParam['MB']):
                    BAvail[jj] = sum(np.sum(BInvent[jj],axis = 1) >= TaskWorkers[:,j])
                    
                BTemp = np.where(BAvail == PTypes)
                BTemp = np.array(BTemp).ravel()
                if BTemp.size:
                    # find out the optimal boat can cause the min cost of task j
                    BPosArray = [[] for _ in range(len(BTemp))]
                    tTempArray = [[] for _ in range(len(BTemp))]
                    for ii in range(len(BTemp)):
                        BPosArray[ii] = max([item for item in range(len(BIndindi)) if BIndindi[item] == BTemp[ii]])
                        # tTempArray[ii] = t[BPosArray[ii]] + Data['TransportTime'][popu[ps,BPosArray[ii]],popu[ps,j]]
                        # modified 0906
                        #tTempArray[ii] = copy.deepcopy(t[BPosArray[ii]] + Data['TransportTime'][Var.popu[ps,BPosArray[ii]]+1,Var.popu[ps,j]+1])
                        tTempArray[ii] = copy.deepcopy(t[BPosArray[ii]] + np.ceil(Data['TransportTime'][Var.popu[ps,BPosArray[ii]]+1,Var.popu[ps,j]+1]))
                    
                    ind = np.argmin(tTempArray)
                    tTemp = tTempArray[ind] + Var.popuWW[ps,j] + np.ceil(GAParam['DockTime'])
                    tBTemp = t[BPosArray[ind]] + Var.popuWW[ps,j] + np.ceil(GAParam['DockTime'])
                    
                    
                    if (int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['RegularEndtime']+1: 
                        t[j] = copy.deepcopy(tTemp)
                        tB[j] = copy.deepcopy(tBTemp)
                        
                        costIndividualPL[j] = sum(Data['PL0'][range(t[j]),Var.popu[ps,j]]) * GAParam['ElecPrice'] + sum(Data['PL1'][range(t[j],int(t[j]+Data['TTR'][Var.popu[ps,j]])),Var.popu[ps,j]])*GAParam['ElecPrice']
                        costIndividualOT[j] = 0
                        costIndividualBoat[j] = Data['CostTransport'][Var.popu[ps,BPosArray[ind]]+1,Var.popu[ps,j]+1]
                        costIndividualTask[j] = costIndividualPL[j] + costIndividualOT[j] + costIndividualBoat[j]

                    elif (((int(tTemp) + Data['TTR'][Var.popu[ps,j]]) > GAParam['RegularEndtime']) & ((int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['OTEndtime']+1)):
                        t[j] = tTemp
                        tB[j] = tBTemp
                        
                        costIndividualPL[j] = sum(Data['PL0'][range(t[j]),Var.popu[ps,j]]) * GAParam['ElecPrice'] + sum(Data['PL1'][range(t[j]-1,int(t[j]+Data['TTR'][Var.popu[ps,j]])),Var.popu[ps,j]])*GAParam['ElecPrice']
                        costIndividualOT[j] = GAParam['OTSalary']/4 * (t[j]%96+Data['TTR'][Var.popu[ps,j]]-GAParam['RegularEndtime']) # fix the bug sometimes OT neg 0608
                        costIndividualBoat[j] = Data['CostTransport'][Var.popu[ps,BPosArray[ind]]+1,Var.popu[ps,j]+1]
                        costIndividualTask[j] = costIndividualPL[j] + costIndividualOT[j] + costIndividualBoat[j]

                    else:
                        t[j] = 0
                        tB[j] = 0
                        pickupTime[j] = []; #101717
                        pickupTurbineOrd[j] = [] #101717
                        
                        # costIndividualPL[j] = 0 # when doesn't take PL of unscheduled tasks into account
                        costIndividualPL[j] = sum(Data['PL0'][range(96),Var.popu[ps,j]]) * GAParam['ElecPrice']*100
                        costIndividualOT[j] = 0
                        costIndividualBoat[j] = 0
                        costIndividualTask[j] = costIndividualPL[j] + costIndividualOT[j] + costIndividualBoat[j]
                    '''
                    if the task can be done in the day, then update:
                        BIndindi,
                        WIndindi,
                        WonB
                    Otherwise not    
                    '''
                    if ((int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['OTEndtime']+1):
                        BIndindi[j] = copy.deepcopy(BTemp[ind])
                    
                        # assign the worker on boat BIndindi(j) to the task j, and update the
                        # BInvent{jj} and WIndindij
                        # from updateBInventAfterAssign import updateBInventAfterAssign
                        [BInvent[BIndindi[j]], WIndindij] = updateBInventAfterAssign(BInvent[BIndindi[j]],TaskWorkers[:,j]);
                        
                        for kk in range(len(WIndindij)):
                            if type(WIndindij[kk]) == list:
                                WIndindi[kk][j] = copy.deepcopy(WIndindij[kk])
                            else:
                                WIndindi[kk][j] = WIndindij[kk].tolist()
            
                        # record the workers on the boat
                        WonB = copy.deepcopy(WorkersonBoats(BInvent))
                        
                else:
                    # if all boats don't have enough inventory workers for task j, then
                    # go back to bring some workers from previous turbines
                    
                    # find out the boat closest to meet the task workers requirements
                    BAvailTwin = copy.deepcopy(BAvail)
                    
                    while True:
                        BTemp = np.argmax(BAvailTwin)
                        BIndindi[j] = copy.deepcopy(BTemp)
                        # find out the lack worker type
                        lackPType = (np.sum(BInvent[int(BIndindi[j])],axis=1) < TaskWorkers[:,j])
                        lackPType = lackPType * 1
                        lackNumber = -np.sum(BInvent[int(BIndindi[j])],axis=1) + TaskWorkers[:,j]
                        lackNumber[lackNumber<0] = 0
                        lackPType = np.column_stack((lackPType,lackNumber))
                        
                        # locate the shortage workers on turbines
                        WPosArray = findWorkers(lackPType,WonB,WIndindi,GAParam['NP'])
                        
                        # determine which turbine should be visited to pick up enough
                        # workers
                        # if (1) the workers in one turbine are enough, then just pick up the
                        # workers on this turbine;
                        # else (2) determine which turbines should be visted to pick up
                        # enough workers.
                        
                        # summarize # of different workers on turbines now
                        WonT = summaryWonT(WPosArray)
                        
                        if np.any(np.sum(WonT,axis=1) < lackPType[lackPType[:,0]==1,1]):
                            BAvailTwin[BTemp] = 0
                            # this if-statement is to fix the bug that when any
                            # boat + all turbines can't meet the requirement, then
                            # consider to assign the boat departure to another boat
                            # to find enough workers;
                            if sum(BAvailTwin) == 0:
                                BIndindi[j] = []

                                [BTemp,pickupBoat,tBTemp,tTemp,BInvent,costTemp] = boattoVisit(BInvent,TaskWorkers[:,j],t,BIndindi,ps,Data,j,Var.popu)
                                BIndindi[j] = copy.deepcopy(BTemp)
                                break
                        else:
                            break
                     
                    # end while
                    
                    # switch into 2 cases: (1) BAvailTwin ~= 0; (2) BAvailTwin ==
                    # 0, which correspond to (1)sum(BAvailTwin) ~= 0;
                    # (2)sum(BAvailTwin) == 0
                    
                    #if sum(BAvailTwin) != 0:
                    if np.any(np.sum(WonT,axis=1) >=lackPType[lackPType[:,0]==1,1]):
                        #print BAvailTwin
                        
                        turbineID = copy.deepcopy(turbinetoVisit(WonT,lackPType))
                        #BPos = max(np.where(BIndindi[range(j)] == BTemp))
                        #print 'BIndindi[0:j]'
                        #print BIndindi[0:j]
                        #print 'BTemp'
                        #print BTemp

                        BIndindTemp = copy.deepcopy(BIndindi[0:j])
                        BPos = max([item for item in range(len(BIndindTemp)) if BIndindTemp[item] == BTemp])
                        #print 'BPos'
                        #print BPos
                        
                        # if there is only 1 turbine needed to be visited, no optimization strategy
                        if np.size(turbineID) == 1:
                            pickupTurbineOrd[j] = copy.deepcopy(turbineID)
                            pickupTime[j] = t[BPos]
                            #tBTemp = max(t[BPos] + Data['TransportTime'][popu[ps,BPos],popu[ps,turbineID]],t[turbineID]+Data['TTR'][popu[ps,turbineID]-1]) + np.ceil(GAParam['DockTime'])
                            tBTemp = max(t[BPos] + Data['TransportTime'][Var.popu[ps,BPos]+1,Var.popu[ps,turbineID]+1],t[turbineID]+Data['TTR'][Var.popu[ps,turbineID]]) + np.ceil(GAParam['DockTime'])
                            #tTemp = tBTemp + Data['TransportTime'][popu[ps,turbineID],popu[ps,j]] + np.ceil(GAParam['DockTime'])
                            tTemp = tBTemp + Data['TransportTime'][Var.popu[ps,turbineID]+1,Var.popu[ps,j]+1] + np.ceil(GAParam['DockTime'])
                            costTemp = Data['CostTransport'][Var.popu[ps,BPos]+1, Var.popu[ps,turbineID]+1] + Data['CostTransport'][Var.popu[ps,turbineID]+1,Var.popu[ps,j]+1]
                        else:
                            # optimize this tour using traversal
                            costTemp, pickupTurbineOrd[j], pickupTime[j], tBTemp, tTemp = pickupTourOptimization(BPos,j,Var.popu,ps,turbineID,t,Data);
                        '''
                        if the task can be done in the day, then update:
                            BIndindi,
                            WIndindi,
                            WonB
                        Otherwise not    
                        '''
                        if (int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['OTEndtime']+1:
          
                            # update BInvent, WonB
                            # (BIndindi has been updated above)
                            BInvent[int(BIndindi[j])] = updateBInventAfterPickup(BInvent[int(BIndindi[j])],turbineID,WIndindi)
                            
                            # record the workers on the boat
                            WonB = copy.deepcopy(WorkersonBoats(BInvent))
                    '''
                        if the task can be done in the day, then update:
                            BIndindi,
                            WIndindi,
                            WonB
                        Otherwise not    
                    '''
                    if (int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['OTEndtime']+1:
                        # update the BInvent after assign the workers out.
                        [BInvent[BIndindi[j]], WIndindij] = updateBInventAfterAssign(BInvent[int(BIndindi[j])],TaskWorkers[:,j])
                        
                        for kk in range(len(WIndindij)):
                            WIndindi[kk][j] = copy.deepcopy(WIndindij[kk])
                        
                        # record the workers on the boat
                        WonB = copy.deepcopy(WorkersonBoats(BInvent))
                    else:
                        BIndindi[j] = []
                    
                    # cost computation
                    if (int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['RegularEndtime']+1:
                        t[j] = copy.deepcopy(tTemp)
                        tB[j] = copy.deepcopy(tBTemp)
                        
                        costIndividualPL[j] = sum(Data['PL0'][range(t[j]),Var.popu[ps,j]]) * GAParam['ElecPrice'] + sum(Data['PL1'][range(t[j],int(t[j]+Data['TTR'][Var.popu[ps,j]])),Var.popu[ps,j]])*GAParam['ElecPrice']
                        costIndividualOT[j] = 0
                        costIndividualBoat[j] = copy.deepcopy(costTemp)
                    elif (((int(tTemp) + Data['TTR'][Var.popu[ps,j]]) > GAParam['RegularEndtime']) & ((int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['OTEndtime']+1)):
                        t[j] = copy.deepcopy(tTemp)
                        tB[j] = copy.deepcopy(tBTemp)
                        
                        costIndividualPL[j] = sum(Data['PL0'][range(t[j]),Var.popu[ps,j]]) * GAParam['ElecPrice'] + sum(Data['PL1'][range(t[j],int(t[j]+Data['TTR'][Var.popu[ps,j]])),Var.popu[ps,j]])*GAParam['ElecPrice']
                        costIndividualOT[j] = GAParam['OTSalary']/4 * (t[j]%96+Data['TTR'][Var.popu[ps,j]]-GAParam['RegularEndtime'])
                        costIndividualBoat[j] = copy.deepcopy(costTemp)
                        
                    else:
                        t[j] = 0
                        tB[j] = 0
                        pickupTime[j] = []; #101717
                        pickupTurbineOrd[j] = [] #101717
                        # costIndividualPL[j] = 0 # when doesn't take PL of unscheduled tasks into account
                        costIndividualPL[j] = sum(Data['PL0'][range(96),Var.popu[ps,j]]) * GAParam['ElecPrice']*100##
                        costIndividualOT[j] = 0
                        costIndividualBoat[j] = 0
                        #costIndividualTask[j] = 0
                        
                    costIndividualTask[j] = costIndividualPL[j] + costIndividualOT[j] + costIndividualBoat[j]
                
                    
                Var.startTime[ps,j] = t[j]
                Var.departTime[ps,j] = tB[j]
                Var.individualCost[ps,j] = copy.deepcopy(costIndividualTask[j])
                Var.individualCostPL[ps,j] = copy.deepcopy(costIndividualPL[j])
                Var.individualCostOT[ps,j] = copy.deepcopy(costIndividualOT[j])
                Var.individualCostBoat[ps,j] = copy.deepcopy(costIndividualBoat[j])
            #print 'BIndindi before'
            #print BIndindi
            nanPos = [item for item in range(len(BIndindi)) if np.isnan(BIndindi[item])]
                        
            for item in nanPos:
                BIndindi[item] = []
            
            
            Var.BInd[ps] = copy.deepcopy(BIndindi)
            Var.WInd[ps] = copy.deepcopy(WIndindi)
            Var.BInventFinal[ps] = copy.deepcopy(BInvent)
    
            Var.pickupTurbineOrdAll[ps] = copy.deepcopy(pickupTurbineOrd)
            Var.pickupTimeAll[ps] = copy.deepcopy(pickupTime)
                
            try:
                #pickupBoatAll[ps] = pickupBoat
                Var.pickupBoatAll[ps][j] = copy.deepcopy(pickupBoat) #101717
            except:
                pass
                
            BEnd = np.zeros((GAParam['MB']), dtype = np.int)
            for bi in range(GAParam['MB']):

                BIndTemp = copy.deepcopy(Var.BInd[ps])
                BEnd[bi] = max([item for item in range(len(BIndTemp)) if BIndTemp[item] == bi])
            
            Var.BEndFinal[ps] = copy.deepcopy(BEnd)      
            Var.totalCost[ps] = sum(Var.individualCost[ps,:])+sum(GAParam['BoatRentPrice']*np.ceil(t[BEnd]/96.0))
            Var.totalCostPL[ps] = sum(Var.individualCostPL[ps,:]);
            Var.totalCostOT[ps] = sum(Var.individualCostOT[ps,:]);
            Var.totalCostBoat[ps] = sum(Var.individualCostBoat[ps,:])+sum(GAParam['BoatRentPrice']*np.ceil(t[BEnd]/96.0))
        else:
        # if DPFlag is false, then skip the DP, cand set totalCost a big number
            # totalCost[ps] = 1e10  # when doesn't take PL of unscheduled tasks into account
            Var.totalCost[ps] = 10e5
    
    Var.fitness_value = 1.0 / Var.totalCost
    return Var



#import numpy as np
# from unionUsedWorkers import unionUsedWorkers
# from equipartitionGroup import equipartitionGroup
def BInventInitial(WIndInitial,NP):
    
    # this func is used to assign the work on boarding at initial stage.
    # Based on the variables WIndInitial and NP, the available workers of each
    # type can be calculated. The the available workers will be assigned to
    # different boats "randomly" and "evenly".
        
    # Input: 
    #       WIniInitial: the workers has already been assigned to the task;
    #       NP: the number of workers of each type;
    # Output:
    #       BInvent: the inventory of each boat.
    
    MB = len(WIndInitial[0])
    PType = len(WIndInitial)
    
    
    BInvent = [ [] for _ in range(MB)]
    for i in np.arange(0,MB):
        # BInvent{i} = zeros(PType,NP(1)); version 1.2
        BInvent[i]=np.zeros((PType,np.max(NP))) # version 1.3 - 20170127

    usedWorker = unionUsedWorkers(WIndInitial)
    
    unusedWorker = [[] for _ in range(PType)]
    for i in np.arange(0,PType).reshape(-1):
        # unusedWorker[i] = np.setdiff1d(np.arange(1,NP[i]+1),usedWorker[i]) 
        unusedWorker[i] = np.setdiff1d(np.arange(0,NP[i]),usedWorker[i])

    for i in np.arange(0,PType).reshape(-1):
        BInventTemp = equipartitionGroup(unusedWorker[i],MB)
        # 注：由于 BInventTemp表示的是位置，因此此处把其中的值都减去1,创建一个新的变量BInventTemp2
        # BInventTemp2 = [[] for _ in range(np.size(BInventTemp))]
        # for m in np.arange(0,np.size(BInventTemp)):
        #     BInventTemp2[m] = np.asarray(BInventTemp[m]) - 1
        
        # 注：Matlab中此处中每个cell进行计算，如果BInventTemp2[j]为零需要额外的处理。
        for j in np.arange(0,MB).reshape(-1):
            if np.size(BInventTemp[j]) == 0:
                BInvent[j] = copy.deepcopy(BInvent[j])
            else: 
                BInvent[j][i,BInventTemp[j]] = 1
                
    return BInvent


def unionUsedWorkers(WIndInitial):

    MB = len(WIndInitial[0])
    PType = len(WIndInitial)

    usedWorkers = [ [] for _ in range(PType)]
    for i in np.arange(0,PType).reshape(-1):
        for j in np.arange(0,MB).reshape(-1):
            if np.size(WIndInitial[i][j]) != 1:
                usedWorkers[i]=usedWorkers[i] + list(WIndInitial[i][j])
            elif np.size(WIndInitial[i][j]) == 1:
                #usedWorkers[i]=usedWorkers[i] + [WIndInitial[i][j]]
                usedWorkers[i].append(WIndInitial[i][j][0]) 
    return usedWorkers


#import random as rand
def equipartitionGroup(sample,groupNo):
    # this equation is used to partition the samples into different groups to
    # meet 2 requirements:
    #   (1) the number of samples in each group should be as even as possible;
    #   (2) the samples should be partitioned randomly

    
    # sample4 can help to test this script
    if np.all(np.size(sample)==0):
        group =[ [] for _ in np.arange(0,groupNo).reshape(-1)]
    else:
        sampLen=np.size(sample)
        group = []
        for i in np.arange(0,np.mod(sampLen,groupNo)).reshape(-1):
            # sample2 can help to test this script
            k=np.ceil(float(sampLen)/float(groupNo))
            # sample3 can help to test this script
            if np.size(sample) != 1:
                group.append(random.sample(sample.tolist(), int(k)))
            # sample2 can help to test this script
            else:
                group.append(sample)
            sample=np.setdiff1d(sample,group[i])

        for i in np.arange(np.mod(sampLen,groupNo),groupNo).reshape(-1):
            k=np.floor(float(sampLen)/float(groupNo))

            if np.size(sample) != 1:
                group.append(random.sample(sample.tolist(), int(k)))
            else:
                group.append(sample)
            sample=np.setdiff1d(sample,group[i])
    return group


def WorkersonBoats(BInvent):

    PType = np.size(BInvent, 1)
    MB    = np.size(BInvent, 0)
    WonB = []
    for i in range(0, PType):
        temp = []
        for j in range(0, MB):
            temp = temp + np.where(BInvent[j][i] == 1)[0].tolist()

        WonB.append(temp)
        del temp
        
    return WonB


def updateBInventAfterAssign(BInventj, TaskjWorker):

    WIndj = []
    for i in range(0, len(TaskjWorker)):
        temp_BInventj, temp_WIndj  = inventWorkerSample(BInventj[i], int(TaskjWorker[i]))
        BInventj[i, ] = copy.deepcopy(temp_BInventj)
        WIndj.append(temp_WIndj)

    return BInventj, WIndj


def inventWorkerSample(invent,numberReq):

# input:
#       invent: a binary vector
#       numberReq: number of samples
# output:
#       indSam: the indice of sampled workers from inventory
#       newInvent: the remaining inventory of the workers on certain boat
    
    if numberReq == 0:
        newInvent=copy.deepcopy(invent)
        indSam=[]
    else:
        #popuInd=np.where(np.array([invent]) == 1)
        popuInd = np.where(invent == 1) 
        # 注：此处np.where返回的是横纵位置，我们只需要横向位置，所以加了[1]
        # 注：此处是如果有一个为1，那么把它变成0，如果有多个为1，那么选一个变成0
        popuInd = popuInd[0]
        if np.size(popuInd) == 1:
            indSam=copy.deepcopy(popuInd)
        else:
            indSam = np.random.choice(popuInd,numberReq,replace=False)
        
        newInvent = np.array(invent)
        newInvent[indSam] = 0
    return newInvent,indSam


def summaryWonT(WPosArray):
#this function is used to summarize the number of different workers on the
#turbines which have workers currently


    lackPTypeNum = np.size(WPosArray,0) 
    #lackPTypeNum = len(WPosArray) 
    
    workTurbine = np.unique(WPosArray)
    workTurbine = workTurbine[~np.isnan(workTurbine)]
    
    if len(workTurbine) == 0:
        WonT = np.array([[0]])
    else:
        workTurbine = workTurbine.astype(int)
        
        # find out the workers numbers of different type on each turbine
        # WonT = np.zeros((np.size(WPosArray,0),np.size(workTurbine,0)-1))
        WonT = np.zeros((np.size(WPosArray,0),max(workTurbine)+1), dtype=np.int)
        for i in np.arange(0,lackPTypeNum).reshape(-1):
            # 注np.histogram实现了和matlab中histc相同的功能，但是返回的是2个tuple的信息，但是只需要第[0]个tuple的信息
            #WonT[i,:]=np.histogram((WPosArray[i,:]),np.arange(0,max(workTurbine)+1))[0]
            WonT[i,:] = np.histogram((WPosArray[i,:][~np.isnan(WPosArray[i,:])]),np.arange(0,max(workTurbine)+2))[0]
    
    return WonT


#import itertools as itt
#import random

def boattoVisit(BInvent,TaskWorkersj,t,BIndindi,ps,Data,j,popu):
    
    # change BIndindi from list with [] to array with nan
    for iii in range(len(BIndindi)):
        if BIndindi[iii] == []:
            BIndindi[iii] = np.nan
    BIndindi = np.array(BIndindi)  
    # summary of the inventory on each boat
    BInventSum = np.empty((BInvent[0].shape[0],len(BInvent)))
    
    for i in range(len(BInvent)):
        BInventSum[:,i] = np.sum(BInvent[i], axis = 1)
        
    # find out the boat combination can meet the worker requirement
    for ii in range(2, len(BInvent)+1):
        combj = [list(x) for x in itt.combinations(range(0,len(BInvent)), ii)]
        combj = np.array(combj)
        for jj in range(0,combj.shape[0]):
            #if sum(np.sum(BInventSum[:,combj[jj,:]], axis = 1) <= TaskWorkersj):
            if sum(np.sum(BInventSum[:,combj[jj,:]], axis = 1) < TaskWorkersj):            
                combj[jj,:] = np.zeros((1,ii))
        if sum(sum(combj)) != 0:
            break
        
    
    # find out the best tour
    delColIdx = np.where(np.sum(combj, axis = 1) == 0)
    combj = np.delete(combj, delColIdx, axis = 0)
    
    
    #print 'range(combj.shape[0])'
    #print ( range(combj.shape[0]))
    
    for i in range(combj.shape[0]):
        
        tourTemp = [list(x) for x in itt.permutations(combj[i])]
        tourTemp = np.array(tourTemp)
        if i == 0:
            tourAll = copy.deepcopy(tourTemp)
        else:
            tourAll = np.row_stack((tourAll, tourTemp))

    BPosArray = np.empty((len(BInvent)), dtype=np.int)   
    for ii in range(len(BInvent)):
        BPosArray[ii] = np.max(np.where(BIndindi == ii))    
    
    import sys
    try:
        costTemp = np.zeros(tourAll.shape[0])
    except:
        print ('Current worker numbers are too few! No schedule found!')
        sys.exit(1)
        
    for jj in range(tourAll.shape[0]):
        for i in range(tourAll.shape[1]):
            if i < tourAll.shape[1]-1:
                if popu.ndim == 2: # to check whether it's a 2d array or not
                    costTemp[jj] = costTemp[jj] + Data['CostTransport'][popu[ps,BPosArray[tourAll[jj,i]]]+1,popu[ps,BPosArray[tourAll[jj,i+1]]]+1]
                else:
                    costTemp[jj] = costTemp[jj] + Data['CostTransport'][popu[BPosArray[tourAll[jj,i]]]+1,popu[BPosArray[tourAll[jj,i+1]]]+1]
            else:
                if popu.ndim == 2:
                    costTemp[jj] = costTemp[jj] + Data['CostTransport'][popu[ps,tourAll[jj,i]]+1, popu[ps,j]+1]
                else:
                    costTemp[jj] = costTemp[jj] + Data['CostTransport'][popu[tourAll[jj,i]]+1, popu[j]+1]
                
    
    # find out the tour which cause the least cost
    indMinCost = np.where(costTemp == min(costTemp))
    BoatOrd = tourAll[indMinCost]
    # convert BoatOrd into 1d shape
    BoatOrd = BoatOrd[0]
    
    pickupTime = np.zeros(len(BoatOrd)-1)
    pickupTime[0] = copy.deepcopy(t[BPosArray[BoatOrd[0]]])
    if popu.ndim == 2:
        costTemp = Data['CostTransport'][popu[ps,BPosArray[BoatOrd[0]]], popu[ps,BPosArray[BoatOrd[1]]]]
    else:
        costTemp = Data['CostTransport'][popu[BPosArray[BoatOrd[0]]], popu[BPosArray[BoatOrd[1]]]]          
               
    
    
    for i in range(len(BoatOrd)-1):
        if i < len(BoatOrd)-2:
            if popu.ndim == 2:
                pickupTime[i+1] = np.max((pickupTime[i] + Data['TransportTime'][popu[ps,BPosArray[BoatOrd[i]]], popu[ps,BPosArray[BoatOrd[i+1]]]],  t[BPosArray[BoatOrd[i+1]]]))
                costTemp = costTemp + Data['CostTransport'][popu[ps,BPosArray[BoatOrd[i+1]]], popu[ps,BPosArray[BoatOrd[i+2]]]]
            else:
                pickupTime[i+1] = np.max((pickupTime[i] + Data['TransportTime'][popu[BPosArray[BoatOrd[i]]], popu[BPosArray[BoatOrd[i+1]]]],  t[BPosArray[BoatOrd[i+1]]]))
                costTemp = costTemp + Data['CostTransport'][popu[BPosArray[BoatOrd[i+1]]], popu[BPosArray[BoatOrd[i+2]]]]
            
                
        else:
            if popu.ndim == 2:
                tBTemp = np.max((pickupTime[i] + Data['TransportTime'][popu[ps,BPosArray[BoatOrd[i]]], popu[ps,BPosArray[BoatOrd[i+1]]]],  t[BPosArray[BoatOrd[i+1]]]))
                tTemp = tBTemp + Data['TransportTime'][popu[ps,BPosArray[BoatOrd[i+1]]], popu[ps,j]]
                costTemp = costTemp + Data['CostTransport'][popu[ps,BPosArray[BoatOrd[i+1]]], popu[ps,j]]
            else:
                tBTemp = np.max((pickupTime[i] + Data['TransportTime'][popu[BPosArray[BoatOrd[i]]], popu[BPosArray[BoatOrd[i+1]]]],  t[BPosArray[BoatOrd[i+1]]]))
                tTemp = tBTemp + Data['TransportTime'][popu[BPosArray[BoatOrd[i+1]]], popu[j]]
                costTemp = costTemp + Data['CostTransport'][popu[BPosArray[BoatOrd[i+1]]], popu[j]]
            
    BTemp = copy.deepcopy(BoatOrd[0])
    boattoPickup = copy.deepcopy(BoatOrd[1:])
    BPosArray = copy.deepcopy(BPosArray[1:])
    
    
    pickupBoatj = dict()
    pickupBoatj['boatID'] = copy.deepcopy(boattoPickup)
    pickupBoatj['boatPos'] = copy.deepcopy(BPosArray)
    pickupBoatj['DTime'] = copy.deepcopy(pickupTime)
    
    #determine how many workers should pick up from each boat
    newTaskWorkers = np.int32(TaskWorkersj) - np.int32(np.sum(BInvent[BTemp],axis = 1))
    newTaskWorkers[newTaskWorkers<0] = 0
    
    
    pickupBoatj['pickupWorker'] = [[[] for _ in range(len(boattoPickup))] for _ in range(len(newTaskWorkers))]
    for i in range(len(boattoPickup)):
        for ii in range(len(newTaskWorkers)):
            if newTaskWorkers[ii] != 0:
                if newTaskWorkers[ii] >= np.sum(BInvent[boattoPickup[i]][ii]):
                    pickupBoatj['pickupWorker'][ii][i] = np.where(BInvent[boattoPickup[i]][ii] == 1)
                    newTaskWorkers[ii] = newTaskWorkers[ii] - np.sum(BInvent[boattoPickup[i]][ii])
                    BInvent[boattoPickup[i]][ii] = 0
                    BInvent[BTemp][ii][pickupBoatj['pickupWorker'][ii][i]] = 1
                else:
                    pickupBoatj['pickupWorker'][ii][i] = random.sample(np.where(BInvent[boattoPickup[i]][ii] == 1)[0].tolist(),newTaskWorkers.tolist()[ii])
                    BInvent[boattoPickup[i]][ii][pickupBoatj['pickupWorker'][ii][i]] = 0
                    newTaskWorkers[ii] = 0
                    BInvent[BTemp][ii][pickupBoatj['pickupWorker'][ii][i]] = 1
        if sum(newTaskWorkers) == 0:
            break
        
    return BTemp, pickupBoatj, tBTemp, tTemp, BInvent, costTemp


def findWorkers(lackPType,WonB,WInd,NP):

    # This function is used to find the locations of required workers from
    # turbine;
    # output is the WPosArray of require PTypes
        
    # version 1.1
    # PTypes = size(WInd,1);
    # lackPTypeInd = find(lackPType == 1);
    # WPosArray = zeros(length(lackPTypeInd),NP(1));
    # for jj = 1:length(lackPTypeInd)    
    #     for ii = setdiff(1:NP(1),WonB{lackPTypeInd(jj)})
    #         WPosArray(jj,ii) = find(WInd(lackPTypeInd(jj),:) == ii,1,'last');
    #     end
    # end
    
     
    lackPTypeInd = np.where(lackPType[:,0] == 1)[0]
    #WPosArray = np.zeros((len(lackPTypeInd),max(NP)))
    WPosArray = [[[] for _ in range(max(NP))] for _ in range(len(lackPTypeInd))]
    #WPosArray =  len(lackPTypeInd),max(NP)

    for jj in np.arange(0,len(lackPTypeInd)).reshape(-1):
        # 注：下列语句会产生一个警告，由Python特性决定，对结果没有影响。__main__:1: VisibleDeprecationWarning: converting an array with ndim > 0 to an index will result in an error in the future
        workersonTurb = np.setdiff1d(np.arange(0,NP[lackPTypeInd[jj]]),WonB[lackPTypeInd[jj]])

        for ii in workersonTurb.reshape(-1):
            # WPosArray(jj,ii) = find(WInd(lackPTypeInd(jj),:) == ii,1,'last');
            for j in np.arange(0,np.size(WInd,1)).reshape(-1):
                # if (find(WInd{jj,j} == ii)) # vresion 1.1, bugs
                #if (np.where(WInd[lackPTypeInd[jj],j] == ii)):
                #if (np.where(WInd[lackPTypeInd[jj]][j] == ii)):
                temp = copy.deepcopy(WInd[lackPTypeInd[jj]][j])  
                if len([item for item in range(len(temp)) if temp[item] == ii]):
                    WPosArray[jj][ii]=copy.deepcopy(j)
                    
    # now WPosArray is a 2d list with [], which needs to be converted to a array with nan
    WPosArray = np.array([[(element == [])*np.NaN or element for element in sublist] for sublist in WPosArray])
    return WPosArray


def turbinetoVisit(WonT,lackPType):
    
    lackPNum = lackPType[lackPType[:,0] == 1, 1]
    
    # if original length of lackPNum is 1, the variable WonT is a 1d array, 
    # which needs to be conveted to a 2d array
    if len(WonT.shape) == 1:
        WonT = np.reshape(WonT, (-1, len(WonT)))
    
    # to find out whether each turbine can meet the requirement of the number
    # of the different type of worker
    TurbFlag = np.empty(WonT.shape)
    for i in range(len(lackPNum)):
        TurbFlag[i,:] = WonT[i,:] >= lackPNum[i]
        
    # if one turbine can meet the requirement, then just need to visit this
    # certain turbine
    if sum(np.sum(TurbFlag, axis=0) == len(lackPNum)):
        turbineID = np.where(np.sum(TurbFlag, axis=0) == len(lackPNum))
        # now turbineID is a tuple, first element in which is an array, so the
        # array needs to be taken out first, then the first element of the array be
        # taken
        turbineID = copy.deepcopy(turbineID[0][0])
        
        # one turbine can not meet the requirement, then recursive approach is
        # employed to find out the combination of turbines to meet the requirement
    else:
        if sum(sum(TurbFlag)) == 0:
            turbineID = np.where(WonT[0,:] == max(WonT[0,:]))
        else:
            turbineID = np.where(np.sum(TurbFlag,axis=0) == max(np.sum(TurbFlag,axis=0)))
        # now turbineID is a tuple, first element in which is an array, so the
        # array needs to be taken out first, then the first element of the array be
        # taken
        turbineID = copy.deepcopy(turbineID[0][0])
        
        # find out the remaining requirement when first turbineID is determined
        # update lackPType
        newLackPNum = lackPNum - WonT[:,turbineID]
        newLackPNum[newLackPNum<0] = 0
        newLackPType = copy.deepcopy(lackPType)
        newLackPType[(newLackPType[:,0] == 1),1] = copy.deepcopy(newLackPNum)
        newLackPType[(newLackPType[:,1] == 0),0] = 0
        lackPType = copy.deepcopy(newLackPType)
        
        # update WonT
        delRowIdx = np.where(lackPNum <= WonT[:,turbineID])
        WonT = np.delete(WonT, delRowIdx, axis = 0)
        WonT[:,turbineID]= 0
        
        turbineID = np.array(turbineID)
        # use recursive to get the final turbineID combination
        turbineID = np.append(turbineID, turbinetoVisit(WonT,lackPType))
        
    if np.size(turbineID) == 1:
        newTurbineID = np.array([turbineID])
        turbineID = np.zeros(1)
        turbineID = copy.deepcopy(newTurbineID)
            
    return turbineID

#import itertools as itt
def pickupTourOptimization(BPos,j,popu,ps,turbineID,t,Data):
    # input: turbineID, t, TTR
    # process: optimize this tour using traversal
    # output:
    #        costTemp: cost of pick up process
    #        pickupTurbineOrd: the turbine order by which the boat visits
    #        pickupTime: start time of each pickup
    #        tTemp: the start time when the boat can be able to departure to the task

    # to fix the bug if popu only has one row (pr when popu is a 1d array)
    if popu.ndim == 1:
        popu = np.reshape(popu, (1, -1))
    
    
    
    tours = [list(x) for x in itt.permutations(turbineID)]
    numTours = len(tours)
    costTemp = np.zeros(numTours)
    
    
    for jj in range(numTours):
        for i in range(len(turbineID)):
            if i == 0:
                costTemp[jj] = Data['CostTransport'][popu[ps,BPos]+1, popu[ps,tours[jj][0]]+1]
            else:
                costTemp[jj] = costTemp[jj] + Data['CostTransport'][popu[ps,tours[jj][i-1]]+1, popu[ps,tours[jj][i]]+1]
    
    # find out the tour which cause the least cost
    indMinCost = np.where(costTemp == min(costTemp))[0][0]
    pickupTurbineOrd = copy.deepcopy(tours[indMinCost])
    
    # determine the visit time for each turbine and once the optimal tour is found
    pickupTime = np.zeros(len(pickupTurbineOrd))
    pickupTime[0] = copy.deepcopy(t[BPos])
    
    
    for i in range(len(pickupTurbineOrd)):
        if i == 0:
            pickupTime[i+1] = max(t[BPos] + Data['TransportTime'][popu[ps,BPos]+1, popu[ps,pickupTurbineOrd[0]]+1], t[pickupTurbineOrd[0]]+Data['TTR'][popu[ps,pickupTurbineOrd[0]]])
            pickupTime[i+1] = np.ceil(pickupTime[i+1])
            costTemp = Data['CostTransport'][popu[ps,BPos]+1, popu[ps,pickupTurbineOrd[0]]+1]
        elif i < len(pickupTurbineOrd)-1:
            pickupTime[i+1] = max(pickupTime[i] + Data['TransportTime'][popu[ps,pickupTurbineOrd[i-1]]+1, popu[ps,pickupTurbineOrd[i]]+1], t[pickupTurbineOrd[i]]+Data['TTR'][popu[ps,pickupTurbineOrd[i]]])
            pickupTime[i+1] = np.ceil(pickupTime[i+1]) 
            costTemp = costTemp + Data['CostTransport'][popu[ps,pickupTurbineOrd[i-1]]+1, popu[ps,pickupTurbineOrd[i]]+1]
        else:
            tBTemp = max(pickupTime[i] + Data['TransportTime'][popu[ps,pickupTurbineOrd[i-1]]+1, popu[ps,pickupTurbineOrd[i]]+1], t[pickupTurbineOrd[i]]+Data['TTR'][popu[ps,pickupTurbineOrd[i]]])
            tTemp = tBTemp + np.ceil(Data['TransportTime'][popu[ps,pickupTurbineOrd[i]]+1,popu[ps,j]+1]) 
            #tTemp = tBTemp + np.ceil(Data['TransportTime'][popu[ps,pickupTurbineOrd[i]]+1,popu[ps,j]+1]) 
            costTemp = costTemp + Data['CostTransport'][popu[ps,pickupTurbineOrd[i-1]]+1, popu[ps,pickupTurbineOrd[i]]+1]
    
    costTemp = costTemp + Data['CostTransport'][popu[ps,pickupTurbineOrd[i]]+1,popu[ps,j]+1]
    
    return costTemp, pickupTurbineOrd, pickupTime, tBTemp, tTemp


def updateBInventAfterPickup(BInventj, turbineID, WInd):

    
    # for i in range(0, len(turbineID)):
    for i in range(0, np.size(turbineID)): 
        for j in range(0, np.size(WInd, 0)):

            # turbineID indices are wrt matlab - python indices start from 0 hence : turbineID[i]-1
            #if np.size(WInd[j, turbineID[i]-1]) != 0:
            if np.size(WInd[j][turbineID[i]]) != 0:
                temp = copy.deepcopy(WInd[j][turbineID[i]])
                for k in range(0, np.size(WInd[j][turbineID[i]])):
                    
                    # matlab indices in temp start from 1, python it starts from 0
                    #BInventj[j][temp[k]-1] = 1
                    BInventj[j][temp[k]] = 1 

    return BInventj


def rank(Var, GAParam, G):
    # sample ranking based on fitness
# =============================================================================
#     global fitness_value
#     global fitness_table
#     global fitness_avg
#     global best_fitness
#     global best_individual
#     global best_individualWW
#     global best_generation
#     global popu
#     global popuWW
#     global G
#     global startTime
#     global bestStartTime
#     global departTime
#     global bestDepartTime
#     global BInd
#     global WInd
#     global BInventFinal # 0723
#     global BEndFinal # 0723
#     global best_BInd
#     global best_WInd
#     global best_BInventFinal # 0723
#     global best_BEndFinal # 0723
#     global pickupTurbineOrdAll
#     global pickupTimeAll
#     global pickupBoatAll
#     global best_pickupBoat
#     global best_pickupTurbineOrd
#     global best_pickupTime
#     global totalCostPL
#     global totalCostOT
#     global totalCostBoat
#     global individualCost
#     global individualCostPL
#     global individualCostOT
#     global individualCostBoat
#     global best_totalCostPL   
#     global best_totalCostOT
#     global best_totalCostBoat
#     global best_individualCost
#     global best_individualCostPL
#     global best_individualCostOT
#     global best_individualCostBoat
# 
#     fitness_table = np.zeros(GAParam['popSize'])
# =============================================================================
    for i in range(GAParam['popSize']):
        Var.fitness_table[i] = 0.
    
    ind = np.argsort(Var.fitness_value)
    Var.popu = copy.deepcopy(Var.popu[ind])
    Var.popuWW = copy.deepcopy(Var.popuWW[ind])
    Var.startTime = copy.deepcopy(Var.startTime[ind])
    Var.departTime = copy.deepcopy(Var.departTime[ind])
    Var.fitness_value = copy.deepcopy(Var.fitness_value[ind])
    temp = [Var.BInd[i] for i in ind]
    BInd = copy.deepcopy(temp)
    
    #WInd = WInd[ind] # version 1.2 # doesn't work, following is modfied in 0417
    temp = [Var.WInd[i] for i in ind]
    WInd = copy.deepcopy(temp)
    

    temp = [Var.BInventFinal[i] for i in ind]
    BInventFinal = copy.deepcopy(temp)
    temp = [Var.BEndFinal[i] for i in ind]
    BEndFinal = copy.deepcopy(temp)
    
    Var.fitness_table = np.cumsum(Var.fitness_value)
    Var.fitness_avg[G] = Var.fitness_table[GAParam['popSize']-1] / GAParam['popSize']
    
    if Var.fitness_value[GAParam['popSize']-1] > Var.best_fitness:
        Var.best_fitness = copy.deepcopy(Var.fitness_value[GAParam['popSize']-1])
        Var.best_generation = copy.deepcopy(G)
        Var.best_individual = copy.deepcopy(Var.popu[GAParam['popSize']-1])
        Var.best_individualWW = copy.deepcopy(Var.popuWW[GAParam['popSize']-1])
        Var.bestStartTime = copy.deepcopy(Var.startTime[GAParam['popSize']-1])
        Var.bestDepartTime = copy.deepcopy(Var.departTime[GAParam['popSize']-1])
        Var.best_BInd = copy.deepcopy(BInd[GAParam['popSize']-1])
        Var.best_WInd = copy.deepcopy(WInd[GAParam['popSize']-1]) # version 1.2
        Var.best_BInventFinal = copy.deepcopy(BInventFinal[GAParam['popSize']-1]) # 0723
        Var.best_BEndFinal = copy.deepcopy(BEndFinal[GAParam['popSize']-1])
       

        Var.best_pickupTurbineOrd = copy.deepcopy(Var.pickupTurbineOrdAll[ind[-1]])
        Var.best_pickupTime = copy.deepcopy(Var.pickupTimeAll[ind[-1]])
        Var.best_pickupBoat = copy.deepcopy(Var.pickupBoatAll[ind[-1]])
        Var.best_totalCostPL = copy.deepcopy(Var.totalCostPL[ind[-1]])
        Var.best_totalCostOT = copy.deepcopy(Var.totalCostOT[ind[-1]])
        Var.best_totalCostBoat = copy.deepcopy(Var.totalCostBoat[ind[-1]])
        Var.best_individualCost = copy.deepcopy(Var.individualCost[ind[-1]])
        Var.best_individualCostPL = copy.deepcopy(Var.individualCostPL[ind[-1]])
        Var.best_individualCostOT = copy.deepcopy(Var.individualCostOT[ind[-1]])
        Var.best_individualCostBoat = copy.deepcopy(Var.individualCostBoat[ind[-1]])
    
    del i
    del ind
    return Var


def selection(Var, GAParam):
    # this function is used to select good results the best after GA.
    
    #global popu
    #global fitness_table
    popu_new=np.zeros((np.size(Var.popu,0),np.size(Var.popu,1)))
    popuWW_new=np.zeros((np.size(Var.popuWW,0),np.size(Var.popuWW,1)))

    for i in range(GAParam['popSize']):
        # 注：GAParam['popSize']是fitness_table里的位置信息，因此要减1
        r=np.dot(np.random.random(), Var.fitness_table[GAParam['popSize']-1])
    #     first = 1;
    #     last = GAParam.popSize;
    #     mid = round((last+first)/2);
    #     idx = -1;
    #     while (first <= last) && (idx == -1)
    #         if r > fitness_table(mid)
    #             first = mid;
    #         elseif r < fitness_table(mid)
    #             last = mid;
    #         else
    #             idx = mid;
    #             break;
    #         end
    #         mid = round((last+first)/2);
    #         if (last - first) == 1
    #             idx = last;
    #             break;
    #         end
    #    end # version 1.1

        
        idx=(abs(r - Var.fitness_table)).argmin(0)
    
        for j in range(GAParam['chromoSize']):
            popu_new[i,j]=copy.deepcopy(Var.popu[idx,j])
            
    # same operation to popuWW    
    for i in range(0,GAParam['popSize']):
        r=np.dot(np.random.random(), Var.fitness_table[GAParam['popSize']-1])
        idx=(abs(r - Var.fitness_table)).argmin(0)

        for j in np.arange(0,GAParam['chromoSize']).reshape(-1):
            popuWW_new[i,j]=copy.deepcopy(Var.popuWW[idx,j])


    if GAParam['elitism']:
        p=GAParam['popSize'] - 1
    else:
        p=GAParam['popSize']

    for i in range(p):
        for j in range(GAParam['chromoSize']):
            Var.popu[i,j]=copy.deepcopy(popu_new[i,j])
            Var.popuWW[i,j]=copy.deepcopy(popuWW_new[i,j])
    
    del i
    del j
    del popu_new
    del popuWW_new
    del idx
    return Var


def crossover(Var, GAParam):
    #global popu
    #global popuWW
    
    # if priority task list exist, then no crossover or mutation to that part, 171101
    if GAParam['Priority_condition'] == True:
        crossover_len = GAParam['chromoSize'] - len(GAParam['Priority_list'])
        priority_len = len(GAParam['Priority_list'])
    else:
        crossover_len = GAParam['chromoSize']
        priority_len = 0
    
    crossover_popu = Var.popu[:,priority_len:]
    
    for i in range(0,GAParam['popSize'],2):
        if (np.random.random() < GAParam['crossRate']):
            # 注：将crossPos转化为整数，因为它回头将用于定位。
            crossPos=int(np.ceil(np.dot(np.random.random(),crossover_len) - 0.1))
            # 注：需要声明a,b为空数组
            a = np.zeros(crossover_len)
            b = np.zeros(crossover_len)
            # 注：matlab中有行信息，但是介于a和b只有1行，所以在此省略，并且定位信息减1
            
            a[0:(crossPos-1)]=copy.deepcopy(crossover_popu[(i-1),0:(crossPos-1)])
            b[0:(crossPos-1)]=copy.deepcopy(crossover_popu[i,0:(crossPos-1)])
            
            k1=crossPos 
            k2=crossPos

            for ii in range(crossover_len):
                # 定位信息各自减1
                if np.logical_not((np.any(crossover_popu[i,ii-1] == a[0:crossPos-1]))):
                    a[k1-1]=copy.deepcopy(crossover_popu[i,ii-1])
                    k1=k1 + 1
                    
                # 定位信息各自减1
                if np.logical_not((np.any(crossover_popu[i-1,ii-1] == b[0:crossPos-1]))):
                    b[k2-1]=copy.deepcopy(crossover_popu[i-1,ii-1])
                    k2=k2 + 1

            crossover_popu[i,:]=copy.deepcopy(a)
            crossover_popu[i + 1,:]=copy.deepcopy(b)
            del a
            del b
            
        Var.popu[:,len(GAParam['Priority_list']):] = copy.deepcopy(crossover_popu)
    
    del i
    #del j
    #del k1
    #del k2
    
    for i in np.arange(0,GAParam['popSize'],2).reshape(-1):
        if (np.random.random() < GAParam['crossRate']):
            crossPos=int(np.round(np.dot(np.random.random(),GAParam['chromoSize'])))
            if np.logical_or(crossPos == 0,crossPos == 1):
                continue
            temp=copy.deepcopy(Var.popuWW[i,range(crossPos-1,GAParam['chromoSize'])])
            Var.popuWW[i,range(crossPos-1,GAParam['chromoSize'])]= copy.deepcopy(Var.popuWW[i + 1,range(crossPos-1,GAParam['chromoSize'])])
            Var.popuWW[i + 1,range(crossPos-1,GAParam['chromoSize'])]=copy.deepcopy(temp)
    del i
    #del temp
    #del crossPos
    
    return Var


def mutation(Var, GAParam):
    #global popu
    #global popuWW
       # if priority task list exist, then no crossover or mutation to that part, 171101
    if GAParam['Priority_condition'] == True:
        mutation_len = GAParam['chromoSize'] - len(GAParam['Priority_list'])
        priority_len = len(GAParam['Priority_list'])
    else:
        mutation_len = GAParam['chromoSize']
        priority_len = 0
        
        
    for i in np.arange(0,GAParam['popSize']).reshape(-1):
        if (np.random.random() < GAParam['mutateRate']):
            randInd=np.random.permutation(range(mutation_len)) + priority_len
            randInd1=[randInd[0],randInd[1]]
            randInd2=[randInd[1],randInd[0]]
            Var.popu[i,randInd1]=copy.deepcopy(Var.popu[i,randInd2])#deepcopy比copy更符合对复制的定义

            #del np.random.random()
            del randInd
            del randInd1
            del randInd2
    del i 
    
    for i in np.arange(0,GAParam['popSize']).reshape(-1):
        if (np.random.random() < GAParam['mutateRate']):
            randInd=np.random.permutation(range(GAParam['chromoSize']))
            randInd1=[randInd[0],randInd[1]]
            randInd2=[randInd[1],randInd[0]]
            Var.popuWW[i,randInd1]=copy.deepcopy(Var.popuWW[i,randInd2])

#           del np.random.random()
            del randInd
            del randInd1
            del randInd2
    del i 
    
    return Var


def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")
        
        

class returnTourOptimization(object):
    def __init__(self, aLifeCount,Data,startLoc,endLoc):
     #def __init__(self, aLifeCount = 100,Data = Data):
        self.Data = Data
        self.initCitys()
        self.lifeCount = aLifeCount
        self.startLoc = startLoc
        self.endLoc = endLoc
        self.ga = GA4TSP(aCrossRate = 0.7, 
                  aMutationRate = 0.02,
                  aLifeCount = self.lifeCount, 
                  aGeneLength = len(self.citys),
                  aMatchFun = self.matchFun())
    def initCitys(self):
        self.citys = []
        #self.citys = self.Data['Task_coor'] 
        self.citys = self.Data 

      #order: a sequence of city(turbine) ID
      
    def distance(self, order):
        '''
        distance = 0.0
            #i从-1到citys#,-1是倒数第一个
        for i in range(-1, len(self.citys) - 1):
                 index1, index2 = order[i], order[i + 1]
                 city1, city2 = self.citys[index1], self.citys[index2]
                 distance += math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
        return distance
        '''
        distance = 0.0
        startLoc = self.startLoc
        endLoc = self.endLoc
        # distance from startLoc
        index1 = order[0]
        city1 = self.citys[index1]
        distance += haversine(startLoc[0], startLoc[1], city1[0], city1[1])
        # distance in the traveling process
        for i in range(0, len(self.citys) - 1):
                 index1, index2 = order[i], order[i + 1]
                 city1, city2 = self.citys[index1], self.citys[index2]
                 distance += haversine(city1[0], city1[1], city2[0], city2[1])
        # distance to the endLoc
        index1 = order[len(self.citys) - 1]
        city1 = self.citys[index1]
        distance += haversine(endLoc[0], endLoc[1], city1[0], city1[1])
        return distance       

      #fitness func, as the target is to find min distance, so take fitness func as 1/distance
    def matchFun(self):
        return lambda life: 1.0 / self.distance(life.gene)


    def run(self, n = 0):
        while n > 0:
            self.ga.next()
            distance = self.distance(self.ga.best.gene)
            #print (("%d : %f") % (self.ga.generation, distance))
            #print self.ga.best.gene
            n -= 1
            #print "After %d iteration，optimal distance is：%f"%(self.ga.generation, distance)
            #print "Turbine order："
            
        for i in self.ga.best.gene:
                  #print self.citys[i][2],
                  return self.citys





#import random  
#from Life import Life  
  
class GA4TSP(object):  
      #GA algorithm fir TSP  
      def __init__(self, aCrossRate, aMutationRate, aLifeCount, aGeneLength, aMatchFun = lambda life : 1):  
            self.crossRate = aCrossRate               #交叉概率  
            self.mutationRate = aMutationRate         #突变概率  
            self.lifeCount = aLifeCount               #种群数量，就是每次我们在多少个城市序列里筛选，这里初始化为100  
            self.geneLength = aGeneLength             #其实就是城市数量  
            self.matchFun = aMatchFun                 #适配函数  
            self.lives = []                           #种群  
            self.best = None                          #保存这一代中最好的个体  
            self.generation = 1                       #一开始的是第一代  
            self.crossCount = 0                       #一开始还没交叉过，所以交叉次数是0  
            self.mutationCount = 0                    #一开始还没变异过，所以变异次数是0  
            self.bounds = 0.0                         #适配值之和，用于选择时计算概率  
            self.initPopulation()                     #初始化种群  
  
  
      def initPopulation(self):  
            # popu initialization  
            self.lives = []  
            for i in range(self.lifeCount):  
                  #gene = [0,1,…… ,self.geneLength-1]  
                  gene = list(range(self.geneLength))   
                  random.shuffle(gene)  
                  life = Life(gene)
                  self.lives.append(life)  
  
  
      def judge(self):     
            self.bounds = 0.0  
            # assume the 1st one is chosen  
            self.best = self.lives[0]  
            for life in self.lives:  
                  life.score = self.matchFun(life)  
                  self.bounds += life.score   
                  if self.best.score < life.score:  
                        self.best = life  
  
  
      def cross(self, parent1, parent2):    
            index1 = random.randint(0, self.geneLength - 1)  
            index2 = random.randint(index1, self.geneLength - 1)  
            tempGene = parent2.gene[index1:index2]                      
            newGene = []  
            p1len = 0  
            for g in parent1.gene:  
                  if p1len == index1:  
                        newGene.extend(tempGene)                                
                        p1len += 1  
                  if g not in tempGene:  
                        newGene.append(g)  
                        p1len += 1  
            self.crossCount += 1  
            return newGene  
  
  
      def  mutation(self, gene):  
            #相当于取得0到self.geneLength - 1之间的一个数，包括0和self.geneLength - 1  
            index1 = random.randint(0, self.geneLength - 1)  
            index2 = random.randint(0, self.geneLength - 1)   
            gene[index1], gene[index2] = gene[index2], gene[index1]   
            self.mutationCount += 1  
            return gene  
  
  
      def getOne(self):    
            r = random.uniform(0, self.bounds)  
            for life in self.lives:  
                  r -= life.score  
                  if r <= 0:  
                        return life  
  
            raise Exception("error selection", self.bounds)  
  
  
      def newChild(self):    
            parent1 = self.getOne()  
            rate = random.random()  
  
            #按概率交叉  
            if rate < self.crossRate:  
                  #交叉  
                  parent2 = self.getOne()  
                  gene = self.cross(parent1, parent2)  
            else:  
                  gene = parent1.gene  
  
            #按概率突变  
            rate = random.random()  
            if rate < self.mutationRate:  
                  gene = self.mutation(gene)  
  
            return Life(gene)  
  
  
      def next(self):  # next generation
            self.judge()#评估，计算每一个个体的适配值  
            newLives = []  
            newLives.append(self.best)#把最好的个体加入下一代  
            while len(newLives) < self.lifeCount:  
                  newLives.append(self.newChild())  
            self.lives = newLives  
            self.generation += 1 
            
SCORE_NONE = -1
class Life(object): # individual class
      def __init__(self, aGene = None):
            self.gene = aGene
            self.score = SCORE_NONE
            
def distanceMatConvert(task_coor, shoreStation_coor):
    task_coor_ext = np.concatenate((shoreStation_coor, task_coor[:,:2]), axis=0)
    distanceMat = np.zeros((len(task_coor_ext), len(task_coor_ext)))
    
    for i in range(len(task_coor_ext)):
        for j in range(len(task_coor_ext)):
            distanceMat[i][j] = haversine(task_coor_ext[i][1],
                       task_coor_ext[i][0],
                       task_coor_ext[j][1],
                       task_coor_ext[j][0])
            
    return distanceMat
   
         
#from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r






################################################################
# Pre-processing
################################################################
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import pandas as pd
# import MySQLdb
import pymysql as MySQLdb # pymysql is an alternative 
# from astropy import units as u
from astropy.coordinates import Angle

def SchedulingPreprocessing(main_sch_id,host,user,passwd,Data_base):
#    #Data_base='xa0002' # for PC verson
#    #conn = MySQLdb.connect(host='10.84.1.111', user='root', passwd='123456', db=Data_base, charset="utf8")
#    #conn = MySQLdb.connect(host='localhost', user='root', passwd='123456', db=Data_base, charset="utf8")
#    # cur = conn.cursor()
    conn = MySQLdb.connect(host=host, user=user, passwd=passwd, db=Data_base, charset="utf8")    
    # available boats
    #sql1 = "SELECT * FROM boatinput"
    sql1 =("SELECT * FROM boatinput WHERE Main_sch_Id LIKE '%s'" %(main_sch_id+'%'))
    boats_df = pd.read_sql(sql=sql1, con=conn)
    #Ship_ID = boats_df['Ship_ID']
    #MB = len(Ship_ID)
    
    del sql1
    
    
    # workers # update 201705
    sql1 = ("SELECT * FROM workerinput WHERE Main_sch_Id LIKE '%s'" %(main_sch_id+'%'))
    workers_df = pd.read_sql(sql=sql1, con=conn)
    del sql1   
    
    
    
    # turbine coordinate
    sql1 = "SELECT * FROM farms"
    turbine_coor_df = pd.read_sql(sql=sql1, con=conn)   
    
    del sql1
    
    # main_sch_id = 'XA0002_2017-05-09'
    sql1 = ("SELECT * FROM processinput WHERE Main_sch_Id LIKE '%s'" %(main_sch_id+'%'))
    # sql1 = "SELECT * FROM processinput WHERE Main_sch_Id LIKE 'XA0002_2017-05-09%'"
    task_df = pd.read_sql(sql=sql1, con=conn)
    # task_df = task_df.drop(task_df.index[range(31)])
    task_df = task_df.reset_index(drop=True)
    del sql1
    
    '''
    input data organization 
    '''
    task_df["task_ID"] = task_df["Turbine_ID"].map(str) +'_' + task_df["OM_ID"]
    task_ID = pd.unique(task_df[['task_ID']].values.ravel())
    #print "task_ID",task_ID 
    task_workerType = pd.unique(task_df[['Worker_Type']].values.ravel())
    
    turbine_ID = [None] * len(task_ID)
    #print "turbine_ID",turbine_ID
    OM_Time = [None] * len(task_ID)
    # worker_Num = np.zeros((3,len(task_ID)), dtype = np.int) # need further modification
    worker_Num = np.zeros((len(task_workerType),len(task_ID)), dtype = np.int)

    
    # unicode of different skills
    '''
    u'\u673a\u68b0' - '机械'
    u'\u7535\u6c14' - '电气'
    u'\u7b2c\u4e09\u65b9' - '第三方'
    '''
    # modification - 201705
    # read the skill unicode from table

        
    for i in range(len(task_ID)):
        idx = np.where(task_df[['task_ID']] == task_ID[i])
        #print "idx",idx
        idxi = idx[0][0]
        #print "idxi",idxi
        # turbine ID & OM_Time finding
        turbine_ID[i] = task_df['Turbine_ID'][idxi]
        OM_Time[i] = task_df['OM_Time'][idxi]
        del idx, idxi
        
        # unicode of different skills
        '''
        u'\u673a\u68b0' - '机械'
        u'\u7535\u6c14' - '电气'
        u'\u7b2c\u4e09\u65b9' - '第三方'
        '''
        for j in range(len(task_workerType)):
            idx1 = np.where(task_df['task_ID'] == task_ID[i]) 
            #print "idx1",idx1
            idx2 = np.where(task_df['Worker_Type'] == task_workerType[j])
            #print "idx2",idx2
            idx = np.intersect1d(idx1, idx2)#求交集
            if idx.size != 0:
                worker_Num[j][i] = task_df['Worker_Num'][idx[0]]
            
            del idx1, idx2, idx
      
        
    # calculate the num of different type of input workers 
    # following lines are to guarantee the uniqueness of each worker
    workers_ID = pd.unique(workers_df[['Workers_ID']].values.ravel())
    workers_ID_idx = [None] * len(workers_ID)
    for i in range(len(workers_ID)):
        idx = np.where(workers_df[['Workers_ID']] == workers_ID[i])
        idxi = idx[0][0]
        workers_ID_idx[i] = idxi
        del idx, idxi
    workers_df = workers_df[workers_df.index.isin(workers_ID_idx)] 
    workers_df = workers_df.reset_index(drop=True)
    
    # worker organized by skills
    worker_list_by_skills = [None] * len(task_workerType)
    worker_list = workers_df['Workers_ID'].tolist()
    for j in range(len(task_workerType)):
        idx = np.where(workers_df['Worker_Type'] == task_workerType[j])
        worker_list_by_skills[j] = [worker_list[i] for i in idx[0]]
        del idx
        

    
    # task coordinate
    task_coor = np.zeros((len(task_ID),3))
    for i in range(len(task_ID)):
        idx = np.where(turbine_coor_df['Turbine_ID'] == turbine_ID[i]) 
        idxi = idx[0][0]
        task_coor[i][0] = Angle(turbine_coor_df['WGS84Latitude'][idxi]).degree
        task_coor[i][1] = Angle(turbine_coor_df['WGS84Longitude'][idxi]).degree
        task_coor[i][2] = turbine_ID[i]
    
    
    
    shoreStation_coor = np.array([[30.847838889,121.8448194444]]) # need further modification
    
    distanceMat = np.zeros((len(task_coor)+1, len(task_coor)+1))
    distanceMat = distanceMatConvert(task_coor, shoreStation_coor)           
    #distanceMat = distanceMat/20/1.852*(60.0/15.0)
    distanceMat = distanceMat
    
    Data = dict()
    Data['Task_coor'] = task_coor
    Data['ShoreStation_coor'] = shoreStation_coor
    Data['Distance'] = distanceMat
    # Data['CostTransport'] = distanceMat*125
    Data['TTR'] = np.asarray(OM_Time)/15
    #Data['TTR'] = np.asarray(OM_Time)/15 + 1
    # modified by Liwei, 
    
    '''
    PL_0 and PL_1 载入
    '''
    import scipy.io
    import os
    # 读取数据
    #datapath = os.path.abspath('') +'//Data0615.mat' 
    datapath = 'D://model//Data0615.mat'
    raw_data = scipy.io.loadmat(datapath, struct_as_record=False, squeeze_me=True)
    # 将数据从raw_data中提取并转化为python适用格式
    DataRaw = raw_data['Data'].__dict__
    
    Data['PL0'] = DataRaw['PL0']
    Data['PL1'] = DataRaw['PL1']
    
    tbID=Data['Task_coor'][:,2]
    seq=np.zeros(len(tbID))

    for i in range(len(tbID)):
        for j in range(len(tbID)):
            if(tbID[i]==DataRaw['PL0'][0,j]):
                seq[i]=j
    for i in range(len(tbID)):
        Data['PL0'][:,i]=DataRaw['PL0'][:,int(seq[i])]
        Data['PL1'][:,i]=DataRaw['PL1'][:,int(seq[i])]    
            
    Data['PL0']=np.delete(Data['PL0'],0,axis=0)
    Data['PL1']=np.delete(Data['PL1'],0,axis=0)
    
    del datapath, raw_data, DataRaw
    
    
    '''
    general params
    '''
    Params = dict()
    # boats
    boats_ID = pd.unique(boats_df[['Ship_ID']].values.ravel())
    #boats_ID = boats_ID[[1,2]] # 0808, trival
    Params['MB'] = len(boats_ID)
    Params['boats_ID'] = boats_ID
    
    # workers
    '''
    Worker_Type unicode:
        u'\u673a\u68b0' - '机械'
        u'\u7535\u6c14' - '电气'
        u'\u7b2c\u4e09\u65b9' - '第三方'
    '''
    NP = np.zeros(len(task_workerType), dtype = np.int)
    #NP[0] = np.sum(workers_df['Worker_Type'] == u'\u673a\u68b0')
    #NP[1] = np.sum(workers_df['Worker_Type'] == u'\u7535\u6c14')
    #NP[2] = np.sum(workers_df['Worker_Type'] == u'\u7b2c\u4e09\u65b9')
    
    for j in range(len(task_workerType)):
        NP[j] = np.sum(workers_df['Worker_Type'] == task_workerType[j])
    
    Params['NP'] = NP
    Params['taskLen'] = len(task_ID)
    Params['TaskWorkers'] = worker_Num
    
    Params['worker_list_by_skills'] = worker_list_by_skills
    Params['task_workerType'] = task_workerType
    Params['task_df'] = task_df
    return Data, Params

'''
def distanceMatConvert(task_coor, shoreStation_coor):
    task_coor_ext = np.concatenate((shoreStation_coor, task_coor[:,:2]), axis=0)
    distanceMat = np.zeros((len(task_coor_ext), len(task_coor_ext)))
    
    for i in range(len(task_coor_ext)):
        for j in range(len(task_coor_ext)):
            distanceMat[i][j] = haversine(task_coor_ext[i][1],
                       task_coor_ext[i][0],
                       task_coor_ext[j][1],
                       task_coor_ext[j][0])
            
    return distanceMat
'''
'''
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
    
'''

################################################################
# Post-processing
################################################################
# post-processing based on modeling results


# import datetime
from sqlalchemy import create_engine#把关系数据库的表结构映射到对象上

def engineer_wind_to_id(task_i_wind, worker_list_by_skills):
    task_i_engineer = [];
    for expertise_i in range(len(task_i_wind)):
         taski_expertisei_id = copy.deepcopy(task_i_wind[expertise_i])
         if len(taski_expertisei_id) > 0:
             for worker in taski_expertisei_id:
                 task_i_engineer.append(worker_list_by_skills[expertise_i][worker])
    return task_i_engineer

# boat path
def BInd_to_boat_id(boat_id_array, BInd, maintenance_ord_array):
    # boat_id_array = Params['boats_ID']
    # BInd = results['bestBInd']

    task_i_boat_id = [None] * len(maintenance_ord_array)
    for i, item in enumerate(maintenance_ord_array.ravel()):
        task_i_boat_id[i] = copy.deepcopy(boat_id_array[BInd[item]])
   
    return task_i_boat_id

def TurbineAndWorkers(task_i_boat_id, turbine_id, pick_turbine, bestSequence, 
                           task_list, task_engineer_str):
    # This is a function to extract Turbine series and associate workers
    # start_turbine: store the task part of boat routes
    start_turbine = [None] * len(task_i_boat_id)
    turbine_temp = []
    pickup_turbine_arrive = []  # Turbines where workers needs being picked
    pickup_turbine_start = []   # Turbines from which boats start to pick workers
    pickup_workers = []  # Workers need picking
    boat_id = []  # boat id 'has been seen' set
    for i in range(len((task_i_boat_id))):
        if task_i_boat_id[i] in boat_id: # if curent boat id has been seen before
            for index in range(i)[::-1]: # find the previous turbine the boat had reached
                if task_i_boat_id[i] == task_i_boat_id[index]:
                    turbine_temp = turbine_id.tolist()[index] # store the turbine in temp
                    break
        else: # if current boat id hasn't been seen
            turbine_temp = 'Dock' # it comes from Station. store Station into temp
            boat_id.append(task_i_boat_id[i]) # add the boat id into 'has been seen' set
        if len(pick_turbine[i]) != 0: # if current pick_turbine has element
            # the boat nees to go to another turbine to pick up workers
            # start_turbine is the pick-up turbine
            for j in range(len(pick_turbine[i])):
                turbineId = copy.deepcopy(bestSequence[pick_turbine[i][j]]) 
                if j == 0:
                    pickup_turbine_start.append(turbine_temp)
                    pickup_turbine_arrive.append(turbine_id[turbineId])
                    pickup_workers.append(task_engineer_str[np.where(task_list==turbineId)[0][0]])
                    if j == len(pick_turbine[i])-1:
                        start_turbine[i] = copy.deepcopy(turbine_id[turbineId])
                #turbineId = bestSequence[pick_turbine[i][0]]
                if j > 0:
                    pickup_turbine_start.append(pickup_turbine_arrive[-1])
                    pickup_turbine_arrive.append(turbine_id[turbineId])
                    pickup_workers.append(task_engineer_str[np.where(task_list==turbineId)[0][0]])
                    if j == len(pick_turbine[i])-1:
                        start_turbine[i] = copy.deepcopy(turbine_id[turbineId])
                # for pickup part of boat routes:
                # 'pickup turbine arrive' is the turbine that needs workers for task
                # 'pickup turbine start' is the turbine
                # 'pickup workers' is the workers on the pick-up turbine
        else: # if current pick_turbine has no element
            start_turbine[i] = copy.deepcopy(turbine_temp) # temp is the start_turbine, no 'pick-up' happens
    return start_turbine, pickup_turbine_arrive, pickup_turbine_start, pickup_workers

def StartArriveTime(start_turbine, turbine_id, pickup_turbine_arrive, pickup_turbine_start, 
                    start_time, pick_time, transtime, task_i_boat_id):
    # This is a function to extract all time information
    arrive_time = [None] * len(start_turbine) # arrive time for task part of boat routes
    # time info for pick-up part of boat routes
    #print "pick_time",pick_time
    pickup_time_start = []
    pickup_boat = []
    pickup_time_arrive = [None]*len(pickup_turbine_start)
    for i in range(len(start_turbine)):
        if type (start_turbine[i]) is str: # if start_turbine is 'Station'
            start_index = 0 # it starts from station, '0' on the transport-time matrix
        else:
            # otherwise find the index of the start turbine on the transport-time matrix
            start_index = turbine_id[turbine_id==start_turbine[i]].index[0]+1
        # find the index of the arrive turbine on the transport time matrix
        arrive_index = turbine_id[turbine_id==turbine_id.tolist()[i]].index[0]+1
        # arrive time is the start time plus transport time
        arrive_time[i] = np.ceil(start_time[i]+transtime[start_index, arrive_index])
    # for pick-up part of boat routes
    #for i in range(len(pick_time)):
        #print "type(pick_time[i])",type(pick_time[i])
    for i in range(len(pick_time)):
        if type(pick_time[i]) == type(np.int32(1)) or type(pick_time[i]) == type(np.int64(1)): # if corresponding pick-up time exits
        #linux int64!!
            # add the pick-up time to pick-up start time list
            pickup_time_start.append(pick_time[i])
            pickup_boat.append(task_i_boat_id[i]) # add the corresponding boat id
        if type(pick_time[i]) == type(np.array([1])):
            for j in range(len(pick_time[i])):
                pickup_time_start.append(pick_time[i][j])
                pickup_boat.append(task_i_boat_id[i])
    #print "pickup_time_start",pickup_time_start
    for i in range(len(pickup_turbine_start)):
        #print "len(pickup_turbine_start)",len(pickup_turbine_start)
        # for every pick-up turbine and time, 
        # find the start and arrive index in the transport time matrix
        start_index = turbine_id[turbine_id==pickup_turbine_start[i]].index[0]+1
        arrive_index = turbine_id[turbine_id==pickup_turbine_arrive[i]].index[0]+1
        # pick-up time is start pick up time plus transport time
        #print "pickup_time_arrive[i]",pickup_time_arrive[i]
        pickup_time_arrive[i] = np.ceil(pickup_time_start[i]+transtime[start_index, arrive_index])
        #print "pickup_time_arrive[i]",pickup_time_arrive[i]
        
    return arrive_time, pickup_time_start, pickup_time_arrive, pickup_boat

def ReturnTour(return_turbine, turbine_id, task_engineer_str, boat_id):
    # This is a funtion to extract turbine, time and worker info for return part of boat routes
    # Use BInd_to_boat_id function to extract all boat ids
    boat = BInd_to_boat_id(boat_id, list(range(len(return_turbine))), np.arange(len(return_turbine)))
    turbine_id = turbine_id.tolist()
    return_turbine_start = []
    return_turbine_arrive = []
    return_worker = []
    return_boat = []
    for i in range(len(boat)): # for every boat
        for j in range(len(return_turbine[i])):
            return_boat.append(boat[i]) # record the boat id
            return_turbine_start.append(return_turbine[i][j])  # record the return-start turbine
            # record the workers on the turbine. 
            return_worker.append(task_engineer_str[turbine_id.index(return_turbine[i][j])])
            if j == len(return_turbine[i])-1: # if the turbine is the last one
                return_turbine_arrive.append('Station') # arrive info is Station
            else: # otherwise arrive info is the next turbine
                return_turbine_arrive.append(return_turbine[i][j+1])
    # return tour has no disembarking workers except the last (Station), all works disembark
    return_disembark = [[]for _ in range(len(return_worker))]
    return_disembark[-1] = 'All Disembarking'
    return return_turbine_start, return_turbine_arrive, return_worker, return_boat, return_disembark
    
def Change2Str(thelist): 
    # This is just a function to convert all info into string
    # Unify all data types before writing it into MySQL
    for i in range(len(thelist)):
        if thelist[i] == [] or thelist[i] == [None]:
            thelist[i] = '' # if empty set to ''
        if type(thelist[i]) is str:
            continue
        else:
              thelist[i] = str(thelist[i])  
    return thelist

def changeToDatetime(lists,year,month,day):
    import datetime
    initial = datetime.datetime(year, month, day)
    result = [[] for _ in range(len(lists))]
    for _ in range(len(lists)):
        result[_] = initial + int(lists[_]) * datetime.timedelta(minutes=15)
    return result


def schedulingPostProcessing(Data, Params, results, main_sch_id,host,user,passwd,Data_base):      
    #engine = create_engine('mysql+pymysql://root:123456@10.84.1.111/xa0002') #初始化数据库连接
    engine = create_engine('mysql+pymysql://'+user+':'+passwd+'@'+host+'/'+Data_base)      
    #engine = create_engine('mysql+pymysql://root:123456@10.84.1.111/xa0002') #初始化数据库连接
    #engine = create_engine('mysql+pymysql://root:123456@10.84.1.154/xa0002')
    #conn = MySQLdb.connect(host='10.84.1.154', user='root', passwd='123456', db=Data_base, charset="utf8")
    # get the maintenance list
    best_result = results
    maintenance_ord_array = np.argwhere(best_result['bestSTime'])#得到索引#去掉开始时间为0，也就是未安排的任务
#    print "maintenance_ord_array",maintenance_ord_array
    # task list
    task_list = best_result['bestSequence'][maintenance_ord_array]
#    print "task_list",task_list
    # task_ttr
    task_downtime = Data['TTR'][task_list]
#    print "task_downtime",task_downtime
    
    #去掉重复风机和任务
    df=Params['task_df'].loc[:,['Turbine_ID','OM_ID']]
    df=pd.DataFrame.drop_duplicates(df)
#    print "df",df
    df=df.reset_index(drop = True)
#    print "df",df
    # turbine_id
    turbine_id = df['Turbine_ID'][task_list.ravel()]#变成行向量
#    print "turbine_id",turbine_id
    # om_id
    om_id = df['OM_ID'][task_list.ravel()]
#    print "om_id",om_id


    # task start time
    task_start_time = best_result['bestSTime'][maintenance_ord_array]
    task_end_time = task_start_time + task_downtime
#    print "task_start_time",task_start_time
#    print "task_end_time",task_end_time
    # task end time
    
 
    # engineers at each task
    task_wind_individual = list(zip(*best_result['bestWInd']))
    #print "task_wind_individual",task_wind_individual
    
    pick_time = [[] for _ in range(len(maintenance_ord_array.ravel()))]
    pick_turbine = [[] for _ in range(len(maintenance_ord_array.ravel()))]
    task_engineer = [[] for _ in range(len(maintenance_ord_array.ravel()))]
    for count, task_i_ord in enumerate(maintenance_ord_array.ravel()):
        task_i_wind = copy.deepcopy(task_wind_individual[task_i_ord])
        pick_time[count] = best_result['bestPickupTime'][task_i_ord]
        pick_turbine[count] = best_result['bestPickupTurbineOrd'][task_i_ord]
        task_engineer[count] = engineer_wind_to_id(task_i_wind, Params['worker_list_by_skills'])
#        print "task_i_wind",task_i_wind
#        print "Params['worker_list_by_skills']",Params['worker_list_by_skills']
#        print "task_engineer[count]",task_engineer[count]
#        print "count",count

    
    task_engineer_str = [[] for _ in range(len(maintenance_ord_array.ravel()))]
    #print task_engineer
    for count, item in enumerate(task_engineer):
        task_engineer_str[count] = '/'.join(item)
    #print "task_engineer_str!!!",task_engineer_str
    import datetime
    i = datetime.datetime.now()
    year=i.year
    month=i.month
    day=i.day                 
 
    table_task = pd.DataFrame({'Main_sch_id':main_sch_id, 
                               'Turbine_id': turbine_id.tolist(),
                               #'task_id': task_list[:,0],
                               'OM_ID': om_id.tolist(),
                               'Workers_ID': task_engineer_str,
                               'startT': changeToDatetime(task_start_time[:,0],year,month,day), 
                               'endT': changeToDatetime(task_end_time[:,0],year,month,day) 
                               } )                              
 
    table_task.to_sql('schedule', engine, if_exists='append',index=False)  # or 'append'
 
    # task cost
    #engine = create_engine('mysql+pymysql://root:123456@127.0.0.1/shanghaielectric0808')       
    
    table_cost = pd.DataFrame({'Main_sch_id':main_sch_id,
                               'total_cost': best_result['objValue'], 
                               'production_loss': best_result['bestTotalCostPL'], 
                               'OT_cost': best_result['bestTotalCostOT'], 
                               'boat_cost': best_result['bestTotalCostBoat']}, index=[0])
    table_cost.to_sql('schedulecost', engine, if_exists='append',index=False)
    
    # task cost breakdown
    task_cost_total = best_result['bestIndividualCost'][maintenance_ord_array]
    task_cost_PL = best_result['bestIndividualCostPL'][maintenance_ord_array]
    task_cost_OT = best_result['bestIndividualCostOT'][maintenance_ord_array]
    task_cost_Boat = best_result['bestIndividualCostBoat'][maintenance_ord_array]
    
    #engine = create_engine('mysql+pymysql://root:123456@127.0.0.1/shanghaielectric0808')       
    
    table_cost_detail = pd.DataFrame({#'task_id': task_list[:,0],
                                         'Main_sch_id':main_sch_id,
                                         'Turbine_ID': turbine_id.tolist(),
                                         'OM_ID': om_id.tolist(),
                                         'TotalCost': task_cost_total[:,0], 
                                         'powerlost': task_cost_PL[:,0], 
                                         'Overtimecost': task_cost_OT[:,0], 
                                         'shipcost': task_cost_Boat[:,0]})
    table_cost_detail.to_sql('schcostlist', engine, if_exists='append',index=False)
    
    
    # downtime stats
    #engine = create_engine('mysql+pymysql://root:123456@127.0.0.1/shanghaielectric0808')       
    table_task_downtime = pd.DataFrame({'task_id': task_list[:,0],
                                        'Main_sch_id':main_sch_id,
                                        'turbine_id': turbine_id.tolist(),
                                        'om_id': om_id.tolist(),
                                        'task_downtime(min)': task_downtime[:,0]})
    table_task_downtime.to_sql('table_task_downtime', engine, if_exists='append')
    
    
    # boat id for each task
    task_i_boat_id = BInd_to_boat_id(Params['boats_ID'], results['bestBInd'], maintenance_ord_array)
    # the corresponding task id will be the arrive task id
    #arrived_turbine_id = Params['task_df']['Turbine_ID'][task_list.ravel()]
    # om_id
    #arrived_om_id = Params['task_df']['OM_ID'][task_list.ravel()]
    
    # boat_departure_time
    boat_departure_time = best_result['bestDTime'][maintenance_ord_array]
    
    # dock time
    dock_time = best_result['bestWW'][maintenance_ord_array]
    
    # start turbine and workers
    start_turbine, pickup_turbine_arrive, pickup_turbine_start, pickup_workers = TurbineAndWorkers(task_i_boat_id, 
                            turbine_id, pick_turbine, best_result['bestSequence'], task_list, task_engineer_str)
    
    # start time and arrive time
    arrive_time, pickup_time_start, pickup_time_arrive, pickup_boat= StartArriveTime(start_turbine, turbine_id, 
                                 pickup_turbine_arrive, pickup_turbine_start, boat_departure_time[:,0], 
                                 pick_time, Data['TransportTime'], task_i_boat_id)
    
    # Return Journey
    #return_turbine_start, return_turbine_arrive, return_worker, return_boat, return_disembark = ReturnTour(best_result['returnTour'], turbine_id, task_engineer_str, Params['boats_ID'])
    
    """
    All boat routes information includes three parts: Task part, pick-up part and return part.
    Task part & pick-up part info is extracted from function TurbineAndWorkers and StartArriveTime
    Return part info is extracted from function ReturnTour. No time info for return part.
    """
    #ship_id = task_i_boat_id + pickup_boat + return_boat
    ship_id = task_i_boat_id + pickup_boat
    #start_time = boat_departure_time[:,0].tolist() + pickup_time_start + [[]for _ in range(len(return_turbine_start))]
    start_time = boat_departure_time[:,0].tolist() + pickup_time_start
    #start_turbine = start_turbine + pickup_turbine_start + return_turbine_start
    start_turbine = start_turbine + pickup_turbine_start
    #arrive_time = arrive_time + pickup_time_arrive + [[]for _ in range(len(return_turbine_arrive))]
    arrive_time = arrive_time + pickup_time_arrive
    #arrive_turbine = turbine_id.tolist() + pickup_turbine_arrive + return_turbine_arrive
    arrive_turbine = turbine_id.tolist() + pickup_turbine_arrive
    #dock_time = dock_time[:,0].tolist() + [0.0 for _ in range(len(pickup_time_arrive)+len(return_turbine_arrive))]
    dock_time = dock_time[:,0].tolist() + [0.0 for _ in range(len(pickup_time_arrive))]
    #disembark_workers = task_engineer_str + [[]for _ in range(len(pickup_turbine_arrive))] + return_disembark
    disembark_workers = task_engineer_str + [[]for _ in range(len(pickup_turbine_arrive))] 
    #embark_workers = [[]for _ in range(len(task_i_boat_id))] + pickup_workers + return_worker
    embark_workers = [[]for _ in range(len(task_i_boat_id))] + pickup_workers
    # Convert info into string type before writing it into MySQL
#    start_time = Change2Str(start_time)
    start_turbine = Change2Str(start_turbine)
#    arrive_time = Change2Str(arrive_time)
    arrive_turbine = Change2Str(arrive_turbine)
#    dock_time = Change2Str(dock_time)
    disembark_workers = Change2Str(disembark_workers)
    embark_workers = Change2Str(embark_workers)
    
    # temp = [[] for _ in range(len(ship_id))] # will be removed later
    #engine = create_engine('mysql+pymysql://root:123456@127.0.0.1/shanghaielectric0808')     
    for _ in range(len(dock_time)):
        dock_time[_]=dock_time[_]*15
        
    table_boat_route = pd.DataFrame({'Main_sch_id':main_sch_id,
                                     'ship_id': ship_id,
                                      #
                                     'startturbine': start_turbine, #
                                     'starttime': changeToDatetime(start_time,year,month,day),
                                     'arriveturbine': arrive_turbine,
                                     'arrivetime': changeToDatetime(arrive_time,year,month,day), #
                                      # 
                                     'dock_time': dock_time,
                                     'disembark_workers': disembark_workers,
                                     'embark_workers': embark_workers})
    table_boat_route.to_sql('boatroute', engine, if_exists='append',index=False)
    
    return 
    
