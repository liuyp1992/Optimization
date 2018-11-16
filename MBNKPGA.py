#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import itertools as itt
import random
from math import radians, cos, sin, asin, sqrt
import copy


def MBNKPGA(GAParam, Data):#遗传算法模块
    '''
    遗传算法设置:
        popSize: 每一代的种群大小
        chromoSize: 染色体长度
        generationSize:迭代次数
        crossRate: 交叉率
        mutateRate: 变异率
        elitism: 是否使用精英策略
    '''
    GAParam['elitism'] = True#使用精英策略
    GAParam['popSize'] = GAParam['taskLen']*6#种群大小
    GAParam['chromoSize'] = GAParam['taskLen']#任务长度
    GAParam['generationSize'] = 100#迭代次数
    Var = Variables(GAParam)
    result = dict()
    result = GeneticAlgorithm(Var, GAParam,Data)
    return result

class Variables(object):
    def __init__(self, GAParam):
        self.fitness_value = np.zeros(GAParam['generationSize']) 
        self.fitness_avg = np.zeros(GAParam['generationSize'])
        self.fitness_table = np.zeros(GAParam['popSize'])
        self.best_fitness = 0.
        self.best_individual = [[] for _ in range(GAParam['chromoSize'])]
        self.best_generation = 0
        self.bestStartTime = np.zeros(GAParam['chromoSize'])
        self.bestDepartTime = np.zeros(GAParam['chromoSize'])
        self.best_BInd = np.zeros(GAParam['MB'],)
        self.best_WInd = [[] for _ in range(GAParam['chromoSize'])]
        self.best_BInventFinal = [[] for _ in range(GAParam['MB'])] 
        self.best_BEndFinal = [[] for _ in range(GAParam['MB'])]  
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
        
        self.popu, self.popuWW = self.initialize(GAParam) 
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
        self.BInventFinal = [[[] for _ in range(GAParam['MB'])] for _ in range(GAParam['popSize'])] 
        self.BEndFinal = [[[] for _ in range(GAParam['MB'])] for _ in range(GAParam['popSize'])] 
        self.pickupTurbineOrdAll = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
        self.pickupTimeAll = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
        self.pickupBoatAll = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(GAParam['popSize'])]
    
    def initialize(self, GAParam):#初始化
        popSize = GAParam['popSize']
        chromoSize = GAParam['chromoSize']
        popu = np.zeros((popSize,chromoSize), dtype = int)
        if GAParam['Priority_condition'] == True:#指定任务优先处理
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



def GeneticAlgorithm(Var, GAParam, Data):#基于遗传算法求解最优任务序列
    
    allbestresult = np.zeros(shape=[GAParam['generationSize'],])
    for G in range(GAParam['generationSize']):
        Var = fitnessMBNKP(Var, GAParam,Data) #计算适应度
        Var = rank(Var, GAParam, G)  #排序
        Var = selection(Var, GAParam) #选择
        Var = crossover(Var, GAParam) #交叉
        Var = mutation(Var, GAParam) #变异
        
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
    results['bestBInvent'] = Var.best_BInventFinal 
    results['bestBEnd'] = Var.best_BEndFinal 
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
    

def fitnessMBNKP(Var, GAParam,Data):#计算适应度
    for ps in range(GAParam['popSize']):
        
        #参数初始化
        # t, tB, costIndividualTask, costIndividualPL, costIndividualOT, costIndividualBoat
        t = np.zeros(GAParam['chromoSize'], dtype = np.int)#任务开始时间
        tB = np.zeros(GAParam['chromoSize'], dtype = np.int)#船只开始时间
        BIndindi = [[] for _ in range(GAParam['chromoSize'])]
        costIndividualTask = np.zeros(GAParam['chromoSize'])
        costIndividualPL = np.zeros(GAParam['chromoSize'])
        costIndividualOT = np.zeros(GAParam['chromoSize'])
        costIndividualBoat = np.zeros(GAParam['chromoSize'])
        
        # timeIni = 36
        timeIni = GAParam['timeIni']
        
        # 工人具有PTypes种不同技能(比如: 机械 & 电气)
        PTypes = GAParam['PType']
        
        TaskWorkers = GAParam['TaskWorkers'][:, Var.popu[ps]]
        pickupTurbineOrd = [[] for _ in range(GAParam['chromoSize'])] 
        pickupTime = [[] for _ in range(GAParam['chromoSize'])]
        
        
        # 任务初始化
        # 进行前MB个任务的分配工作
        WIndindi = [[[] for _ in range(GAParam['chromoSize'])] for _ in range(PTypes)]
        
        for j in range(GAParam['MB']):
            t[j] = timeIni + Var.popuWW[ps,j] + np.ceil(Data['TransportTime'][0,Var.popu[ps,j]+1])
            tB[j] = timeIni + Var.popuWW[ps,j]
            
            costIndividualPL[j] = sum(Data['PL0'][range(0,int(t[j])),Var.popu[ps,j]])*GAParam['ElecPrice'] + sum(Data['PL1'][range(int(t[j]),int(t[j]+Data['TTR'][Var.popu[ps,j]])),Var.popu[ps,j]])*GAParam['ElecPrice']
            costIndividualOT[j] = 0
            costIndividualBoat[j] = copy.deepcopy(Data['CostTransport'][0,Var.popu[ps,j]+1])
            
            costIndividualTask[j] = costIndividualPL[j] + costIndividualOT[j] + costIndividualBoat[j]

            for i in range(PTypes):
                if j == 0:
                    WIndindi[i][j] = list(range(TaskWorkers[i,j]))#将所需人数转化为人员索引，即给人员进行编号
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
            # 分配余下在船上的工人
            WIndindiIni = [[]  for _ in range(PTypes)]
            for pp in range(PTypes):
                WIndindiIni[pp] = copy.deepcopy(WIndindi[pp][:GAParam['MB']])
    
            BInvent = copy.deepcopy(BInventInitial(WIndindiIni,GAParam['NP']))
               
            # 记录在船上的工人
            WonB = WorkersonBoats(BInvent)
            
            # 任务分配
            for j in range(GAParam['MB'],GAParam['chromoSize']):
                BAvail = np.zeros(GAParam['MB'])
                # 找到是否有船只上的人员可以直接满足任务j的需求
                for jj in range(GAParam['MB']):
                    BAvail[jj] = sum(np.sum(BInvent[jj],axis = 1) >= TaskWorkers[:,j])#axis=1表示行，axis=0表示列
                    #对某个矩阵（某艘船上）的行求和，需要大于该位置出现的任务的人员需求，得出有几项技能可以满足需求
                BTemp = np.where(BAvail == PTypes)#所有技能类型都能满足的船只
                BTemp = np.array(BTemp).ravel()#把一个矩阵变成行向量，原来是array，和reshape（-1）功能相似
                if BTemp.size:#找到所有可以满足条件的船当中完成时间最短的
                    # find out the optimal boat can cause the min cost of task j
                    BPosArray = [[] for _ in range(len(BTemp))]
                    tTempArray = [[] for _ in range(len(BTemp))]
                    for ii in range(len(BTemp)):
                        BPosArray[ii] = max([item for item in range(len(BIndindi)) if BIndindi[item] == BTemp[ii]])
                        #找到符合要求的船只最后所在的风机位置，表示风机位置数组
                        tTempArray[ii] = copy.deepcopy(t[BPosArray[ii]] + np.ceil(Data['TransportTime'][Var.popu[ps,BPosArray[ii]]+1,Var.popu[ps,j]+1]))
                    
                    ind = np.argmin(tTempArray)#找到最早结束的船只
                    tTemp = tTempArray[ind] + Var.popuWW[ps,j] + np.ceil(GAParam['DockTime'])
                    tBTemp = t[BPosArray[ind]] + Var.popuWW[ps,j] + np.ceil(GAParam['DockTime'])
                    
                    #计算各项成本
                    if (int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['RegularEndtime']+1: 
                        t[j] = copy.deepcopy(tTemp)
                        tB[j] = copy.deepcopy(tBTemp)
                        
                        costIndividualPL[j] = sum(Data['PL0'][range(t[j]),Var.popu[ps,j]]) * GAParam['ElecPrice'] + sum(Data['PL1'][range(t[j],int(t[j]+Data['TTR'][Var.popu[ps,j]])),Var.popu[ps,j]])*GAParam['ElecPrice']
                        #风功率损失成本
                        costIndividualOT[j] = 0#加班成本
                        costIndividualBoat[j] = Data['CostTransport'][Var.popu[ps,BPosArray[ind]]+1,Var.popu[ps,j]+1]#船只成本
                        costIndividualTask[j] = costIndividualPL[j] + costIndividualOT[j] + costIndividualBoat[j]#单个任务的总成本

                    elif (((int(tTemp) + Data['TTR'][Var.popu[ps,j]]) > GAParam['RegularEndtime']) & ((int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['OTEndtime']+1)):
                        t[j] = tTemp
                        tB[j] = tBTemp
                        
                        costIndividualPL[j] = sum(Data['PL0'][range(t[j]),Var.popu[ps,j]]) * GAParam['ElecPrice'] + sum(Data['PL1'][range(t[j]-1,int(t[j]+Data['TTR'][Var.popu[ps,j]])),Var.popu[ps,j]])*GAParam['ElecPrice']
                        costIndividualOT[j] = GAParam['OTSalary']/4 * (t[j]%96+Data['TTR'][Var.popu[ps,j]]-GAParam['RegularEndtime']) 
                        costIndividualBoat[j] = Data['CostTransport'][Var.popu[ps,BPosArray[ind]]+1,Var.popu[ps,j]+1]
                        costIndividualTask[j] = costIndividualPL[j] + costIndividualOT[j] + costIndividualBoat[j]

                    else:
                        t[j] = 0
                        tB[j] = 0
                        pickupTime[j] = []; 
                        pickupTurbineOrd[j] = [] 
                        
                        costIndividualPL[j] = sum(Data['PL0'][range(96),Var.popu[ps,j]]) * GAParam['ElecPrice']*100
                        costIndividualOT[j] = 0
                        costIndividualBoat[j] = 0
                        costIndividualTask[j] = costIndividualPL[j] + costIndividualOT[j] + costIndividualBoat[j]
                    #如果当天可以完成此任务则更新船只人员信息
                    if ((int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['OTEndtime']+1):
                        BIndindi[j] = copy.deepcopy(BTemp[ind])                    
                        #更新船只人员信息
                        [BInvent[BIndindi[j]], WIndindij] = updateBInventAfterAssign(BInvent[BIndindi[j]],TaskWorkers[:,j]);
                        
                        for kk in range(len(WIndindij)):
                            if type(WIndindij[kk]) == list:
                                WIndindi[kk][j] = copy.deepcopy(WIndindij[kk])
                            else:
                                WIndindi[kk][j] = WIndindij[kk].tolist()
            
                        WonB = copy.deepcopy(WorkersonBoats(BInvent))
                        
                else:
                    #如果没有一艘船只可以直接满足任务需求，那么需要从之前的风机上接人
                    BAvailTwin = copy.deepcopy(BAvail)
                    
                    while True:
                        BTemp=np.where(BAvail==np.max(BAvail))#找到不符合要求中最符合要求的几条船只
                        BTemp= np.array(BTemp).ravel()
                        PTypeArray= [[] for _ in range(len(BTemp))]
                        NumberArray= [[] for _ in range(len(BTemp))]
                        for ii in range(len(BTemp)):
                            PTypeArray[ii]=np.where(np.sum(BInvent[int(BTemp[ii])],axis=1)<TaskWorkers[:,j])#找到不符合的列（技能）
                            PTypeArray[ii]=np.array(PTypeArray[ii]).ravel()
                            NumberArray[ii]=np.sum(TaskWorkers[PTypeArray[ii],j]-np.sum(BInvent[int(BTemp[ii])],axis=1)[PTypeArray[ii]])#与需求求差值后求和
                        BIndindi[j]=np.argmin(NumberArray)
                        #确定缺少的类能类型和数量
                        lackPType = (np.sum(BInvent[int(BIndindi[j])],axis=1)<TaskWorkers[:,j])
                        lackPType = lackPType * 1
                        lackNumber = -np.sum(BInvent[int(BIndindi[j])],axis=1) + TaskWorkers[:,j]
                        lackNumber[lackNumber<0] = 0
                        lackPType = np.column_stack((lackPType,lackNumber))

                        
                        #找到缺少的人员所在位置
                        WPosArray = findWorkers(lackPType,WonB,WIndindi,GAParam['NP'])
                        #决定去哪些风机上接人，如果一台风机上的人员 能够满足要求则接这台风机上的人，如果不能则决定去哪几台风机接人           
                        
                        WonT = summaryWonT(WPosArray)#统计拥有缺少技能类型人员的风机上有多少该类型技能人员数
                        
                        if np.any(np.sum(WonT,axis=1) < lackPType[lackPType[:,0]==1,1]):#某技能类型风机上人员总数与缺少的人员数作比较
                            BAvailTwin[BTemp] = 0
                            if sum(BAvailTwin) == 0:#如果船上和所有风机上的人都不能满足要求那么就从其他船上找人
                                BIndindi[j] = []

                                [BTemp,pickupBoat,tBTemp,tTemp,BInvent,costTemp] = boattoVisit(BInvent,TaskWorkers[:,j],t,BIndindi,ps,Data,j,Var.popu)
                                BIndindi[j] = copy.deepcopy(BTemp)
                                break
                        else:
                            break
                     
                    # end while
                    

                    if np.any(np.sum(WonT,axis=1) >=lackPType[lackPType[:,0]==1,1]):#去风机上接人
                        
                        BTemp=np.where(BAvail==np.max(BAvail))#找到不符合要求中最符合要求的几条船只
                        BTemp= np.array(BTemp).ravel()
                        PTypeArray= [[] for _ in range(len(BTemp))]
                        NumberArray= [[] for _ in range(len(BTemp))]
                        for ii in range(len(BTemp)):
                            PTypeArray[ii]=np.where(np.sum(BInvent[int(BTemp[ii])],axis=1)<TaskWorkers[:,j])#找到不符合的列（技能）
                            PTypeArray[ii]=np.array(PTypeArray[ii]).ravel()
                            NumberArray[ii]=np.sum(TaskWorkers[PTypeArray[ii],j]-np.sum(BInvent[int(BTemp[ii])],axis=1)[PTypeArray[ii]])#与需求求差值后求和
                        BTemp=np.argmin(NumberArray)
                        turbineID = copy.deepcopy(turbinetoVisit(WonT,lackPType))#找到需要访问的风机号

                        BIndindTemp = copy.deepcopy(BIndindi[0:j])
                        BPos = max([item for item in range(len(BIndindTemp)) if BIndindTemp[item] == BTemp])
                        
                        
                        if np.size(turbineID) == 1:#只有一台风机需要访问
                            pickupTurbineOrd[j] = copy.deepcopy(turbineID)
                            pickupTime[j] = t[BPos]
                            tBTemp = max(t[BPos] + Data['TransportTime'][Var.popu[ps,BPos]+1,Var.popu[ps,turbineID]+1],t[turbineID]+Data['TTR'][Var.popu[ps,turbineID]]) + np.ceil(GAParam['DockTime'])
                            tTemp = tBTemp + Data['TransportTime'][Var.popu[ps,turbineID]+1,Var.popu[ps,j]+1] + np.ceil(GAParam['DockTime'])
                            costTemp = Data['CostTransport'][Var.popu[ps,BPos]+1, Var.popu[ps,turbineID]+1] + Data['CostTransport'][Var.popu[ps,turbineID]+1,Var.popu[ps,j]+1]
                        else:#有多台风机需要访问，优化路线
                            costTemp, pickupTurbineOrd[j], pickupTime[j], tBTemp, tTemp = pickupTourOptimization(BPos,j,Var.popu,ps,turbineID,t,Data);
                        
                        if (int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['OTEndtime']+1:#如果当天可以完成此任务则更新船只人员信息

                            BInvent[int(BIndindi[j])] = updateBInventAfterPickup(BInvent[int(BIndindi[j])],turbineID,WIndindi)
                            
                            WonB = copy.deepcopy(WorkersonBoats(BInvent))

                    #如果当天可以完成此任务则更新船只人员信息
                    if (int(tTemp) + Data['TTR'][Var.popu[ps,j]]) < GAParam['OTEndtime']+1:
                        [BInvent[BIndindi[j]], WIndindij] = updateBInventAfterAssign(BInvent[int(BIndindi[j])],TaskWorkers[:,j])
                        
                        for kk in range(len(WIndindij)):
                            WIndindi[kk][j] = copy.deepcopy(WIndindij[kk])
                        
                        WonB = copy.deepcopy(WorkersonBoats(BInvent))
                    else:
                        BIndindi[j] = []
                    
                    #成本计算
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
                        pickupTime[j] = []
                        pickupTurbineOrd[j] = [] 
                        costIndividualPL[j] = sum(Data['PL0'][range(96),Var.popu[ps,j]]) * GAParam['ElecPrice']*100
                        costIndividualOT[j] = 0
                        costIndividualBoat[j] = 0
                        
                    costIndividualTask[j] = costIndividualPL[j] + costIndividualOT[j] + costIndividualBoat[j]
                
                    
                Var.startTime[ps,j] = t[j]
                Var.departTime[ps,j] = tB[j]
                Var.individualCost[ps,j] = copy.deepcopy(costIndividualTask[j])
                Var.individualCostPL[ps,j] = copy.deepcopy(costIndividualPL[j])
                Var.individualCostOT[ps,j] = copy.deepcopy(costIndividualOT[j])
                Var.individualCostBoat[ps,j] = copy.deepcopy(costIndividualBoat[j])
                
            nanPos = [item for item in range(len(BIndindi)) if np.isnan(BIndindi[item])]
                        
            for item in nanPos:
                BIndindi[item] = []
            
            
            Var.BInd[ps] = copy.deepcopy(BIndindi)
            Var.WInd[ps] = copy.deepcopy(WIndindi)
            Var.BInventFinal[ps] = copy.deepcopy(BInvent)
    
            Var.pickupTurbineOrdAll[ps] = copy.deepcopy(pickupTurbineOrd)
            Var.pickupTimeAll[ps] = copy.deepcopy(pickupTime)
                
            try:
                Var.pickupBoatAll[ps][j] = copy.deepcopy(pickupBoat) 
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
            Var.totalCost[ps] = 10e5
    
    Var.fitness_value = 1.0 / Var.totalCost
    return Var



def BInventInitial(WIndInitial,NP):
    
    #初始船只分配，将人员随机和平均分配到各个船只    
    # 输入: 
    #       WIniInitial:初始人员;
    #       NP: 每种技能类型的人员数量;
    # 输出:
    #       BInvent:每艘船的人员库存
    
    MB = len(WIndInitial[0])
    PType = len(WIndInitial)
    
    
    BInvent = [ [] for _ in range(MB)]#长度为船只数量
    for i in np.arange(0,MB):
        BInvent[i]=np.zeros((PType,np.max(NP)))#NP是不同技能类型所具有的人员数 

    usedWorker = unionUsedWorkers(WIndInitial)
    
    unusedWorker = [[] for _ in range(PType)]
    for i in np.arange(0,PType).reshape(-1): 
        unusedWorker[i] = np.setdiff1d(np.arange(0,NP[i]),usedWorker[i])#找不同 找到未安排人员

    for i in np.arange(0,PType).reshape(-1):
        BInventTemp = equipartitionGroup(unusedWorker[i],MB)#平等随机分，把某种未使用过的技能人员平均分配到MB艘船上

        for j in np.arange(0,MB).reshape(-1):
            if np.size(BInventTemp[j]) == 0:#某艘船没有被分配到该技能的人员
                BInvent[j] = copy.deepcopy(BInvent[j])
            else: 
                BInvent[j][i,BInventTemp[j]] = 1#某艘船上某技能的那一行所对应的若干个人员的索引置为1
                
    return BInvent


def unionUsedWorkers(WIndInitial):#前MB个任务被分配后哪些人员被使用了

    MB = len(WIndInitial[0])
    PType = len(WIndInitial)

    usedWorkers = [ [] for _ in range(PType)]
    for i in np.arange(0,PType).reshape(-1):#把每种类型技能的人员并到1一起（原来是按任务分开的）
        for j in np.arange(0,MB).reshape(-1):
            if np.size(WIndInitial[i][j]) != 1:
                usedWorkers[i]=usedWorkers[i] + list(WIndInitial[i][j])
            elif np.size(WIndInitial[i][j]) == 1:
                #usedWorkers[i]=usedWorkers[i] + [WIndInitial[i][j]]
                usedWorkers[i].append(WIndInitial[i][j][0]) 
    return usedWorkers#得到某种技能用了哪些人员编号


def equipartitionGroup(sample,groupNo):#平均和随机分配

    if np.all(np.size(sample)==0):
        group =[ [] for _ in np.arange(0,groupNo).reshape(-1)]
    else:
        sampLen=np.size(sample)
        group = []
        for i in np.arange(0,np.mod(sampLen,groupNo)).reshape(-1):
            k=np.ceil(float(sampLen)/float(groupNo))
            if np.size(sample) != 1:
                group.append(random.sample(sample.tolist(), int(k)))
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


def updateBInventAfterAssign(BInventj, TaskjWorker):#任务分配后船只上的人员库存更新

    WIndj = []
    for i in range(0, len(TaskjWorker)):
        temp_BInventj, temp_WIndj  = inventWorkerSample(BInventj[i], int(TaskjWorker[i]))
        BInventj[i, ] = copy.deepcopy(temp_BInventj)
        WIndj.append(temp_WIndj)

    return BInventj, WIndj


def inventWorkerSample(invent,numberReq):
    
    if numberReq == 0:
        newInvent=copy.deepcopy(invent)
        indSam=[]
    else:
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
#统计目前每天风机上具有的不同工人的数量

    lackPTypeNum = np.size(WPosArray,0) 
    
    workTurbine = np.unique(WPosArray)
    workTurbine = workTurbine[~np.isnan(workTurbine)]
    
    if len(workTurbine) == 0:
        WonT = np.array([[0]])
    else:
        workTurbine = workTurbine.astype(int)

        WonT = np.zeros((np.size(WPosArray,0),max(workTurbine)+1), dtype=np.int)
        for i in np.arange(0,lackPTypeNum).reshape(-1):
            # 注np.histogram实现了和matlab中histc相同的功能，但是返回的是2个tuple的信息，但是只需要第[0]个tuple的信息
            WonT[i,:] = np.histogram((WPosArray[i,:][~np.isnan(WPosArray[i,:])]),np.arange(0,max(workTurbine)+2))[0]
    
    return WonT



def boattoVisit(BInvent,TaskWorkersj,t,BIndindi,ps,Data,j,popu):
    for iii in range(len(BIndindi)):
        if BIndindi[iii] == []:
            BIndindi[iii] = np.nan
    BIndindi = np.array(BIndindi)  

    BInventSum = np.empty((BInvent[0].shape[0],len(BInvent)))
    
    for i in range(len(BInvent)):
        BInventSum[:,i] = np.sum(BInvent[i], axis = 1)
        
    #找到可以符合人员要求的船只组合
    for ii in range(2, len(BInvent)+1):
        combj = [list(x) for x in itt.combinations(range(0,len(BInvent)), ii)]
        combj = np.array(combj)
        for jj in range(0,combj.shape[0]):
            if sum(np.sum(BInventSum[:,combj[jj,:]], axis = 1) < TaskWorkersj):            
                combj[jj,:] = np.zeros((1,ii))
        if sum(sum(combj)) != 0:
            break
        
    #找到最优路线
    delColIdx = np.where(np.sum(combj, axis = 1) == 0)
    combj = np.delete(combj, delColIdx, axis = 0)
    
    
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
                if popu.ndim == 2: 
                    costTemp[jj] = costTemp[jj] + Data['CostTransport'][popu[ps,BPosArray[tourAll[jj,i]]]+1,popu[ps,BPosArray[tourAll[jj,i+1]]]+1]
                else:
                    costTemp[jj] = costTemp[jj] + Data['CostTransport'][popu[BPosArray[tourAll[jj,i]]]+1,popu[BPosArray[tourAll[jj,i+1]]]+1]
            else:
                if popu.ndim == 2:
                    costTemp[jj] = costTemp[jj] + Data['CostTransport'][popu[ps,tourAll[jj,i]]+1, popu[ps,j]+1]
                else:
                    costTemp[jj] = costTemp[jj] + Data['CostTransport'][popu[tourAll[jj,i]]+1, popu[j]+1]
                
    
    # 找到费用最小的路线
    indMinCost = np.where(costTemp == min(costTemp))
    BoatOrd = tourAll[indMinCost]

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
    
    #决定从每艘船上带多少人
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


def findWorkers(lackPType,WonB,WInd,NP):#找到人员所在位置

    lackPTypeInd = np.where(lackPType[:,0] == 1)[0]
    WPosArray = [[[] for _ in range(max(NP))] for _ in range(len(lackPTypeInd))]

    for jj in np.arange(0,len(lackPTypeInd)).reshape(-1):
        # 注：下列语句会产生一个警告，由Python特性决定，对结果没有影响。__main__:1: VisibleDeprecationWarning: converting an array with ndim > 0 to an index will result in an error in the future
        workersonTurb = np.setdiff1d(np.arange(0,NP[lackPTypeInd[jj]]),WonB[lackPTypeInd[jj]])

        for ii in workersonTurb.reshape(-1):
            for j in np.arange(0,np.size(WInd,1)).reshape(-1):
                temp = copy.deepcopy(WInd[lackPTypeInd[jj]][j])  
                if len([item for item in range(len(temp)) if temp[item] == ii]):
                    WPosArray[jj][ii]=copy.deepcopy(j)
                    
    WPosArray = np.array([[(element == [])*np.NaN or element for element in sublist] for sublist in WPosArray])
    return WPosArray


def turbinetoVisit(WonT,lackPType):
    
    lackPNum = lackPType[lackPType[:,0] == 1, 1]
    
    if len(WonT.shape) == 1:
        WonT = np.reshape(WonT, (-1, len(WonT)))
    #找到哪些风机可以满足不同技能的人员需求
    TurbFlag = np.empty(WonT.shape)
    for i in range(len(lackPNum)):
        TurbFlag[i,:] = WonT[i,:] >= lackPNum[i]
        
    # 如果一台风机可以满足任务需求，那么只需要访问这一台风机
    if sum(np.sum(TurbFlag, axis=0) == len(lackPNum)):
        turbineID = np.where(np.sum(TurbFlag, axis=0) == len(lackPNum))
        turbineID = copy.deepcopy(turbineID[0][0])
        
        #如果一台风机不能满足任务需求，那么需要访问多台风机
    else:
        if sum(sum(TurbFlag)) == 0:
            turbineID = np.where(WonT[0,:] == max(WonT[0,:]))
        else:
            turbineID = np.where(np.sum(TurbFlag,axis=0) == max(np.sum(TurbFlag,axis=0)))
        turbineID = copy.deepcopy(turbineID[0][0])
        

        newLackPNum = lackPNum - WonT[:,turbineID]
        newLackPNum[newLackPNum<0] = 0
        newLackPType = copy.deepcopy(lackPType)
        newLackPType[(newLackPType[:,0] == 1),1] = copy.deepcopy(newLackPNum)
        newLackPType[(newLackPType[:,1] == 0),0] = 0
        lackPType = copy.deepcopy(newLackPType)
        
        delRowIdx = np.where(lackPNum <= WonT[:,turbineID])
        WonT = np.delete(WonT, delRowIdx, axis = 0)
        WonT[:,turbineID]= 0
        
        turbineID = np.array(turbineID)

        turbineID = np.append(turbineID, turbinetoVisit(WonT,lackPType))
        
    if np.size(turbineID) == 1:
        newTurbineID = np.array([turbineID])
        turbineID = np.zeros(1)
        turbineID = copy.deepcopy(newTurbineID)
            
    return turbineID

def pickupTourOptimization(BPos,j,popu,ps,turbineID,t,Data):
    #多台风机接人优化路线，使用遍历方法
    # input: turbineID, t, TTR

    # output:
    #        costTemp: 接人过程的费用
    #        pickupTurbineOrd: 访问风机的顺序
    #        pickupTime: 每次接人的开始时间
    #        tTemp: 船只可以出发到任务的开始时间
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
    
    # 找到产生最小费用的路线
    indMinCost = np.where(costTemp == min(costTemp))[0][0]
    pickupTurbineOrd = copy.deepcopy(tours[indMinCost])
    
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
            costTemp = costTemp + Data['CostTransport'][popu[ps,pickupTurbineOrd[i-1]]+1, popu[ps,pickupTurbineOrd[i]]+1]
    
    costTemp = costTemp + Data['CostTransport'][popu[ps,pickupTurbineOrd[i]]+1,popu[ps,j]+1]
    
    return costTemp, pickupTurbineOrd, pickupTime, tBTemp, tTemp


def updateBInventAfterPickup(BInventj, turbineID, WInd):

    for i in range(0, np.size(turbineID)): 
        for j in range(0, np.size(WInd, 0)):

            if np.size(WInd[j][turbineID[i]]) != 0:
                temp = copy.deepcopy(WInd[j][turbineID[i]])
                for k in range(0, np.size(WInd[j][turbineID[i]])):                    
                    BInventj[j][temp[k]] = 1 

    return BInventj


def rank(Var, GAParam, G):#排序

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


def selection(Var, GAParam):#选择

    popu_new=np.zeros((np.size(Var.popu,0),np.size(Var.popu,1)))
    popuWW_new=np.zeros((np.size(Var.popuWW,0),np.size(Var.popuWW,1)))

    for i in range(GAParam['popSize']):
        # 注：GAParam['popSize']是fitness_table里的位置信息，因此要减1
        r=np.dot(np.random.random(), Var.fitness_table[GAParam['popSize']-1])       
        idx=(abs(r - Var.fitness_table)).argmin(0)
    
        for j in range(GAParam['chromoSize']):
            popu_new[i,j]=copy.deepcopy(Var.popu[idx,j])
              
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


def crossover(Var, GAParam):#交叉

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


def mutation(Var, GAParam):#变异

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

            del randInd
            del randInd1
            del randInd2
    del i 
    
    return Var


def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")          

            
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
    conn = MySQLdb.connect(host=host, user=user, passwd=passwd, db=Data_base, charset="utf8")#数据库连接    
    # available boats
    #sql1 = "SELECT * FROM boatinput"
    sql1 =("SELECT * FROM boatinput WHERE Main_sch_Id LIKE '%s'" %(main_sch_id+'%'))#读取船只信息
    boats_df = pd.read_sql(sql=sql1, con=conn)
    #Ship_ID = boats_df['Ship_ID']
    #MB = len(Ship_ID)
    
    del sql1
    
    
    # workers # update 201705
    sql1 = ("SELECT * FROM workerinput WHERE Main_sch_Id LIKE '%s'" %(main_sch_id+'%'))#读取人员信息
    workers_df = pd.read_sql(sql=sql1, con=conn)
    del sql1   
    
    
    
    # turbine coordinate
    sql1 = "SELECT * FROM farms"#读取风场信息
    turbine_coor_df = pd.read_sql(sql=sql1, con=conn)   
    
    del sql1
    
    # main_sch_id = 'XA0002_2017-05-09'
    sql1 = ("SELECT * FROM processinput WHERE Main_sch_Id LIKE '%s'" %(main_sch_id+'%'))#读取任务信息
    # sql1 = "SELECT * FROM processinput WHERE Main_sch_Id LIKE 'XA0002_2017-05-09%'"
    task_df = pd.read_sql(sql=sql1, con=conn)
    # task_df = task_df.drop(task_df.index[range(31)])
    task_df = task_df.reset_index(drop=True)
    del sql1
    
    '''
    输入数据处理 
    '''
    task_df["task_ID"] = task_df["Turbine_ID"].map(str) +'_' + task_df["OM_ID"]
    task_ID = pd.unique(task_df[['task_ID']].values.ravel())
    #print "task_ID",task_ID 
    task_workerType = pd.unique(task_df[['Worker_Type']].values.ravel())
    
    turbine_ID = [None] * len(task_ID)
    #print "turbine_ID",turbine_ID
    OM_Time = [None] * len(task_ID)
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
      
        
    #  计算不同类型工人的数量
    # 保证工人的唯一性
    workers_ID = pd.unique(workers_df[['Workers_ID']].values.ravel())
    workers_ID_idx = [None] * len(workers_ID)
    for i in range(len(workers_ID)):
        idx = np.where(workers_df[['Workers_ID']] == workers_ID[i])
        idxi = idx[0][0]
        workers_ID_idx[i] = idxi
        del idx, idxi
    workers_df = workers_df[workers_df.index.isin(workers_ID_idx)] 
    workers_df = workers_df.reset_index(drop=True)
    
    worker_list_by_skills = [None] * len(task_workerType)
    worker_list = workers_df['Workers_ID'].tolist()
    for j in range(len(task_workerType)):
        idx = np.where(workers_df['Worker_Type'] == task_workerType[j])
        worker_list_by_skills[j] = [worker_list[i] for i in idx[0]]
        del idx
        

    
    task_coor = np.zeros((len(task_ID),3))
    for i in range(len(task_ID)):
        idx = np.where(turbine_coor_df['Turbine_ID'] == turbine_ID[i]) 
        idxi = idx[0][0]
        task_coor[i][0] = Angle(turbine_coor_df['WGS84Latitude'][idxi]).degree
        task_coor[i][1] = Angle(turbine_coor_df['WGS84Longitude'][idxi]).degree
        task_coor[i][2] = turbine_ID[i]
    
    
    
    shoreStation_coor = np.array([[30.847838889,121.8448194444]]) 
    
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
    datapath = os.path.abspath('') +'//Data0613.mat' 
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
    船只
    '''
    Params = dict()
    # boats
    boats_ID = pd.unique(boats_df[['Ship_ID']].values.ravel())
    #boats_ID = boats_ID[[1,2]] # 0808, trival
    Params['MB'] = len(boats_ID)
    Params['boats_ID'] = boats_ID
    
    # 工人
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

################################################################
#后处理
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
    
