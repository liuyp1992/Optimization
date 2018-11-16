#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Mon Apr 24 01:58:52 2017

main_func
which incldues:
    scheduling preprocessing
    GA modeling process
    boat routine optimization
    scheduling postprocessing
'''
import numpy as np
import sys
import pymysql as MySQLdb
#import pandas as pd

####
# Data loading from database and Params pre-processing
####
#database connect
def test(main_sch_id):
    Data_base='xa0002' # for PC verson
    #conn = MySQLdb.connect(host='10.84.1.111', user='root', passwd='123456', db=Data_base, charset="utf8")
    conn = MySQLdb.connect(host='localhost', user='root', passwd='root', db=Data_base, charset="utf8")
    cur = conn.cursor()
    try:
        cur.execute("UPDATE schedul_plan SET statue='2' where Main_sch_Id='"+main_sch_id+"'")
        process(main_sch_id)
        cur.execute("UPDATE schedul_plan SET statue='3' where Main_sch_Id='"+main_sch_id+"'")
        conn.commit()
    except Exception,e:
        cur.execute("UPDATE schedul_plan SET statue='4' where Main_sch_Id='"+main_sch_id+"'")
        conn.commit()
        print "Error: 没有成功写入数据库"
        print "Error:",e
        print False
        return False
    else:
        print "内容写入文件成功"
        print True
        return True
    finally:
        cur.close()
        conn.close()

####
# Data loading from database and Params pre-processing
####
def process(main_sch_id):
    from MBNKPGA import SchedulingPreprocessing
    host='localhost'
    #host='10.84.6.50'
    user='root'
    passwd='root'
    Data_base='xa0002'
    #main_sch_id = 'XA0002_2017-12-08-10:52:33:125'
    #main_sch_id = 'xa0002_2017-12-29-11:24:28:68'
    #main_sch_id ='xa0002_2018-07-09-09:58:00:383'
    #main_sch_id ='xa0002_2018-06-13-09:58:00:383'
    #main_sch_id ='xa0002_2018-06-27-11:00:00:300'
    #main_sch_id='xa0002_2018-06-15-09:00:00:300'
    Data, Params = SchedulingPreprocessing(main_sch_id,host,user,passwd,Data_base)
    
    #########################GA Modeling####################################
    
    ####
    # Params settings for GA modeling
    ####
    
    # maintenance params
    GAParam = dict()
    GAParam['PType'] = len(Params['NP'])
    # Params['MB'] = 2 # for test
    #Params['NP'] = np.array([8, 9, 3]) # for test
    GAParam['MB'] = Params['MB']
    GAParam['NP'] = Params['NP'] 
    GAParam['taskLen'] = Params['taskLen']
    GAParam['TaskWorkers'] = Params['TaskWorkers'] 
    GAParam['timeIni'] = 32 # start work time
    GAParam['RegularEndtime'] = 88 # regular end time
    GAParam['OTEndtime'] = 88 # OT end time
    GAParam['OTSalary'] = 300 # overtime salary
    GAParam['ElecPrice'] = 0.8 # price of electricity
    #GAParam['BoatRentPrice'] = 4500 # boat rental price
    GAParam['BoatRentPrice'] = 0
    GAParam['OilPrice'] = 125 #
    #GAParam['BoatSpd'] = 20 # knot/hr
    GAParam['BoatSpd'] = 12 # knot/hr海里/小时
    GAParam['DockTime'] = 1 # unit:quater (15 mins)
    
    
    Data['TransportTime'] = np.round(Data['Distance']/GAParam['BoatSpd']/1.852*(60.0/15.0))
    
    Data['CostTransport'] = Data['Distance']/GAParam['BoatSpd']/1.852*(60.0/15.0) * GAParam['OilPrice']
    
    # add params of high priority task list
    #GAParam['Priority_condition'] = True
    GAParam['Priority_condition'] = False
    # GAParam['Priority_len'] = 3
    # here the priority_list is a relative No. 
    GAParam['Priority_list'] = np.array([0])
    if GAParam['Priority_condition'] == False:
        GAParam['Priority_list']=[]
    
    if GAParam['taskLen']-len(GAParam['Priority_list'])<=1:
        print "请输入多个维护任务"
        sys.exit(1)
    
    for _ in range(len(GAParam['NP'])):
        if GAParam['NP'][_]<max(GAParam['TaskWorkers'][_]):
           print "所带维护人员数不够"
           sys.exit(1)
    
    if GAParam['MB']==0:
        print "请输入维护船只"
        sys.exit(1)
    
    # GA params
    GAParam['crossRate'] = 0.6
    GAParam['mutateRate'] = 0.1
    
    # M: iteration times
    M = 5
    
    ####
    # Model Initialization
    ####
    
    bestSequence = np.zeros((M,GAParam['taskLen']), dtype = np.int)
    totalCost = np.zeros((M,1))
    bestSTime = np.zeros((M,GAParam['taskLen']))
    bestGeneration = np.zeros((M,1), dtype = np.int)
    
    
    ####
    # Model computation and result output
    ####
    from MBNKPGA import tic, toc
    results = [None] * M
    
    from MBNKPGA import MBNKPGA
    for i in range(M):
        tic()
        results[i] = MBNKPGA(GAParam, Data)
        toc()
        
    totalCost = [None] * M
    for m in range(M):
        totalCost[m] = results[m]['objValue']
    
    bestInd = np.argsort(totalCost)
    bestInd = bestInd[0]
    scheduling = results[bestInd]['bestSequence']
    optTotalCost = totalCost[bestInd]
    
    
    
    
    ####
    # results write to database
    ####
    from MBNKPGA import schedulingPostProcessing
    schedulingPostProcessing(Data, Params, results[0],main_sch_id,host,user,passwd,Data_base)
import sys
if __name__ == '__main__':
  print "开始调用py"
  #test(sys.argv[1])
  test('xa0002_2018-06-15-09:00:00:300')