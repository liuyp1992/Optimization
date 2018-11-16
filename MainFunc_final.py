# -*- coding: utf-8 -*-
"""
Created on Tue May 08 10:55:23 2018

@author: 03010290
"""

import numpy as np
import sys
import os
####
#打印日志
####
import logging

# 创建Logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 创建Handler

# 终端Handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

# 文件Handler
#fileHandler = logging.FileHandler('log.log', mode='a', encoding='UTF-8')

fileHandler = logging.FileHandler(os.path.join(os.getcwd(), 'log.txt'))
fileHandler.setLevel(logging.NOTSET)


# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# 添加到Logger中
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

## 打印日志
#logger.debug('debug 信息')
#logger.info('info 信息')
#logger.warning('warn 信息')
#logger.error('error 信息')
#logger.critical('critical 信息')

####
# 从数据库载入数据 参数预处理
####
#数据库连接
from MBNKPGA import SchedulingPreprocessing
host='10.84.1.111'
#host='10.84.6.50'
user='root'
passwd='123456'
Data_base='xa0002'
#main_sch_id = 'XA0002_2017-12-08-10:52:33:125'
#main_sch_id = 'xa0002_2017-12-29-11:24:28:68'
#main_sch_id ='xa0002_2018-07-09-09:58:00:383'
main_sch_id ='xa0002_2018-06-13-09:58:00:383'
#main_sch_id ='xa0002_2018-06-27-11:00:00:300'#维护方案代码
#main_sch_id='xa0002_2018-06-15-09:00:00:300'
Data, Params = SchedulingPreprocessing(main_sch_id,host,user,passwd,Data_base)#数据预处理

#########################遗传算法####################################

####
# 遗传算法参数设置
####

# 维护任务参数
GAParam = dict()
GAParam['PType'] = len(Params['NP'])#技能类型数
# Params['MB'] = 2 # for test
#Params['NP'] = np.array([8, 9, 3]) # for test
GAParam['MB'] = Params['MB']#船只数
GAParam['NP'] = Params['NP']#各种类型技能具有的人员数量
GAParam['taskLen'] = Params['taskLen']#任务数量
GAParam['TaskWorkers'] = Params['TaskWorkers']#各个任务所需的各种技能类型的人数
GAParam['timeIni'] = 32 #任务开始时间，目前设为九点
GAParam['RegularEndtime'] = 88 #正常下班时间，目前设为六点
GAParam['OTEndtime'] = 88 #有加班的下班时间
GAParam['OTSalary'] = 300 #加班时新
GAParam['ElecPrice'] = 0.8 #电价 元/度
#GAParam['BoatRentPrice'] = 4500 # boat rental price
GAParam['BoatRentPrice'] = 0# 租船费用，目前设为0
GAParam['OilPrice'] = 125 #油费 元/15钟
GAParam['BoatSpd'] = 12 #海里/小时
GAParam['DockTime'] = 1 # 上下船时间 15分钟


Data['TransportTime'] = np.round(Data['Distance']/GAParam['BoatSpd']/1.852*(60.0/15.0))#各个风机之间所需的路程时间

Data['CostTransport'] = Data['Distance']/GAParam['BoatSpd']/1.852*(60.0/15.0) * GAParam['OilPrice']#各个风机之间所需的路程费用

# add params of high priority task list
#GAParam['Priority_condition'] = True#有需要优先处理的维修任务
GAParam['Priority_condition'] = False#没有需要优先处理的维修任务
# GAParam['Priority_len'] = 3
# here the priority_list is a relative No. 
GAParam['Priority_list'] = np.array([0])#需要优先处理的任务
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
GAParam['crossRate'] = 0.6#交叉率
GAParam['mutateRate'] = 0.1#变异率

M = 5#算法重复次数

####
# 模型初始化
####
#初始化任务顺序、总成本和开始时间等设为0

bestSequence = np.zeros((M,GAParam['taskLen']), dtype = np.int)
totalCost = np.zeros((M,1))
bestSTime = np.zeros((M,GAParam['taskLen']))
bestGeneration = np.zeros((M,1), dtype = np.int)

try:
    ####
    # 模型计算和结果输出
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
    #结果回写数据库
    ####
    from MBNKPGA import schedulingPostProcessing
    schedulingPostProcessing(Data, Params, results[0],main_sch_id,host,user,passwd,Data_base)

except :
    logger.exception("Exception Logged")

