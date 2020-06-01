import numpy as np 
import time
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data as Data
import matplotlib.pyplot as plt
#from dataSplit import DataSplit

from CNN_models import ConvNet_2
from tools import check_path,make_image,image2gif,plot_loss,format_loss_list
from dataSet import KinugawaDataSets
from trainingTest import TrainAndTest


def get_modelinfo(check_Point):
    bpname = check_Point[0]
    step = check_Point[1]*6
    EOPCH = check_Point[2]
    path = f"../save/{bpname}/TimeStep_{step}/"
    #F:\ArcGIS\Flood\kinugawa\save\BP028\TimeStep_6\model_save
    savedModel=path + f"model_save/modelv8_epoch_{EOPCH}.pt"
    savedModel_=path + f"model_save/modelv6_epoch_{EOPCH}.pt"
    if os.path.exists(savedModel):
        checkpoint = torch.load(savedModel)
    elif os.path.exists(savedModel_):
        checkpoint = torch.load(savedModel_)
    else:
        raise ValueError("指定的checkpoint不存在")

    train_Loss = checkpoint['trainLoss']
    train_Loss = format_loss_list(train_Loss)

    if 'lrList' in checkpoint.keys():
        lr_List = checkpoint['lrList']
    else:
        lr_List = None
    return train_Loss,lr_List



if __name__ == "__main__": 
    # EPOCHS = 10
    # BATCHSIZE = 12
    # LR = 0.0005
    # LOSS_FN_TEST = nn.L1Loss()
    # # LOSS_FN_TRAIN = nn.MSELoss()
    # BPName_List=['BP008','BP018','BP023','BP028','BP033','BP038','BP043'] #ケース番号
    # BPName_List=['BP028'] #ケース番号
    
    # CHECK_POINT = ['BP028',1, 10]
    # BPNAME = 'BP028'
    # step = CHECK_POINT[1]*6

    # kinugawaData = KinugawaDataSets(bpname = BPNAME,timestep = step,
    #                 split_pattern = 'specific',specific = (-1,14,15), fulltrain=True) # 初始化数据集
    
    # net = ConvNet_2() #实例化网络
    # #初始化训练器
    # trainandtest = TrainAndTest(model = net,bpname = BPNAME,step = step,check_Point=CHECK_POINT) 
    # #开始训练
    # loss = trainandtest.test(dataset = kinugawaData,loss_fn=LOSS_FN_TEST,threshold_loss=0.01)
    ####################################################################################################
    LOSS_FN_TEST = nn.L1Loss()
    BPName_List=['BP008','BP018','BP023','BP028','BP033','BP038','BP043'] #ケース番号
    BPName_List=['BP018','BP023','BP028','BP033','BP038','BP043'] #ケース番号
    THRESHOLD_LOSS = 0.03
    START_EPOCH =0
    END_EPOCH = 2000
    EPOCH_STEP = 10
    
    #########################################################

    for BPNAME in BPName_List:
        for hour in range(1,7):
            step = hour*6

            test_loss_list = []
            test_casename_list = []
            kinugawaData = KinugawaDataSets(bpname = BPNAME, timestep = step, 
                                             split_pattern = 'specific',specific = (-1,10,(2,6)), fulltrain=True) # 初始化数据集

            kinugawaData.set_select('test')
            for case_index in kinugawaData.index:
                test_casename_list.append(f'case {case_index}')
            ##############################################################################
            for epoch in range(START_EPOCH, END_EPOCH, EPOCH_STEP):
                epoch += EPOCH_STEP

                CHECK_POINT = [BPNAME,hour,epoch]

                net = ConvNet_2(hour+3) #实例化网络
                #初始化训练测试器
                trainandtest = TrainAndTest(model = net,bpname = BPNAME,step = step,check_Point=CHECK_POINT)
                #开始测试,返回值为list, [['casename1',test_loss1],['casename2',test_loss2]]

                print(f'\n Testing on {BPNAME} {hour} hour\'s model with epoch {epoch} ')
                
                losses = trainandtest.test(dataset = kinugawaData,loss_fn=LOSS_FN_TEST,threshold_loss=THRESHOLD_LOSS)
                temp_loss_list = [epoch]
                for loss in losses: # loss = ['casename1',test_loss1]
                    temp_loss_list.append(loss[1])
                test_loss_list.append(temp_loss_list)
            #########################################################################
            train_loss_list,lr_list = get_modelinfo(CHECK_POINT)

            train_loss_array = np.array(train_loss_list)
            test_loss_array = np.array(test_loss_list)
            if lr_list is not None:
                lr_array = np.array(lr_list)
            else:
                lr_array = None


            path = f'../save/{BPNAME}/TimeStep_{step}/'
            savefilename = path + f'test recoder on {START_EPOCH}-{END_EPOCH}.png'
            savecsvfilename = path+f'test recoder on {START_EPOCH}-{END_EPOCH}.csv'

            csv_header = 'Epochs'
            for case_name in test_casename_list:
                csv_header += ','
                csv_header += case_name
            np.savetxt(savecsvfilename,test_loss_array,fmt = '%10.6f',delimiter = ',', header = csv_header)

            xlim = (START_EPOCH , END_EPOCH)
            print('Plotting...')
            plot_loss(savepath = savefilename,loss_array = train_loss_array,lr_array = lr_array,test_array= test_loss_array,
                    test_label= test_casename_list,xlim =xlim)
            print('Done')

            

            







