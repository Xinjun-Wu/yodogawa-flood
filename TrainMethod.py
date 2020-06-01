import os
import torch
import time
import datetime
from datetime import timedelta
import numpy as np 
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm

from dataSet import YodogawaDataSets
from TrainAndTest import TrainAndTest
from CNN_models import ConvNet_2

    ###################### Initialize Parameters ####################################
READ_VERSION = 1
SAVE_VERSION = 1
TVT_RATIO = [0.2, 0.1, 0.1]
TEST_SPECIFIC = [10, 11]
RANDOM_SEED = 120
STEP_List = [6, 12, 18, 24, 30, 36]
CHECKPOINT = None
CHECKPOINT = [6, 5] ###STEP == 6 , EPOCH == 5
CHECK_EACH_STEP = False

if CHECKPOINT is not None:
    STRT_STEP = CHECKPOINT[0]
    START_STEP_INDEX = STEP_List.index(STRT_STEP)
    STEP_List = STEP_List[START_STEP_INDEX:]
    if isinstance(STEP_List, str):
        STEP_List = [STEP_List]

for STEP in STEP_List:

    INPUT_FOLDER = f'../Save/Step_{STEP}/'
    OUTPUT_FOLDER = f'../Save/Step_{STEP}/'
    Data_FOLDER = f'../TrainData/Step_{STEP}/'


    mydataset = YodogawaDataSets(STEP, Data_FOLDER, TVT_RATIO, TEST_SPECIFIC, RANDOM_SEED)
    model = ConvNet_2(3+int(STEP/6))
    MyTrainAndTest = TrainAndTest(model, mydataset, INPUT_FOLDER, OUTPUT_FOLDER,
                                    CHECKPOINT, READ_VERSION, SAVE_VERSION)
    ############################## Train Paramters #################################
    LR = 0.001
    Train_lambda = lambda epoch: 1/np.sqrt(((epoch % 500)+1.0))
    optimizer = optim.Adam(MyTrainAndTest.MODEL.parameters(), lr = LR, weight_decay = 1e-6)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, Train_lambda)
    TRAIN_PARAMS_DICT = {
                        'EPOCHS' : 10,
                        'BATCHSIZES' : 144,
                        'LOSS_FN' : nn.L1Loss(),
                        'OPTIMIZER' : optimizer,
                        'SCHEDULER' : scheduler,
                        'MODEL_SAVECYCLE' : 1,
                        'RECORDER_SAVECYCLE' : 1,
                        'NUM_WORKERS' : 0,
                        'VALIDATION' : True,
                        'VERBOSE' : 2,
                        'TRANSFER' : False,
                        'CHECK_OPTIMIZER' : True,
                        'CHECK_SCHEDULER' : True,
                        }
    MyTrainAndTest.train(TRAIN_PARAMS_DICT)
    if CHECK_EACH_STEP :
        CHECKPOINT[0] = CHECKPOINT[0] + 6
    else:
        CHECKPOINT = None
