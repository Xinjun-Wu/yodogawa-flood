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
TEST_SPECIFIC = [10, 12]
RANDOM_SEED = 120
TEST_NUM_WORKERS = 0
STEP_List = [6, 12, 18, 24, 30, 36]
START_EPOCH =0
END_EPOCH = 10
EPOCH_STEP = 1

for STEP in STEP_List:

    INPUT_FOLDER = f'../Save/Step_{STEP}/'
    OUTPUT_FOLDER = f'../Save/Step_{STEP}/'
    Data_FOLDER = f'../TrainData/Step_{STEP}/'

    mydataset = YodogawaDataSets(STEP, Data_FOLDER, TVT_RATIO, TEST_SPECIFIC, RANDOM_SEED)
    test_datainfo, testdatasets = mydataset.select('test')
    TEST_BATCHSIZES = test_datainfo['n_sample_eachcase']
    TEST_CASE_LIST = test_datainfo['case_index_List']
    testloader = Data.DataLoader(dataset=testdatasets, batch_size=TEST_BATCHSIZES, 
                                        shuffle=False, num_workers=TEST_NUM_WORKERS, pin_memory=True)
    TEST_INFO_DATA = {
                        'TEST_CASE_LIST' : TEST_CASE_LIST,
                        'TEST_DATALOADER' : testloader
                        }

    for epoch in range(START_EPOCH, END_EPOCH, EPOCH_STEP):
        epoch += EPOCH_STEP
        CHECKPOINT = [STEP, epoch]

        model = ConvNet_2(3+int(STEP/6))
        MyTrainAndTest = TrainAndTest(model, mydataset, INPUT_FOLDER, OUTPUT_FOLDER,
                                    CHECKPOINT, READ_VERSION, SAVE_VERSION)

        TEST_PARAMS_DICT = {
                            'LOSS_FN' : nn.L1Loss(),
                            'NUM_WORKERS': 0
                            }
        TEST_recorder_Dict = MyTrainAndTest.test(TEST_PARAMS_DICT, CHECKPOINT, TEST_INFO_DATA)

