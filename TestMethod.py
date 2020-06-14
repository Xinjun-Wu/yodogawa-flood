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

from dataSet import CustomizeDataSets
from TrainAndTest import TrainAndTest
from CNN_models import ConvNet_2
if __name__ == "__main__":
    ###################### Initialize Parameters ####################################
    READ_VERSION = 1
    SAVE_VERSION = 1
    TVT_RATIO = [0.2, 0.1, 0.05]
    TEST_SPECIFIC = [10, 11]
    RANDOM_SEED = 120
    TEST_NUM_WORKERS = 0
    BPNAME_List = ['Yodogaawa']
    STEP_List = [6, 12, 18, 24, 30, 36]
    #STEP_List = [6]
    START_EPOCH =0
    END_EPOCH = 1
    EPOCH_STEP = 1
    CHECKPOINT = None
    SAVE_CYCLE = 10

    for BPNAME in BPNAME_List:

        for STEP in STEP_List:

            INPUT_FOLDER = f'../Save/{BPNAME}/Step_{STEP}/'
            OUTPUT_FOLDER = f'../Save/{BPNAME}/Step_{STEP}/'
            Data_FOLDER = f'../TrainData/{BPNAME}/Step_{STEP}/'

            mydataset = CustomizeDataSets(STEP, Data_FOLDER, TVT_RATIO, TEST_SPECIFIC, RANDOM_SEED, BPNAME)
            test_datainfo, testdatasets = mydataset.select('test')
            TEST_BATCHSIZES = test_datainfo['n_sample_eachcase']
            TEST_CASE_LIST = test_datainfo['case_index_List']
            testloader = Data.DataLoader(dataset=testdatasets, batch_size=TEST_BATCHSIZES, 
                                                shuffle=False, num_workers=TEST_NUM_WORKERS, pin_memory=True)
            TEST_INFO_DATA = {
                                'TEST_CASE_LIST' : TEST_CASE_LIST,
                                'TEST_DATALOADER' : testloader
                                }
            TEST_PARAMS_DICT = {
                            'LOSS_FN' : nn.L1Loss(),
                            'NUM_WORKERS': 0
                            }

            model = ConvNet_2(3+int(STEP/6))
            MyTrainAndTest = TrainAndTest(model, mydataset, INPUT_FOLDER, OUTPUT_FOLDER,
                                            CHECKPOINT, READ_VERSION, SAVE_VERSION)

            TEST_LOSS_path = os.path.join(OUTPUT_FOLDER, 'test', f'model_V{READ_VERSION} test loss.csv')
            if os.path.exists(TEST_LOSS_path):
                TEST_LOSS = pd.read_csv(TEST_LOSS_path, index_col=0)
            else:
                TEST_LOSS = pd.DataFrame()

            for epoch in range(START_EPOCH, END_EPOCH, EPOCH_STEP):
                epoch += EPOCH_STEP
                CHECKPOINT = [BPNAME, STEP, epoch]
                TEST_recorder_Dict = MyTrainAndTest.test(TEST_PARAMS_DICT, CHECKPOINT, TEST_INFO_DATA)
                TEST_LOSS = TEST_LOSS.append(pd.DataFrame(TEST_recorder_Dict), ignore_index=True)

                if epoch % SAVE_CYCLE == 0:
                    TEST_LOSS.to_csv(TEST_LOSS_path, float_format='%.4f')
            TEST_LOSS.to_csv(TEST_LOSS_path, float_format='%.4f')
            print('Done!')


