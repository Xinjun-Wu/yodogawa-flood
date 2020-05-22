import os 
import numpy as np 
import torch
import torch.nn as nn

from csv2npy import Csv2Npy
from generatedata import GenerateData
from datasplit import DataSplit
from trainandtest import TrainAndTest
from CNN_models import ConvNet_2


#========================================================================
# ステップ１
# 水理解析結果CSVファイルをNumPy独自のバイナリNPYファイルに作成する
#   入力CSVファイルのフォルダINPUT_FOLDER = '../CasesData'
#   出力NPYファイルのフォルダOUTPUT_FOLDER = '../NpyData'
#csv2npy
CSV2NPY = False

#============================================================================
# ステップ２
# 変換された水理解析結果NPYファイルを訓練用NPZファイルに作成する
#   入力NPYファイルのフォルダ、GENERATE_DATA_INPUT_FOLDER指定
#   出力NPZファイルのフォルダ、GENERATE_DATA_OUTPUT_FOLER指定
#generatedata
GENERATEDATA= False
TIMESTEP = 1        #出力訓練データの時間ステップ
GENERATE_DATA_INPUT_FOLDER = r'E:\Wu\Flood\20191127_計算ケース追加\NpyData'
GENERATE_DATA_OUTPUT_FOLER = f"../TrainData/DataSet_TimeStep_{TIMESTEP}"
CELLSIZE = (285.44,231)
TIMEINTERVAL = 10   #計算時間間隔


#============================================================================
#datasplit
DATA_SPLIT_INPUT_FOLDER  = GENERATE_DATA_OUTPUT_FOLER
SUBDATA = 2


#============================================================================
#train and test
TRAIN = False
TEST = True
NEWMODEL = False
PATH_MODEL = f'../save/TimeStep_{TIMESTEP}/model_save/modelv5_1576760144.pt'
PATH_SAVE =  f'../save/TimeStep_{TIMESTEP}'
EPOCHS = 50
BATCHSIZE = 3000
LR = 0.001
NUM_WORKERS=12
LOSS_FN_TRAIN = nn.L1Loss()

#============================================================================
TESTDATA = f'../TrainData/DataSet_TimeStep_{TIMESTEP}/case23_10min_Step_{TIMESTEP}.npz'
LOSS_FN_Test = nn.L1Loss()

if __name__ == "__main__":
    # ステップ１
    # 水理解析結果CSVファイルをNumPy独自のバイナリNPYファイルに作成する
    #csv to npy
    if CSV2NPY:
        csv2npy = Csv2Npy()
        csv2npy.make_npy_data()

    # ステップ２
    # 変換された水理解析結果NPYファイルを訓練用NPZファイルに作成する
    #generate data from .npy files for train and test
    if GENERATEDATA:
        # クラスの初期化
        generatedata = GenerateData(input_folder=GENERATE_DATA_INPUT_FOLDER,
                                    output_folder=GENERATE_DATA_OUTPUT_FOLER)
        # 実行本番
        generatedata.run(cellsize = CELLSIZE,timeinterval=TIMEINTERVAL, timestep=TIMESTEP)


    if NEWMODEL:
        model = ConvNet_2()
    else:
        model = torch.load(PATH_MODEL)
        model.eval()


    trainandtest = TrainAndTest(model,path_save=PATH_SAVE)

    if TRAIN:
        #split the huge data to sub-dataset
        datasplit = DataSplit(input_folder = DATA_SPLIT_INPUT_FOLDER)
        parts_list = datasplit.split_data(subdata = SUBDATA)

        #train model
        i = 0
        for part in parts_list:

            print(f'loading data in part_{i}')
            # lv --> learning value
            # tv --> teacher value
            lv,tv = datasplit.loaddata(part)

            print(f'train part_{i}')
            i +=1
            trainandtest.train(lv,tv,LOSS_FN_TRAIN,EPOCHS,BATCHSIZE,LR)

            

    if TEST:
        #test model
        testdata = np.load(TESTDATA)
        lv = testdata['learning_value']
        lv = torch.tensor(lv,dtype=torch.float)
        tv = testdata['teacher_value']
        tv = torch.tensor(tv,dtype=torch.float)

        trainandtest.test_0(lv,tv,TIMESTEP,LOSS_FN_Test)







