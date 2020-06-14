import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class Csv2Npy():

    def __init__(self, input_folder='../CasesData/', 
                        output_folder='../NpyData/'):
        self.INPUT_FOLDER = input_folder
        self.OUTPUT_FOLDER = output_folder
        self.NPY_COUNT = 0

        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)
    

    def _get_data(self, casefolder_path):
        feature_path_List = []

        for root, dirs, files in os.walk(casefolder_path, topdown=True):
            for dir in dirs:
                feature_path_List.append(os.path.join(root,dir))
            break

        #读取入流数据
        inflow_path = os.path.join(root,files[0])
        inflow_Array = pd.read_csv(inflow_path, header = None).to_numpy()
        
        #遍历waterdepth, xflux, yflux
        data_List = []
        for feature_path in feature_path_List:
            feature_data_List = []#每个通道的数据，比如waterdepth
            index_List = os.listdir(feature_path)
            index_List.sort(key=lambda x:int(x.split('.')[0]))

            for index in index_List:
                file_path = os.path.join(feature_path,index)
                index_data_Array = pd.read_csv(file_path, header= None).to_numpy()
                feature_data_List.append(index_data_Array)

            feature_Array = np.array(feature_data_List)
            data_List.append(feature_Array)

        data_Array = np.array(data_List)
        data_Array = data_Array.transpose(1, 0, 2, 3)

        return data_Array, inflow_Array
    

    def _walk_casesinflow(self):
        caseinflow_Dictionary = {}
        casefolder_List = os.listdir(self.INPUT_FOLDER)
        casefolder_List.sort(key=lambda x:int(x.split('_')[0][4:]), reverse=False)
        #遍历每个case
        for casefolder in tqdm(casefolder_List):
            casefolder_path = os.path.join(self.INPUT_FOLDER,casefolder)
            casename_Str = casefolder.split("_")[0][4:]
            watersituation, inflow = self._get_data(casefolder_path)
            #保存当前case的数据
            savename = os.path.join(self.OUTPUT_FOLDER,casename_Str+'.npy')
            np.save(savename, watersituation)

            caseinflow_Dictionary[casename_Str] = inflow
            self.NPY_COUNT += 1
        #保存所有case的入流
        savename = os.path.join(self.OUTPUT_FOLDER, "inflow.npy")
        np.save(savename, caseinflow_Dictionary)

    def run(self):
        self._walk_casesinflow()
        print(f"Have generated {self.NPY_COUNT} .npy files")

if __name__ == "__main__":
    NAME = 'Yodogawa'
    INPUT = '../CasesData/'
    OUTPUT = f'../NpyData/{NAME}'

    mynpy = Csv2Npy(INPUT,OUTPUT)
    mynpy.run()