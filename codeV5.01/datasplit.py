import os 
import numpy as np
import torch
from tqdm import tqdm




class DataSplit():
    def __init__(self,input_folder):
        self.INPUT_FOLDER = input_folder
        return

    def split_data(self,subdata):
        print(f'creat subsets of data: {subdata}')
        casename_list = os.listdir(self.INPUT_FOLDER)
        parts_list = np.array_split(casename_list,subdata)
        return parts_list

    def loaddata(self,part):
        learning_value = []
        teacher_value = []
        for casename in tqdm(part):
            case_dir = self.INPUT_FOLDER + os.sep + casename
            data = np.load(case_dir)
            l_shape= data['learning_value'].shape
            t_shape = data['teacher_value'].shape
            height = l_shape[-2]
            width = l_shape[-1]
            l_channel = l_shape[-3]
            t_channel = t_shape[-3]
            learning_value.append(data['learning_value'])
            teacher_value.append(data['teacher_value'])
        learning_value = np.asarray(learning_value).reshape(-1,l_channel,height,width)
        teacher_value = np.asarray(teacher_value).reshape(-1,t_channel,height,width)
        
        learning_value_tensor = torch.tensor(learning_value,dtype=torch.float)
        teacher_value_tensor = torch.tensor(teacher_value,dtype=torch.float)

        return learning_value_tensor,teacher_value_tensor



if  __name__ == "__main__":

    datasplit = DataSplit(input_folder =r'E:\Wu\Flood\20191127_計算ケース追加\TrainData\DataSet_ForTest' )
    parts_list = datasplit.split_data(subdata = 4)
    i = 0
    for part in parts_list:
        print(f'loading data in part_{i}')
        # lv --> learning value
        # tv --> teacher value
        lv,tv = datasplit.loaddata(part)
        print(f'train part_{i}')
        i +=1
        print(lv.shape,tv.shape)

