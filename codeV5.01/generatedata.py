import pandas as pd 
import numpy as np 
import os 
from scipy import integrate
from tqdm import tqdm



class GenerateData():
    """ Class used for creating training npz files from case npy files.
    変換された水理解析結果NPYファイルを訓練用NPZファイルに作成する。
    作成時間間隔を指定する可能
    """
    
    def __init__(self, input_folder = '../NpyData', output_folder = '../TrainData/DataSet001'):

        self.INPUT_FOLDER = input_folder    #既存NPYファイルのフォルダ
        self.OUTPUT_FOLDER = output_folder  #出力NPZファイルのフォルダ
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)
            print(f'\nCreating the output folder of npz file at {self.OUTPUT_FOLDER}')
        else:
            print(f'Warning: The output npz folder \' {self.OUTPUT_FOLDER}\' has been existed!')
            print('\n  The files in the output folder will be overwritten!')
            print('\n  You can input a new folder path or Enter for default.')
            choice = input('Type your new output_folder or [no]')

            if choice.upper() != 'NO' and choice!='':
                self.OUTPUT_FOLDER = choice
                os.makedirs(self.OUTPUT_FOLDER)
                print(f'Creating folder for DataSet: {self.OUTPUT_FOLDER}')

    def create_time_series(self, time_series_nums, timestep):
        """Calculate the index of input and output,which are corresponding to each sanple.        
        Arguments:
            time_series_nums {int} -- the nums of the parts each case have been split
        
        Keyword Arguments:
            timestep {int} -- the time step will be used in train and test (default: {1})
        
        Returns:
            input_index, output_index {list} -- the index corresponding to each sample
        """
        input_index = list(range(time_series_nums - timestep))
        output_index = []
        for i in input_index:
            output_index.append(i + timestep)
        return input_index, output_index


    def inflow_integrate(self,eachcase,start,end,cellsize, timeinterval):
        """integrate the inflow between start and end
        
        Arguments:
            eachcase {ndarray} -- all data of each case, example: ndarray.shape = (73,5,48,36)
            start {int} -- the start time
            end {int} -- the end time
            cellsize {tuple} -- the tuple of cellsize
                                example: cellseze = (dx,dy) ,dx --> 30, dy --> 40
            timeinterval {int} -- the minutes of time interval
        
        Returns:
            flux {nadrray} -- the array of flux, 
            example:ndarray.shape = (2,48,36),
                    flux[0] --> flux in x direction
                    flux[1] --> flux in y direction

        """
        x_integrate_item = []
        y_integrate_item = []

        #example: height = 48, width = 36
        height = eachcase.shape[-2] 
        width = eachcase.shape[-1]

        #example: eachcase.shape = (73,5,48,36)
        x_inflow = eachcase[:,-2,:,:]           #x_inflow.shape = (73,48,36)
        y_inflow = eachcase[:,-1,:,:]           #y_inflow.shape = (73,48,36)

        x_flux = x_inflow/cellsize[1]
        y_flux = y_inflow/cellsize[0]

        #integrate the inflow between start and end using simps functions
        i = start
        while i  <= end:
            x_integrate_item.append(x_flux[i,:,:])      #x_integrate_item = [ndarray,ndarray,...ndarray], ndarray.shape = (48,36)
            y_integrate_item.append(y_flux[i,:,:])
            i += 1

        x_flux = integrate.simps(x_integrate_item, dx = timeinterval*60, axis = 0)   #x_flux.shape = (48,36)
        y_flux = integrate.simps(y_integrate_item, dx = timeinterval*60, axis = 0)   #y_flux.shape = (48,36)

        x_flux = x_flux.reshape(1,height,width)     #x_flux.shape = (1,48,36)
        y_flux = y_flux.reshape(1,height,width)     #y_flux.shape = (1,48,36)

        flux = np.concatenate((x_flux,y_flux),axis = 0)     #flux.shape = (2,48,36)
        return flux


    def generate(self,eachcase, time_series_nums,cellsize,timeinterval, timestep):
        input_list = []
        inflow_list = []
        output_list= []
        input_index, output_index = self.create_time_series(time_series_nums,timestep)
        for i in input_index:
            input_list.append(eachcase[i,:3,:,:])      #input_list = [ndarray,ndarray,...ndarray], ndarray.shape = (3,48,36)

        for j in output_index:
            output_list.append(eachcase[j,:3,:,:])

        for k in range(len(input_index)):   #example: len(input_index) = 72
            start  = input_index[k]
            end = output_index[k]

            flux = self.inflow_integrate(eachcase,start,end,cellsize,timeinterval) #example: flux.shape = (2,48,36)
            inflow_list.append(flux)

        input_array = np.array(input_list)      #example: input_array.shape = (72,3,48,36)
        output_array = np.array(output_list)    #example: output_array.shape = (72,3,48,36)
        inflow_array = np.array(inflow_list)    #example: inflow_array.shape = (72,2,48,36)

        learning_value = np.concatenate((input_array,inflow_array), axis = 1)       #example: learning_value.shape = (72,5,48,36)
        teacher_value = output_array                        #example: teacher_value.shape = (72,3,48,36)

        return learning_value, teacher_value


    def run(self,cellsize,timeinterval = 10,timestep = 1):
        casename_list = os.listdir(self.INPUT_FOLDER)
        print('Generating data ...')
        for casename in tqdm(casename_list):
            eachcase = np.load(self.INPUT_FOLDER + os.sep + casename)
            
            time_series_nums = eachcase.shape[0]
            learning_value, teacher_value = self.generate(eachcase,time_series_nums,cellsize,timeinterval,timestep)
            savename = casename.split('.')[0]
            np.savez(self.OUTPUT_FOLDER + os.sep + f"{savename}_Step_{timestep}",learning_value=learning_value,teacher_value=teacher_value)




if __name__ == "__main__":
    CELLSIZE = (285.44,231)
    generatedata = GenerateData(output_folder="../TrainData/DataSet_ForTest")
    generatedata.run(CELLSIZE,timestep=2)


