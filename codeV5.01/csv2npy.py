import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class Csv2Npy():
    """ Class used for creating npy files from csv files.
    水理解析結果CSVファイルをNumPy独自のバイナリNPYファイルに作成する。
    
    """
    # 入力CSVファイルのフォルダ
    INPUT_FOLDER = '../CasesData'
    # 出力NPYファイルのフォルダ
    OUTPUT_FOLDER = '../NpyData'
    # 作成されたNPYファイルの数
    npyCount=0


    def __init__(self):
        # 出力フォルダ既存してない場合、フォルダを新規作成する
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)

    
    def water_3_situation(self,case_dir, folder_list=['waterdepth','x_flux','y_flux']):
        """Create a 4-d ndarray from the csv files in the given case folder.
            指定されたフォルダから四次元配列を作成する。

        Arguments:
            case_dir {str} -- the name of case's directory. 
            Example: '../user/train_case/case1'
        
            folder_list {list} -- [folders'name] (default: {['waterdepth','x_flux','y_flux']})
            水深・フラックスx・フラックyのフォルダ名
        
        Returns:
            (S,C,H,W)=(データ数、チャネル、高さ、横幅)
            watersituation -- a ndarray of water_situation with shape (S,C,H,W)
                                S -- samples index
                                C -- water channels
                                H -- height of area
                                W -- widht of area 
        """
        watersituation =[]

        for folder in folder_list:#水深・フラックスx・フラックyフォルダのロープ
            
            directory = case_dir + os.sep + folder
            #ディレクトリとファイルの一覧を取得する
            dirfiles = os.listdir(path=directory)
            #ケース番号の並べ替え 0,1,2,...  x[:-4] 1.csv==>1
            dirfiles.sort(key= lambda x:int(x[:-4]))

            # 計算刻み時間ごとにデータを取得
            for file in dirfiles:
                filepath = directory + os.sep + file
                channel_i = pd.read_csv(filepath, header= None).to_numpy()
                shape_i = channel_i.shape
                # shape_i[0]=高さ48、shape_i[1]=横幅36
                channel_i = channel_i.reshape(-1, 1, shape_i[0], shape_i[1])

                #appending the data to list
                watersituation.append(channel_i)
        
        #convert list of array to ndarray with 4 dimensions
        #watersituation=「水深」全部＋「フラックスx」全部＋「フラックy」全部
        # shape=(72*3,1,48,36)
        watersituation = np.array(watersituation)
        height = watersituation.shape[-2] #48
        width = watersituation.shape[-1]  #36

        #convert the shape to match (S,C,H,W)
        #watersituation=「水深」＋「フラックスx」＋「フラックy」
        # shape=(3,72,48,36)
        watersituation = watersituation.reshape(3,-1, height, width)
        # shape=(72,3,48,36)
        watersituation = watersituation.transpose(1,0,2,3)
        
        return watersituation 

    
    def get_inflow(self, case_dir,mesh_numbers, point=(29,42), file = 'qbr.csv'):
        """Create a 4-d ndarray from the inflow data in the x direction and y direction
            破堤点から流入する氾濫流量四次元配列を作成する。
        Arguments:
            case_dir {str} -- the string of case's directory. 
                            Example: '../user/train_case/case1'
            mesh_numbers {tuple} -- the tuple of mesh_numbers
                            Example: mesh_numbers = (48,36)
        
            point {tuple} -- the point coordinate of inflow in GIS coordinate system (lower left corner)
                            (default: {(29,42)})
            file {str} -- [the file of inflow] (default: {'qbr.csv'})
        
        Returns:
            (S,C,H,W)=(データ数、チャネル、高さ、横幅)
            inflow -- a ndarray of inflow with shape (S,C,H,W)
                                S -- samples index
                                C -- inflow channels
                                H -- height of area
                                W -- width of area 
        """
        filepath = case_dir + '/' + file
        raw_inflow = pd.read_csv(filepath, header = None)
        raw_inflow = raw_inflow.to_numpy()
            
        '''
        convert the point of GIS coordinate to index of ndarray
        左下隅座標を左上隅座標（0から）に変換する
        ^ y               +---> y
        |         ==>     | 
        +----> x          v x
        '''
        #mesh_numbers[0]=48,point[1](x)=42,point[0](y)=29
        index = (mesh_numbers[0]-point[1], point[0]-1)

        inflow = []
        for flux_value in raw_inflow[:,0]:
            init_X_inflow = np.zeros((mesh_numbers[0],mesh_numbers[1]))
            init_X_inflow[index[0], index[1]] = -flux_value # フラックスx数値入力、-x方向
            inflow.append(init_X_inflow)


        for flux_value in raw_inflow[:,0]:
            init_Y_inflow = np.zeros((mesh_numbers[0],mesh_numbers[1]))
            #init_Y_inflow[index[0], index[1]] = i　# フラックスy＝0
            inflow.append(init_Y_inflow)

        inflow = np.array(inflow)
        height = inflow.shape[-2]
        width = inflow.shape[-1]

        #convert the shape to match (S,C,H,W)
        inflow = inflow.reshape(2,-1, height, width)
        inflow = inflow.transpose(1,0,2,3) # 
        #inflow = inflow.reshape(-1,2, height, width)

        return inflow

    def make_npy_data(self):
        """ Make npy files using the self methods.
            NPYファイル作成用関数
        """
        print('Making the npy data...')

        #ディレクトリとファイルの一覧を取得する
        for root, dirs, files in os.walk(self.INPUT_FOLDER, topdown = False):
            case_dir_list  = dirs

        for case in tqdm(case_dir_list):
            case_dir  = os.path.join(self.INPUT_FOLDER,case)
            # ディレクトリだけ処理する
            if os.path.isdir(case_dir):
                # a ndarray of water_situation with shape (S,C,H,W)
                water = self.water_3_situation(case_dir)
                # (S,C,H,W)==>(H,W) 高さ・幅を取得
                mesh_numbers = water.shape[2:]

                inflow = self.get_inflow(case_dir,mesh_numbers)
                npy_data = np.concatenate((water,inflow),axis = 1)
                np.save(self.OUTPUT_FOLDER + f'/{case}.npy',npy_data)
                self.npyCount += 1

        print(f'report : {self.npyCount} npy data has been created.')



if __name__ == "__main__":
    csv2npy = Csv2Npy()
    csv2npy.make_npy_data()



