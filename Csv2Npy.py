import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class Csv2Npy():
    """ Class used for creating npy files from csv files.
    水理解析結果CSVファイルをNumPy独自のバイナリNPYファイルに作成する。
    """

    def __init__(self, input_folder = '../CasesData/BP028/', 
                    output_folder = '../NpyData/BP028/'):
        # # 入力CSVファイルのフォルダ
        self.INPUT_FOLDER = input_folder
        # # 出力NPYファイルのフォルダ
        self.OUTPUT_FOLDER = output_folder
        # 作成されたNPYファイルの数
        self.npyCount=0
        # 出力フォルダ既存してない場合、フォルダを新規作成する
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)
        self.BPname = self.INPUT_FOLDER.split('/')[-2]
        self.__getBPinfo()
        
    def __getBPinfo(self):
        """
        Get the all information of BP
        Returns:
            save it ro disk for recall in furture
        """
        Points_I_J = self.__get_PointLists()
        ijCoordinates,xyCoordinates,row_col_num = self.__makeIJ_XYCoordinates()
        inflow_DataFrame = self.__get__Inflow()

        npz_save_name = self.OUTPUT_FOLDER + os.sep + f"../{self.BPname}_info.npz"
        np.savez(npz_save_name,ijCoordinates=ijCoordinates,xyCoordinates=xyCoordinates,row_col_num=row_col_num,
                points_I_J = Points_I_J,inflow_DataFrame = [inflow_DataFrame,0])
                #inflow_DataFrame = [inflow_DataFrame,0] 更好的利用numpy储存DataFrame
        print(f'GIS用座標ファイル{self.BPname}_info.npz(ijCoordinates,xyCoordinates,row_col_num,points_I_J,inflow_DataFrame)が作成しました。')
        
    def __get_PointLists(self,listfile=r'../氾濫流量ハイドロ/破堤点毎格子情報_ver20200515.xlsx',skiprows=0,index_col=0):
        """
        获取破堤点网格番号
            
            Keyword Arguments:
                listfile {xlsx file} -- [argument for read file with pandas.read_excel] (default: {r'../氾濫流量ハイドロ/破堤点格子番号.xlsx'})
                skiprows {int} -- [argument for read file with pandas.read_excel] (default: {1})
                index_col {int} -- [argument for read file with pandas.read_excel] (default: {0})
            Returns:
                Points_I_J[list] -- [破堤点对应的 I J 坐标]
            Example:
                Points_I_J = [[270,1], [271,1], [272,1], [273,1], [274,1]]
        """
        pointlists = pd.read_excel(listfile, skiprows=skiprows, index_col=index_col)
        BPname = self.BPname
        # return a array : array[270,271,272,273,274.1]
        points = pointlists.loc[BPname].to_numpy()[3:-1] 
        Points_I_J = []
        for i in range(len(points)-1):
            Points_I_J.append([points[i],points[-1]])
        #Points_I_J = [[270,1], [271,1], [272,1, [273,1], [274,1]]
        return Points_I_J

    def __makeIJ_XYCoordinates(self):
        """
        获取研究区域的各网格行列坐标（IJ）,平面坐标（XY)和行列数（rows，columns）
        
        Returns:
            ijCoordinates[ndarray] -- [网格点对应的 I J 坐标，二维坐标的压缩方式为从左上角至左下角，遍历列]
            xyCoordinates[ndarray] -- [网格点对应的 X Y 坐标，二维坐标的压缩方式为从左上角至左下角，遍历列]
            row_col_num[list] -- [形式为[rows,columns]]
        """

        csvfile=self.INPUT_FOLDER+ os.sep + "case01/case01_1.csv"
        if not os.path.exists(csvfile):
            csvfile=self.INPUT_FOLDER+ os.sep + "case1/case1_1.csv"
        
        if not os.path.exists(csvfile):
                print(f'{csvfile}ファイルが既存していませんでした。\n')
        else:
            selectIJ=['I','J']
            selectXY=['X','Y']
            # I,J列タイプを指定
            dataframe = pd.read_csv(csvfile, header= 2,dtype={'I':np.int32,'J':np.int32})
            self.ROWS = dataframe['I'].max() #get the rows of the target area
            self.COLUMNS = dataframe['J'].max() #get the columns of the target area
            #xyCoordinates.shape=(27030, 4) with I・J・X・Y
            ijCoordinates =dataframe[selectIJ].to_numpy()
            xyCoordinates =dataframe[selectXY].to_numpy()
            row_col_num=[self.ROWS,self.COLUMNS]
            return ijCoordinates,xyCoordinates,row_col_num

    def __get__Inflow(self,inflowfile=r'../氾濫流量ハイドロ/氾濫ハイドロケース_10分間隔_20200127.xlsx',
                        header=0,sheet_name='氾濫ハイドロパターン (10分間隔)'):
        """
        获取各个工况下的入流纪录
        
            Keyword Arguments:
                inflowfile {xlsxfile}} -- [description] (default: {r'../氾濫流量ハイドロ/氾濫ハイドロケース_10分間隔_20200127.xlsx'})
                header {int} -- [argument for read file with pandas.read_excel] (default: {0})
                sheet_name {str} -- [argument for read file with pandas.read_excel] (default: {'氾濫ハイドロパターン (10分間隔)'})
            
            Returns:
                inflow_DataFrame[pandas.DataFrame] -- [各工况的入流记录DataFrame纪录表，可用DataFrame['case1']提取数据]
        """
        inflow_DataFrame = pd.read_excel(inflowfile,header=header,sheet_name=sheet_name)
        return inflow_DataFrame
      
    def __load_data(self,datapath):
        """
        加载单个csv文件内数据
        
            Arguments:
                datapath {str} -- [文件路径]
            
            Returns:
                data{DataFrame} -- []
        """
        if os.path.exists(datapath):
            data = pd.read_csv(datapath,header = 2)
            return data

    def __select_data(self,data_PD,select_columns =["Depth","Velocity(ms-1)X","Velocity(ms-1)Y"]):
        """
        选取某时刻三通道数据["Depth","Velocity(ms-1)X","Velocity(ms-1)Y"]
        
            Arguments:
                data_PD {DataFrame} -- __load_data 函数的输出
            
            Keyword Arguments:
                select_columns {list} -- 所要提取数据的所在列的标签名 (default: {["Depth","Velocity(ms-1)X","Velocity(ms-1)Y"]})
            
            Returns:
                depth_XY_Velocity {ndarray} -- 所提取数据的array, # shape = (channel,height,width)
        """
        ROWS = self.ROWS
        COLUMNS = self.COLUMNS
        depth_XY_Velocity = []
        for select in select_columns:
            temp_data = data_PD[select].to_numpy()
            temp_data = temp_data.reshape(COLUMNS,ROWS).transpose()# shape = (height,width)
            depth_XY_Velocity.append(temp_data)
        depth_XY_Velocity = np.array(depth_XY_Velocity) #shape = (channel,height,width)
        return depth_XY_Velocity

    def __walk_folder(self,folderpath,filetype = 'csv'):
        """
        遍历当前文件夹，返回文件夹内指定类型文件的路径，范围类型为list
        
            Arguments:
                folderpath {str} -- 指定文件夹的路径
            
            Keyword Arguments:
                filetype {str} -- 要遍历的文件类型 (default: {'csv'})
            
            Returns:
                filepathlist {list} -- 一组文件路径的list,已经按照时间序列进行排序
        """
        filenamelist=[]
        for root, dirs, files in os.walk(folderpath, topdown = False):
            for file in files:
                if file.split(".")[-1] == filetype:
                    filenamelist.append(file ) # case1_10.csv
        # ケース番号の並べ替え 0,1,2,... x[6:] ../case1_10.csv ==>10.csv ==> 10
        filenamelist.sort(key= lambda x:int(x.split('_')[-1][:-4]))

        filepathlist = []
        for filename in filenamelist:
            filepath = root + filename
            filepathlist.append(filepath)
        return filepathlist

    def __datablock(self,case_dir):
        """
        根据case文件夹路径，读取文件夹内所需数据，并以ndarray形式返回
        
            Arguments:
                case_dir {str} -- the path of case directory
            
            Returns:
                datablock_array {ndarray} -- 以ndarray的形式返回当前case的数据块,datablock_array.shape: [N,C,H,W]
            Example:
                >>>datablock_array.shape
                >>>(73,3,100,100)
        """
        filepathlist = self.__walk_folder(case_dir)
        datablock = []
        for filepath in filepathlist:
            temp_DataFrame = self.__load_data(filepath)
            temp_array = self.__select_data(temp_DataFrame)
            datablock.append(temp_array)
        datablock_array = np.asarray(datablock).reshape(-1,3,self.ROWS,self.COLUMNS)
        return datablock_array
        
    def __walk_BP(self):
        """
        遍历BP下面的所有case文件夹，按照数值排序后，对应返回case文件夹名称list和case文件夹路径list
            
            Returns:
                casename_list {list} -- case文件夹名称list
                casepath_list {list} -- case文件夹路径list
        """
        walk_path = self.INPUT_FOLDER
        casename_list=[]
        casepath_list = []
        for root, dirs, files in os.walk(walk_path, topdown = False):
            for dir in dirs:
                casename_list.append(dir)
        casename_list.sort(key = lambda x:int(x[4:])) # case1 ==> 1
        for casename in casename_list:
            casepath_list.append(root +  casename + os.sep) # "../root/case1/"
        return casename_list, casepath_list

    def make_npy_data(self):
        casename_list,casepath_list = self.__walk_BP()
        for i in tqdm(range(len(casename_list))):
            casename = casename_list[i]
            casepath = casepath_list[i]

            case_array = self.__datablock(casepath)

            savename = self.OUTPUT_FOLDER + f"{casename}.npy"
            np.save(savename,case_array)
        self.npyCount = len(casename_list)

        print(f'report : {self.npyCount} npy data in {self.BPname} has been created with shape = {case_array.shape}')



#　鬼怒川計算データからNPYデータを作成する
if __name__ == "__main__":
    BPName_List=['BP018','BP023','BP028','BP033','BP038','BP043'] #ケース番号
    BPName_List=['BP033'] #ケース番号case01、個別処理する。「氾濫ハイドロケース_10分間隔_20200127.xlsx」case1=>case01

    
    for BPNAME in BPName_List:
        INPUT_FOLDER = F'../CasesData/{BPNAME}/'
        OUTPUT_FOLDER = F'../NpyData/{BPNAME}/'
        print(f"\n水理解析結果{BPNAME}のCSVをバイナリNPYファイルに作成中...")
        csv2npy = Csv2Npy(input_folder=INPUT_FOLDER,output_folder= OUTPUT_FOLDER)
        csv2npy.make_npy_data()








