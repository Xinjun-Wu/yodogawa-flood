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
import time
import datetime

from CNN_models import ConvNet_2
from tools import check_path,make_image,image2gif,plot_loss,plot_loss_by_time,format_loss_list
from dataSet import KinugawaDataSets

# 訓練・テスト用クラス
class TrainAndTest():
    # クラスの初期化
    def __init__(self, model, bpname, step, check_Point = None):
        
        if check_Point is None:
            self.checkOrNot = False
            self.model = model
            self.bpname = bpname
            self.step = step
            self.path = f"../save/{self.bpname}/TimeStep_{self.step}/"
            
        else:
            self.checkOrNot = True
            self.model = model
            self.bpname = check_Point[0]
            self.step = check_Point[1]*6
            self.beginEOPCH = check_Point[2]
            self.path = f"../save/{self.bpname}/TimeStep_{self.step}/"
            #F:\ArcGIS\Flood\kinugawa\save\BP028\TimeStep_6\model_save
            savedModel=self.path + f"model_save/modelv8_epoch_{self.beginEOPCH}.pt"
            savedModel_=self.path + f"model_save/modelv6_epoch_{self.beginEOPCH}.pt"
            if os.path.exists(savedModel):
                checkpoint = torch.load(savedModel)
                self.checkpoint = checkpoint
            elif os.path.exists(savedModel_):
                    checkpoint = torch.load(savedModel_)
                    self.checkpoint = checkpoint

            else:
                raise ValueError("指定的checkpoint不存在")

    # Tensor ==> Numpy
    def tensor2array(self,tensor):
        tensor_cpu = tensor.cpu()
        array_cpu = tensor_cpu.numpy()
        return array_cpu

    # 任意EPOCHから訓練用関数
    def train(self, dataset, loss_fn = nn.L1Loss(), epochs = 500, batchsize = 72, lr = 0.001, transfer = False ):

        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(' Runing on the GPU...')
        else:
            device = torch.device('cpu')
            print(' Runing on the CPU...')
        
        ########################### I N I T I A L I Z A T I O N   ####################################
        train_Loss = [] #訓練損失リストの記録
        start_epoch = 0
        lr_List = []

        model_training = self.model.to(device)  #モデルをGPUへ
        optimizer = optim.Adam(model_training.parameters(), lr = lr, weight_decay = 1e-6)
        loss_function = loss_fn.to(device)
        lambda1 = lambda epoch: 1/np.sqrt(((epoch % 500)+1.0))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)
        
        ################################# L O A D I N G ############################################
        if self.checkOrNot is True:
            if transfer is True:
                model_training.load_state_dict(self.checkpoint['model'])
                #lr_List.append([1,lr])
                print('使用checkpoint处的模型进行迁移学习，迁移过程中只保留以往模型的权重')

            else:
                model_training.load_state_dict(self.checkpoint['model'])
                optimizer.load_state_dict(self.checkpoint['optimizer'])
                start_epoch = self.checkpoint['epoch']
                train_Loss=self.checkpoint['trainLoss']
                train_Loss = format_loss_list(train_Loss)
                if 'lrList' in self.checkpoint.keys():
                    lr_List = self.checkpoint['lrList']
                print(f'EPOCH={start_epoch}を続けて、CNNモデルを学習中...\n')
                if 'scheduler' in self.checkpoint.keys():
                    scheduler.load_state_dict(self.checkpoint['scheduler'])
                else:
                    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1,last_epoch = start_epoch)

        model_training.train() # 训练模式
        dataset_value=dataset.set_select('train')
        ###################################### E P O C H  L O O P ###################################
        for epoch in range(start_epoch+1,epochs+1):
            start_clock = int(time.time())
            ###################################### T R A I N #########################################
            lr_temp = optimizer.state_dict()['param_groups'][0]['lr']
            lr_List.append([epoch,lr_temp]) #提取接下来训练模型的学习率
            #学習データの準備
            datacontainer = Data.DataLoader(dataset=dataset_value, batch_size = batchsize ,shuffle = True, num_workers=8, pin_memory=True)
            epoch_loss = []#纪录一个epoch内loss
            for X_tensor, Y_tensor in datacontainer:

                X_input_tensor_gpu = X_tensor.to(device,dtype=torch.float32,non_blocking=True)
                Y_input_tensor_gpu = Y_tensor.to(device,dtype=torch.float32,non_blocking=True)
                
                model_training.zero_grad()
                Y_output_tensor_gpu = model_training(X_input_tensor_gpu)

                loss = loss_function(Y_output_tensor_gpu,Y_input_tensor_gpu)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            scheduler.step()
            t_loss = np.mean(epoch_loss)#对一个epoch内的loss取平均
            train_Loss.append([epoch,t_loss]) #訓練損失の記録
            print(f'BPName = {self.bpname}, Hours = {int(self.step/6)}, Epoch = {epoch}, Loss = {t_loss}, Lr = {lr_temp} ')
            
            
            end_clock = int(time.time())
            start = datetime.timedelta(seconds=start_clock)
            end = datetime.timedelta(seconds=end_clock)
            timeusage = str(end - start)

            print(f"epoch {epoch} :time usage {timeusage}")

            ########################################## S A V E #######################################
            #EPOCH100毎にモデルを保存
            if epoch % 10 == 0 or epoch == epochs : # 逢百和最后一个epoch保持模型

                state = {'model':model_training.state_dict(), 'optimizer':optimizer.state_dict(), 
                            'scheduler':scheduler.state_dict(),
                            'epoch':epoch,'trainLoss':train_Loss, 'lrList': lr_List}

                model_save_path = f"../save/{self.bpname}/TimeStep_{self.step}/model_save/"
                check_path(model_save_path) #フォルダ既存の確認
                modelFilename=model_save_path+f'/modelv8_epoch_{epoch}.pt'
                torch.save(state, modelFilename)

            if epoch % 200 == 0 or epoch == epochs :
                #損失をCSVファイルに出力
                modelFilename=model_save_path+f'/modelv8_epoch_{epoch}.csv'
                train_Loss_array = np.array(train_Loss)
                np.savetxt(modelFilename,train_Loss_array,fmt='%10.6f', delimiter= ',', header='EPOCH,TRAIN LOSS')

                #損失グラフをPNGファイルに出力
                modelFilename=model_save_path+f'/modelv8_epoch_{epoch}.png'
                lr_array = np.array(lr_List)
                plot_loss(modelFilename,train_Loss_array,lr_array, xlim = (0,epochs))
        
        self.model = model_training


    # テスト用関数
    def test(self, dataset, loss_fn = nn.L1Loss(), timeinterval=10, step = None,model_epoch = None,threshold_loss = 0.05):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(' Testing on the GPU...')
        else:
            device = torch.device('cpu')
            print(' Testing on the CPU...')

        ########################### I N I T I A L I Z A T I O N   ####################################
        loss_function = loss_fn.to(device)
        model_testing = self.model.to(device)

        ################################# L O A D I N G ############################################
        model_testing.load_state_dict(self.checkpoint['model'])

        XYCoornpzFile=f'../NpyData/{self.bpname}_info.npz'
        if not os.path.exists(XYCoornpzFile):
            print(f'{XYCoornpzFile}が既存していませんでした。\n')
            xyCoorArray = None
        else:
            xyCoorArray=np.load(XYCoornpzFile)  #NPZファイルの読み込み
            xyCoordinatesArray=xyCoorArray['xyCoordinates'] # GISのXとY平面座標
            rows,columns=xyCoorArray['row_col_num'] #高さ・幅の取得

        ###################################### T E S T #########################################
        model_testing.eval() # 评估模式
        dataset_value = dataset.set_select('test')
        case_list = dataset.index ##进行测试的case的序号
        n_case  = len(case_list) ##进行测试的case个数
        length_case = dataset.length_case # 当前step下每个case所包含的样本数
        input_plot_list = []
        output_plot_list = []
        test_Loss_with_time = []#纪录case的loss随时间的变化
        with torch.no_grad():
            test_datacontainer = Data.DataLoader(dataset_value, batch_size = 1, shuffle = False, num_workers = 4)
            i = 0
            for X_tensor, Y_tensor in test_datacontainer:
                

                X_input_tensor_gpu = X_tensor.to(device)
                Y_input_tensor_gpu = Y_tensor.to(device)
                
                model_testing.zero_grad()
                Y_output_tensor_gpu = model_testing(X_input_tensor_gpu)

                loss = loss_function(Y_output_tensor_gpu,Y_input_tensor_gpu)

                Y_input_array = self.tensor2array(Y_input_tensor_gpu)
                Y_output_array = self.tensor2array(Y_output_tensor_gpu)
                input_plot_list.append(Y_input_array)
                output_plot_list.append(Y_output_array)
            
                test_Loss_with_time.append([self.step+i+1,loss.item()])
                i += 1
                if self.step+i+1 > 73: #当超过case的时间序列的最大值时，进行归零
                    i = 0

        #convert list to array
        input_plot_array = np.array(input_plot_list).reshape(-1,3,rows,columns)
        output_plot_array = np.array(output_plot_list).reshape(-1,3,rows,columns)
        test_Loss_with_time_array = np.array(test_Loss_with_time).reshape(-1,length_case, 2)

        ###################################### S A V E L O O P #########################################
        case_name_list = []
        test_Loss_average_list = []
        for n in range(n_case):

            case_index = case_list[n] #返回 case1 中的 1 
            case_name = f'case {case_index}' # 'case 1 '
            case_name_list.append(case_name)

            test_Loss_array =  test_Loss_with_time_array[n]
            loss_average = sum(test_Loss_array[:,1])/length_case
            test_Loss_average_list.append([case_name,loss_average])




            
            print(f'The loss of {case_name} : {loss_average}')


            start = n*length_case # case1的每一时刻的数据在 array 里的起止点的index
            end = (n+1)*length_case

            input_array = input_plot_array[start:end] # case1的输入数据
            output_array = output_plot_array[start:end] # case1的预测数据
            difference_array = input_array - output_array
            ###################################### SAVE EARCH CASE DATA #########################################
            if loss_average < threshold_loss:

                print(f'saving results for {case_name} with loss = {loss_average}')

                path_save = self.path + f'test_results_e{self.beginEOPCH}/test_case{case_index}/'
                path_save_csv_all = path_save+ 'csv_all/'
                path_save_csv_depth = path_save + 'csv_depth/'
                path_save_csv_xflow = path_save + 'csv_xflow/'
                path_save_csv_yflow = path_save + 'csv_yflow/'
                path_save_image = path_save + 'image/'

                path_list = [path_save_csv_depth,path_save_csv_xflow,path_save_csv_yflow]

                #check path or create path
                check_path(path_save)
                check_path(path_save_csv_all)
                check_path(path_save_csv_depth)
                check_path(path_save_csv_xflow)
                check_path(path_save_csv_yflow)
                check_path(path_save_image)
                
                for j in tqdm(range(length_case)): #遍历每一个时刻
                    if xyCoorArray is not None:
                        with open(path_save_csv_all+f'/{self.step + j + 1:02}.csv','w',newline='') as csvfile:
                            writer=csv.writer(csvfile)
                            head=['I','J','X','Y',
                                    'Depth_Input','XFlow_Input','YFlow_Input',
                                    'Depth_Output','XFlow_Output','YFlow_Output',
                                    'Input-Output(Depth)','Input-Output(XFlow)','Input-Output(YFlow)']
                            writer.writerow(head)
                            index=0
                            for col in range(columns): #遍历每一列
                                for row in range(rows): #遍历每一行
                                    writer.writerow([f'{row+1}',f'{col+1}',
                                        xyCoordinatesArray[index,0],xyCoordinatesArray[index,1],
                                        input_array[j,0,row,col],input_array[j,1,row,col],input_array[j,2,row,col],
                                        output_array[j,0,row,col],output_array[j,1,row,col],output_array[j,2,row,col],
                                        difference_array[j,0,row,col],difference_array[j,1,row,col],difference_array[j,2,row,col]])
                                    index +=1 
                    for i in range(3):
                        np.savetxt(path_list[i]+f'/{j+self.step+1:02}.csv',output_array[j,i,:,:], delimiter=',',fmt='%10.6f')

                    timestamp = (j+self.step)*timeinterval
                    make_image(target_array=input_array[j],predicted_array=output_array[j],height=rows,width=columns, 
                                supertitle = f'{j+self.step+1:02}_ Time:{timestamp} mins with error : {test_Loss_array[j]}',
                                channeltitle_list = ['Waterdepth','X Velocity(ms-1)', 'Y Velocity(ms-1)'],
                                outputname = path_save_image+ f'{j+self.step+1:02}.png')

                image2gif(input_folder = path_save_image,outputname = 'test.gif')

        return test_Loss_average_list


    #             image2gif(input_folder = path_save_image,outputname = 'test.gif')

 
        

#　鬼怒川各ケースにより、一括でモデルを学習する
if __name__ == "__main__": 
    EPOCHS = 2000
    BATCHSIZE = 36
    LR = 0.005
    LOSS_FN_TRAIN = nn.L1Loss()
    # LOSS_FN_TRAIN = nn.MSELoss()
    BPName_List=['BP008','BP018','BP023','BP028','BP033','BP038','BP043'] #ケース番号
    BPName_List=['BP033','BP023','BP028','BP033','BP038','BP043'] #ケース番号
    
    CHECK_POINT = ['BP033',2, 330]
    CHECK_POINT = None

###################################################################################################
    if CHECK_POINT is not None:  #根据检查点更新BPNAME_LIST
        START_BPNAME = CHECK_POINT[0]
        START_BPNAME_INDEX = BPName_List.index(START_BPNAME)
        BPName_List = BPName_List[START_BPNAME_INDEX:]
        if isinstance(BPName_List, str):
            BPName_List = [BPName_List]

    for BPNAME in BPName_List:

        if CHECK_POINT is not None:#根据检查点更新START_HOUR
            START_HOUR = CHECK_POINT[1]
        else:
            START_HOUR = 2
        for hour in range(START_HOUR,7):

            if CHECK_POINT is not None:
                CHECK_POINT[1] = hour

            step = hour * 6
            kinugawaData = KinugawaDataSets(bpname = BPNAME,timestep = step,
                            split_pattern = 'specific',specific = (-1,2,6), fulltrain=True) # 初始化数据集
            
            net = ConvNet_2(hour+3) #实例化网络
            #初始化训练器
            trainandtest = TrainAndTest(model = net,bpname = BPNAME,step = step,check_Point=CHECK_POINT) 
            #开始训练
            trainandtest.train(dataset = kinugawaData,loss_fn = LOSS_FN_TRAIN, epochs = EPOCHS,
                                batchsize = BATCHSIZE,lr = LR,transfer = False)
            ####################################################################################################
            CHECK_POINT = None

                



