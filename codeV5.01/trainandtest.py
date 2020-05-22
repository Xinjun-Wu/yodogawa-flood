import numpy as np 
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data as Data


from tools_function import check_path,make_image,image2gif
import tools_function as tools






class TrainAndTest():
    def __init__(self, model, path_save):
        self.model = model
        self.path_save = path_save


    def tensor2aray(self,tensor):
        tensor_cpu = tensor.cpu()
        array_cpu = tensor_cpu.numpy()
        return array_cpu

    def traindataloader(self,learning_value,teacher_value,batchsize,shuffle,num_workers=0):
        dataset = Data.TensorDataset(learning_value,teacher_value)
        dataloder = Data.DataLoader(dataset,batch_size=batchsize,
                                    shuffle=shuffle,num_workers=num_workers,pin_memory=True)
        return dataloder

    # def testdataloader(self,learning_value,teacher_value,batchsize,num_workers=0):
    #     dataset = Data.TensorDataset(learning_value,teacher_value)
    #     dataloder = Data.DataLoader(dataset,batch_size=batchsize,
    #                                 num_workers=num_workers)
    #     return dataloder



    def train(self,learn_tensor,teacher_tensor, loss_fn = nn.L1Loss(), epochs = 20, batchsize = 72, lr = 0.001):

        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(' Runing on the GPU...')
        else:
            device = torch.device('cpu')
            print(' Runing on the CPU...')

        model_training = self.model.to(device)
        loss_function = loss_fn.to(device)
        optimizer = optim.Adam(model_training.parameters(), lr = lr, weight_decay = 1e-6)
        lambda1 = lambda epoch: 1/np.sqrt(((epoch % 250)+1.0))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)

        Train_Loss = []

        traindatacontainer = self.traindataloader(learn_tensor,teacher_tensor,
                                     batchsize = 36 ,shuffle = True, num_workers=4)
        
        for epoch in range(epochs):

            if epoch % 10 == 0:
                print(f"The epoch : {epoch} is training...")

            for X_tensor, Y_tensor in traindatacontainer:

                X_input_tensor_gpu = X_tensor.to(device)
                Y_input_tensor_gpu = Y_tensor.to(device)
                #print(device)
            
                model_training.zero_grad()
                Y_output_tensor_gpu = model_training(X_input_tensor_gpu)

                loss = loss_function(Y_output_tensor_gpu,Y_input_tensor_gpu)
                loss.backward()
                optimizer.step()
                scheduler.step()

            Train_Loss.append(loss)
            print(f'Train loss : {loss}')
    
        model_save_path = self.path_save + '/model_save'
        check_path(model_save_path)
        torch.save(model_training, model_save_path + f'/modelv5_{int(time.time())}.pt')

        self.model = model_training
        

        loss_array = np.array(Train_Loss)
        np.savetxt(model_save_path + f'/modelv5_{int(time.time())}_loss.txt',loss_array ,fmt='%10.5f', delimiter= ' ')
        return self   

    


    def test_0(self,learning_tensor,teacher_tensor,step, loss_fn = nn.L1Loss()):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(' Runing on the GPU...')
        else:
            device = torch.device('cpu')
            print(' Runing on the CPU...')

        model_testing = self.model.to(device)
        loss_function = loss_fn.to(device)

        # testdatacontainer = self.testdataloader(learn_tensor,teacher_tensor,
        #                              batchsize = 24,num_workers=8)

        with torch.no_grad():

            # for X_tensor, Y_tensor in testdatacontainer:

            X_tensor=learning_tensor
            Y_tensor=teacher_tensor

            X_input_tensor_gpu = X_tensor.to(device)
            Y_input_tensor_gpu = Y_tensor.to(device)

            Y_output_tensor_gpu = model_testing(X_input_tensor_gpu)

            loss = loss_function(Y_output_tensor_gpu,Y_input_tensor_gpu)

            print(f'Test loss : {loss}')

            Y_input_array = self.tensor2aray(Y_input_tensor_gpu)
            Y_output_array = self.tensor2aray(Y_output_tensor_gpu)

            shape = Y_input_array.shape
            nums = shape[0]
            for j in tqdm(range(nums)):
                image_data_input = Y_input_array[j,0,:,:]
                image_data_output = Y_output_array[j,0,:,:]

                path_save_txt = self.path_save + f'/test_results/txt'
                path_save_image = self.path_save + f'/test_results/image'

                #check path or create path
                check_path(path_save_txt)
                check_path(path_save_image)

                np.savetxt(path_save_txt+f'/{j:02}.csv',image_data_output, delimiter=',',fmt='%10.5f')
                timesite = (j+step)*10
                make_image(data1=image_data_input,data2=image_data_output,size=(9, 3), supertitle = f'Time:{timesite} mins ',
                            outputname = path_save_image+ f'/{j:02}.png')
            image2gif(input_folder = path_save_image,outputname = 'test.gif')
                
            

















            



    # def test_1(self, loss_fn = nn.L1Loss()):

    #     casename_list = tools.get_casename_list(self.path_test_data)

    #     loss_function = loss_fn

    #     with torch.no_grad():
    #         for case in casename_list:

    #             print(f'{case} is testing...')

    #             X_array, Y_array = data_load(casename=case,data_path=self.path_test_data)
    #             flux_path = self.path_test_data + f'/{case}_Xarray_flux.npy'
    #             X_array_flux = np.load(flux_path)

    #             #转换为张量
    #             X_input_tensor_gpu= torch.tensor(X_array, dtype = torch.float).to(device)
    #             Y_target_tensor_gpu= torch.tensor(Y_array, dtype = torch.float).to(device)
    #             X_flux_tensor_gpu = torch.tensor(X_array_flux , dtype = torch.float).to(device)

    #             size = X_input_tensor_gpu.size()
    #             # size[0] = batch = 72
    #             # size[1] = channel = 5
    #             # size[2] = height = 48
    #             # size[3] = weight = 36
            
    #             X_input_tensor_gpu_i = X_input_tensor_gpu[0,:3,:,:].view(1, -1, size[2], size[3])

    #             Test_Loss = []

    #             for i in tqdm(range(0, size[0])):

    #                 if i == 0:
    #                     Y_output_tensor_gpu_i = X_input_tensor_gpu_i


    #                 X_flux_tensor_gpu_i = X_flux_tensor_gpu[i,:,:,:].view(1, -1, size[2], size[3])
    #                 X_input_tensor_gpu_i = torch.cat((Y_output_tensor_gpu_i, X_flux_tensor_gpu_i), dim=1)
    #                 Y_target_tensor_gpu_i = Y_target_tensor_gpu[i,:,:,:].view(1, -1, size[2], size[3])

    #                 Y_output_tensor_gpu_i = self.model(X_input_tensor_gpu_i)
    #                 loss = loss_function(Y_output_tensor_gpu_i,Y_target_tensor_gpu_i)
    #                 Test_Loss.append(loss)

    #                 #convert the tensor on gpu to the array on cpu
    #                 Y_output_tensor_cpu_i = Y_output_tensor_gpu_i.cpu()
    #                 Y_output_array_cpu_i = Y_output_tensor_cpu_i.numpy()

    #                 Y_target_tensor_cpu_i = Y_target_tensor_gpu_i.cpu()
    #                 Y_target_array_cpu_i = Y_target_tensor_cpu_i.numpy()

    #                 #make image
    #                 image_data_input = Y_target_array_cpu_i[0,0,:,:]
    #                 image_data_output = Y_output_array_cpu_i[0,0,:,:]

    #                 path_save_txt = self.path_save + f'/test_results/{case}/txt'
    #                 path_save_image = self.path_save + f'/test_results/{case}/image'

    #                 #check path or create path
    #                 check_path(path_save_txt)
    #                 check_path(path_save_image)

    #                 np.savetxt(path_save_txt+f'/{i:02}.csv',image_data_output, delimiter=',',fmt='%10.5f')
    #                 make_image(data1=image_data_input,data2=image_data_output,size=(9, 3), suptitle = f'Time Interval {i*10}',
    #                            outputname = path_save_image+ f'/{i:02}.png')
                    
    #             break


