import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import scipy.io as sio
from scipy.fft import fft,fftfreq
import time
from model import AutoEncoderConv1D
from models import EntropyLossEncap
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time


#normal_April = np.load("correlated_April_normal.npy")

#abnormal_April = np.load("correlated_April_abnormal.npy")

#normal_May = np.load("correlated_May_normal.npy")
#abnormal_May = np.load("correlated_May_abnormal.npy")

#normal_June = np.load("correlated_June_normal.npy")
#abnormal_June = np.load("correlated_June_abnormal.npy")

#trainsmitted_signal = np.load("transmitted_signal.npy")


dam1 = np.load("dam5.npy")
dam2 = np.load("dam6.npy")

ori_signal = np.vstack([dam1,dam2[0:1096]])
l = ori_signal.shape[0]

toTensor_sig = torch.tensor(ori_signal)

train_data_set = TensorDataset(toTensor_sig)

lr = 0.0001
batch_size = 32
mem_dim = 3
shrink_thres = 0.0025
Epoch = 40


MemAE = AutoEncoderConv1D(mem_dim=2000,shrink_thres=shrink_thres)
MemAE = MemAE.float()



tr_recon_loss_func = nn.MSELoss()                                 #is the cross entropy resonable? what about test stage?
tr_entropy_loss_func = EntropyLossEncap()
tr_optimizer = torch.optim.Adam(MemAE.parameters(),lr = lr)





tr_data_loader = DataLoader(train_data_set,batch_size = batch_size, shuffle = True)


learning_curve = np.zeros(Epoch)
for epoch_idx in range(Epoch):
    print("===============epoch=====================")
    print(epoch_idx+1)
    cur_MSE = 0
    for batch_idx, data in enumerate(tr_data_loader):
        recon_res = MemAE(data[0].float())                       #data is single signal or single signal mulply the batch size?
        recon_sig = recon_res['output']
        att_w = recon_res['att']
        loss = tr_recon_loss_func(recon_sig,data[0].float())
        recon_loss_val = loss.item()
        entropy_loss = tr_entropy_loss_func(att_w)
        entropy_loss_val = entropy_loss.item()
        loss = loss + 0.002*entropy_loss
        loss_val = loss.item()
        #print(loss_val)
        tr_optimizer.zero_grad()
        loss.backward()
        tr_optimizer.step()
        cur_MSE += loss_val
    learning_curve[epoch_idx] = cur_MSE/batch_size      #really should be divided by batch size????

plt.figure()
plt.plot(learning_curve)
plt.show()


torch.save(MemAE,'damage5_6_101.pth')







