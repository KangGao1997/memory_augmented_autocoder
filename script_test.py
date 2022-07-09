import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import scipy.io as sio
import time
from model import AutoEncoderConv1D
from models import EntropyLossEncap
from torch.nn import functional as F
import matplotlib.pyplot as plt

normal_April = np.load("correlated_June_normal.npy")
abnormal_April = np.load("correlated_June_abnormal.npy")

normal_data = normal_April
abnormal_data = abnormal_April

"""
normal_May = np.load("correlated_May_normal.npy")
abnormal_May = np.load("correlated_May_abnormal.npy")

normal_June = np.load("correlated_June_normal.npy")
abnormal_June = np.load("correlated_June_abnormal.npy")

normal_data = np.zeros((3000,2000))
abnormal_data = np.zeros((3000,2000))


normal_data[0:1000] = normal_April[10000:11000]
normal_data[1000:2000] = normal_May[8000:9000]
normal_data[2000:3000] = normal_June[10000:11000]

abnormal_data[0:1000] = abnormal_April[10000:11000]
abnormal_data[1000:2000] = abnormal_May[8000:9000]
abnormal_data[2000:3000] = abnormal_June[10000:11000]
"""

MemAE = torch.load("trained_model_sequence.pth")

test_data_normal = normal_data
test_data_abnormal = abnormal_data

test_normal = torch.tensor(test_data_normal)
test_abnormal = torch.tensor(test_data_abnormal)

normal_dataset = TensorDataset(test_normal)
abnormal_dataset = TensorDataset(test_abnormal)

normal_data_loader = DataLoader(normal_dataset,batch_size=25,shuffle=False)
abnormal_data_loader = DataLoader(abnormal_dataset,batch_size=25,shuffle=False)


recon_err_nor = np.zeros(test_normal.size(0))
recon_err_abnor = np.zeros(test_abnormal.size(0))

#te_recon_loss_func = nn.MSELoss(size_average= False)
encode_normal = np.zeros((normal_data.shape[0],3))
encode_abnormal = np.zeros((abnormal_data.shape[0],3))

decoder_in_normal = np.zeros((normal_data.shape[0],3))
decoder_in_abnormal = np.zeros((abnormal_data.shape[0],3))

print("=====test on normal data=====")
for batch_idx,data in enumerate(normal_data_loader):
    #print(data[0].size())
    recon_res = MemAE(data[0].float())
    #print(recon_res)
    recon_sig = recon_res['output']
    loss = (recon_sig - data[0]).square()
    loss = loss.sum(dim = 1)/(2000)
    loss = loss.detach().numpy()
    recon_err_nor[batch_idx*25:(batch_idx+1)*25] = loss
    enco_val = recon_res['encoder_out']
    enco_val = enco_val.detach().numpy()
    encode_normal[batch_idx*25:(batch_idx+1)*25] = enco_val
    denco_val = recon_res['decoder_in']
    denco_val = denco_val.detach().numpy()
    decoder_in_normal[batch_idx * 25:(batch_idx + 1) * 25] = enco_val


print("=====test on abnormal data=====")

for batch_idx,data in enumerate(abnormal_data_loader):
    #print(data[0].size())
    recon_res = MemAE(data[0].float())
    recon_sig = recon_res['output']
    loss = (recon_sig - data[0]).square()
    loss = loss.sum(dim = 1)/(2000)
    loss = loss.detach().numpy()
    recon_err_abnor[batch_idx*25:(batch_idx+1)*25] = loss
    enco_val = recon_res['encoder_out']
    enco_val = enco_val.detach().numpy()
    encode_abnormal[batch_idx * 25:(batch_idx + 1) * 25] = enco_val
    denco_val = recon_res['decoder_in']
    denco_val = denco_val.detach().numpy()
    decoder_in_abnormal[batch_idx * 25:(batch_idx + 1) * 25] = enco_val



plt.figure()
plt.plot(recon_err_nor,color = 'b')
plt.plot(recon_err_abnor,color = 'r')

plt.figure(figsize=(12.8,4.8))
plt.pcolor(np.transpose(encode_normal))
plt.title("encode results of normal data in June")
plt.yticks([0,1,2])
plt.colorbar()

plt.figure(figsize=(12.8,4.8))
plt.pcolor(np.transpose(encode_abnormal))
plt.title("encode results of abnormal data in June")
plt.yticks([0,1,2])
plt.colorbar()

plt.figure(figsize=(12.8,4.8))
plt.pcolor(np.transpose(decoder_in_normal))
plt.title("input  of decoder(normal data in June)")
plt.yticks([0,1,2])
plt.colorbar()

plt.figure(figsize=(12.8,4.8))
plt.pcolor(np.transpose(decoder_in_abnormal))
plt.title("input  of decoder(abnormal data in June)")
plt.yticks([0,1,2])
plt.colorbar()

plt.figure(figsize=(12.8,4.8))
plt.pcolor((np.transpose(encode_normal - encode_abnormal)))
plt.title("difference between normal data and abnormal data encode results(June)")
plt.yticks([0,1,2])
plt.colorbar()

plt.figure(figsize=(12.8,4.8))
plt.pcolor(np.transpose(decoder_in_abnormal - decoder_in_abnormal))
plt.title("difference of input of decoder between normal data and abnormal data in June")
plt.yticks([0,1,2])
plt.colorbar()



plt.show()