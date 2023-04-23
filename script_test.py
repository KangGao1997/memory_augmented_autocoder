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
import pandas as pd
import math
from scipy.fft import fft,fftfreq
import scipy.io as scio


#dict_data = pd.read_pickle('MeasureDict_00269_02543.pickle')
#transmitted = scio.loadmat("Rawdata_data_01800.mat")



#normal_April = np.load("correlated_may_normal.npy")
#abnormal_April = np.load("correlated_may_abnormal.npy")



#normal_data = np.load("correlated_may_normal.npy")

#abnormal_data = np.load("correlated_may_abnormal.npy")

dam1 = 'dam9'
dam2 = 'dam10'
train = np.load(dam1 +'.npy')
test = np.load(dam2 + '.npy')
#dam4 = np.load('dam4.npy')
#dam5 = np.load('dam5.npy')
#dam6 = np.load('dam6.npy')
#dam7 = np.load('dam7.npy')
#dam8 = np.load('dam8.npy')
#dam9 = np.load('dam9.npy')
#dam10 = np.load('dam10.npy')



normal_data = np.vstack([train,test])


print("data loading===================done")
l = normal_data.shape[0]

print("data size")
print(l)
datetime = np.load('damtime.npy',allow_pickle=True)[27540+14920+7140+10960+116100+25020+3560:]#[27540+14920+7140+10960+116100+25020+3560:]
temperature = np.load("damtemp.npy",allow_pickle=True)[27540+14920+7140+10960+116100+25020+3560:]
bright = np.load("dambright.npy", allow_pickle=True)[27540+14920+7140+10960+116100+25020+3560:]
humidity = np.load("damhum.npy", allow_pickle=True)[27540+14920+7140+10960+116100+25020+3560:]

"environment condition"
"""
temp = dict_data['temperature']
pressure = dict_data['pressure']
brightness = dict_data['brightness']
humidity = dict_data['humidity']
"""
MemAE = torch.load("damage9.pth")
#MemAE_rain = torch.load("trained_model_sequence_15comp_April_rain.pth")
#test_data_normal = normal_data
#test_data_abnormal = abnormal_data



test_normal = torch.tensor(normal_data)
#test_abnormal = torch.tensor(abnormal_data)

normal_dataset = TensorDataset(test_normal)
#abnormal_dataset = TensorDataset(test_abnormal)

normal_data_loader = DataLoader(normal_dataset,batch_size=25,shuffle=False)
#abnormal_data_loader = DataLoader(abnormal_dataset,batch_size=25,shuffle=False)


recon_err_nor = np.zeros(test_normal.size(0))
#recon_err_abnor = np.zeros(test_abnormal.size(0))

#te_recon_loss_func = nn.MSELoss(size_average= False)
encode_normal = np.zeros((normal_data.shape[0],15))
#encode_abnormal = np.zeros((abnormal_data.shape[0],15))

decoder_in_normal = np.zeros((normal_data.shape[0],15))
#decoder_in_abnormal = np.zeros((abnormal_data.shape[0],15))

ori_signal = np.zeros((l,2000))
recon_signal_nor = np.zeros((l,2000))
#recon_signal_abnor = np.zeros((l,2000))

#ori_recon_err = np.zeros(test_normal.size(0))
coef = np.zeros(test_normal.shape[0])

print("=====test on normal data=====")
for batch_idx,data in enumerate(normal_data_loader):
    # print(data[0].size())
    recon_res = MemAE(data[0].float())
    # print(recon_res)
    recon_sig = recon_res['output']
    recon_signal_nor[batch_idx * 25: (batch_idx + 1) * 25] = recon_sig.detach().numpy()
    loss = (recon_sig - data[0]).square()
    loss = loss.sum(dim=1) / (2000)
    loss = loss.detach().numpy()
    recon_err_nor[batch_idx * 25:(batch_idx + 1) * 25] = loss
    #coef[batch_idx * 25:(batch_idx + 1) * 25] = np.corrcoef(recon_sig.detach().numpy(),data[0].detach().numpy())
    enco_val = recon_res['encoder_out']
    enco_val = enco_val.detach().numpy()
    encode_normal[batch_idx * 25:(batch_idx + 1) * 25] = enco_val
    denco_val = recon_res['decoder_in']
    denco_val = denco_val.detach().numpy()
    decoder_in_normal[batch_idx * 25:(batch_idx + 1) * 25] = enco_val



final_coef_train = []
final_coef_test = []
final_date_train = []
final_date_test = []
final_temp_train = []
final_temp_test = []
final_bright_train = []
final_bright_test = []
final_humidity_train = []
final_humidity_test = []
for i in range(recon_signal_nor.shape[0]):
    coef[i] = np.corrcoef(recon_signal_nor[i],test_normal[i])[0][1]
    if coef[i]>0.95:
        if i < train.shape[0]:
            final_coef_train.append(coef[i])
            final_date_train.append(datetime[i])
            final_temp_train.append(temperature[i])
            final_bright_train.append(bright[i])
            final_humidity_train.append(humidity[i])

        else:
            final_coef_test.append(coef[i])
            final_date_test.append(datetime[i])
            final_temp_test.append(temperature[i])
            final_bright_test.append(bright[i])
            final_humidity_test.append(humidity[i])





plt.figure(figsize = (15,5))
plt.subplot(4,1,1)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " overall")
plt.scatter(datetime[0:train.shape[0]],coef[0:train.shape[0]],s = 1, label = dam1)
plt.scatter(datetime[train.shape[0]:train.shape[0]+test.shape[0]],coef[train.shape[0]:],s = 1, label = dam2)
#plt.xticks(datetime[0:train.shape[0]+test.shape[0]], rotation = 90)
#plt.ylim((0.95, 1))
plt.legend(loc = 'upper right')
plt.tight_layout()

plt.subplot(4,1,2)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " overall" + " --> humidity")
plt.scatter(datetime[0:train.shape[0]],temperature[0:train.shape[0]],s = 1, label = dam1)
plt.scatter(datetime[train.shape[0]:train.shape[0]+test.shape[0]],humidity[train.shape[0]:train.shape[0]+test.shape[0]],s = 1, label = dam2)
#plt.xticks(datetime[0:train.shape[0]+test.shape[0]], rotation = 90)
plt.legend(loc = 'upper right')
plt.tight_layout()

plt.subplot(4,1,3)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " overall" + " --> temperature")
plt.scatter(datetime[0:train.shape[0]],temperature[0:train.shape[0]],s = 1, label = dam1)
plt.scatter(datetime[train.shape[0]:train.shape[0]+test.shape[0]],temperature[train.shape[0]:train.shape[0]+test.shape[0]],s = 1, label = dam2)
#plt.xticks(datetime[0:train.shape[0]+test.shape[0]], rotation = 90)
#plt.yticks(np.arange(10,40,1))
plt.legend(loc = 'upper right')
plt.tight_layout()

plt.subplot(4,1,4)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " overall" + " --> bright")
plt.scatter(datetime[0:train.shape[0]],bright[0:train.shape[0]],s = 1, label = dam1)
plt.scatter(datetime[train.shape[0]:train.shape[0]+test.shape[0]],bright[train.shape[0]:train.shape[0]+test.shape[0]],s = 1, label = dam2)
#plt.xticks(datetime[0:train.shape[0]+test.shape[0]], rotation = 90)
plt.legend(loc = 'upper right')
plt.tight_layout()



plt.savefig(dam1+ "_" + dam2+ "_" + 'all3_scatter.png')
plt.close()




plt.figure(figsize = (15,5))
plt.subplot(4,1,1)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out")
plt.scatter(final_date_train,final_coef_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_coef_test,s = 1, label = dam2)
plt.ylim((0.95, 1))
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(0.7,1,0.1))
plt.legend(loc = 'upper right')
plt.tight_layout()

plt.subplot(4,1,2)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out --> humidity")
plt.scatter(final_date_train,final_humidity_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_humidity_test,s = 1, label = dam2)
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(0.7,1,0.1))
plt.legend(loc = 'upper right')
plt.tight_layout()

#plt.figure(figsize = (15,5))
plt.subplot(4,1,3)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out --> temperature")
plt.scatter(final_date_train,final_temp_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_temp_test,s = 1, label = dam2)
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(10,40,1))
plt.legend(loc = 'upper right')
plt.tight_layout()

#plt.figure(figsize = (15,5))
plt.subplot(4,1,4)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out--> bright")
plt.scatter(final_date_train,final_bright_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_bright_test,s = 1, label = dam2)
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(0.7,1,0.1))
plt.legend(loc = 'upper right')
plt.tight_layout()

#plt.show()
plt.savefig(dam1+ "_" + dam2+ "_" +'_ruleout3_scatter.png')
plt.close()


final_coef_train = []
final_coef_test = []
final_date_train = []
final_date_test = []
final_temp_train = []
final_temp_test = []
final_bright_train = []
final_bright_test = []
final_humidity_train = []
final_humidity_test = []
for i in range(recon_signal_nor.shape[0]):
    coef[i] = np.corrcoef(recon_signal_nor[i],test_normal[i])[0][1]
    if humidity[i]< (np.mean(humidity)+np.std(humidity)/4) and coef[i]>0.95:
        if i < train.shape[0]:
            final_coef_train.append(coef[i])
            final_date_train.append(datetime[i])
            final_temp_train.append(temperature[i])
            final_bright_train.append(bright[i])
            final_humidity_train.append(humidity[i])
        else:
            final_coef_test.append(coef[i])
            final_date_test.append(datetime[i])
            final_temp_test.append(temperature[i])
            final_bright_test.append(bright[i])
            final_humidity_test.append(humidity[i])


plt.figure(figsize = (15,5))
plt.suptitle("get all smaller 40")
plt.subplot(4,1,1)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out")
plt.scatter(final_date_train,final_coef_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_coef_test,s = 1, label = dam2)
plt.ylim((0.95, 1))
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(0.7,1,0.1))
plt.legend(loc = 'upper right')
plt.tight_layout()

plt.subplot(4,1,2)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out --> humidity")
plt.scatter(final_date_train,final_humidity_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_humidity_test,s = 1, label = dam2)
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(0.7,1,0.1))
plt.legend(loc = 'upper right')
plt.tight_layout()



#plt.figure(figsize = (15,5))
plt.subplot(4,1,3)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out --> temperature")
plt.scatter(final_date_train,final_temp_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_temp_test,s = 1, label = dam2)
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(10,40,1))
plt.legend(loc = 'upper right')
plt.tight_layout()

#plt.figure(figsize = (15,5))
plt.subplot(4,1,4)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out--> bright")
plt.scatter(final_date_train,final_bright_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_bright_test,s = 1, label = dam2)
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(0.7,1,0.1))
plt.legend(loc = 'upper right')
plt.tight_layout()

plt.savefig(dam1+ "_" + dam2+ "_" +'_ruleout3_scatter_humidity.png')
plt.close()

final_coef_train = []
final_coef_test = []
final_date_train = []
final_date_test = []
final_temp_train = []
final_temp_test = []
final_bright_train = []
final_bright_test = []
final_humidity_train = []
final_humidity_test = []
for i in range(recon_signal_nor.shape[0]):
    coef[i] = np.corrcoef(recon_signal_nor[i],test_normal[i])[0][1]
    if humidity[i]<40 and temperature[i]>(np.mean(temperature) - np.std(temperature)) and coef[i]>0.95:
        if i < train.shape[0]:
            final_coef_train.append(coef[i])
            final_date_train.append(datetime[i])
            final_temp_train.append(temperature[i])
            final_bright_train.append(bright[i])
            final_humidity_train.append(humidity[i])
        else:
            final_coef_test.append(coef[i])
            final_date_test.append(datetime[i])
            final_temp_test.append(temperature[i])
            final_bright_test.append(bright[i])
            final_humidity_test.append(humidity[i])
plt.figure(figsize = (15,5))
plt.suptitle("get all humidity smaller than 40 and temperature larger than mean - std")
plt.subplot(4,1,1)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out")
plt.scatter(final_date_train,final_coef_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_coef_test,s = 1, label = dam2)
plt.ylim((0.95, 1))
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(0.7,1,0.1))
plt.legend(loc = 'upper right')
plt.tight_layout()

plt.subplot(4,1,2)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out --> humidity")
plt.scatter(final_date_train,final_humidity_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_humidity_test,s = 1, label = dam2)
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(0.7,1,0.1))
plt.legend(loc = 'upper right')
plt.tight_layout()

#plt.figure(figsize = (15,5))
plt.subplot(4,1,3)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out --> temperature")
plt.scatter(final_date_train,final_temp_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_temp_test,s = 1, label = dam2)
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(10,40,1))
plt.legend(loc = 'upper right')
plt.tight_layout()

#plt.figure(figsize = (15,5))
plt.subplot(4,1,4)
plt.title("training: " + dam1 +',' + "testing: " + dam2 + " rule out--> bright")
plt.scatter(final_date_train,final_bright_train,s = 1, label = dam1)
plt.scatter(final_date_test,final_bright_test,s = 1, label = dam2)
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(0.7,1,0.1))
plt.legend(loc = 'upper right')
plt.tight_layout()
#plt.xticks(np.array(final_date_train+final_date_test),rotation = 90)
#plt.yticks(np.arange(0.7,1,0.1))
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.savefig(dam1+"_"+dam2+"_"+'ruleout3_scatter_humidity_temperature.png')
plt.close()

"""

partial_recon = recon_err_nor[0:dam3.shape[0]+dam4.shape[0]+dam5.shape[0]]
color = [[0.3,0.3,0.3],[0.5,0.5,0.5],[0.7,0.7,0.7]]
normalized_err3 = []
for i in range(dam3.shape[0]):
    if partial_recon[i]<0.003:
        normalized_err3.append(partial_recon[i])
normalized_err4 = []
for i in range(dam3.shape[0],dam3.shape[0]+dam4.shape[0]):
    if partial_recon[i]<0.003:
        normalized_err4.append(partial_recon[i])
normalized_err5 = []
for i in range(dam3.shape[0]+dam4.shape[0],dam3.shape[0]+dam4.shape[0]+dam5.shape[0]):
    if partial_recon[i]<0.003:
        normalized_err5.append(partial_recon[i])




normalized_err3 = np.array(normalized_err3)
normalized_err4 = np.array(normalized_err4)
normalized_err5 = np.array(normalized_err5)
"""

"""
plt.figure(figsize=(15,4))
plt.plot(np.arange(0,normalized_err3.shape[0]),normalized_err3,color = 'b')
plt.plot(np.arange(normalized_err3.shape[0],normalized_err3.shape[0]+normalized_err4.shape[0]),normalized_err4,color = 'r')
plt.plot(np.arange(normalized_err3.shape[0]+normalized_err4.shape[0],normalized_err3.shape[0]+normalized_err4.shape[0]+normalized_err5.shape[0]),normalized_err5,color = 'y')
"""

"""
print("=====test on abnormal data=====")

for batch_idx,data in enumerate(abnormal_data_loader):
    #print(data[0].size())
    recon_res = MemAE(data[0].float())
    recon_sig = recon_res['output']
    recon_signal_abnor[batch_idx*25 : (batch_idx+1)*25] = recon_sig.detach().numpy()
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
"""


"""======================================compute the coefficient==========================="""



"""=========================================plot section===================================="""
"""
plt.figure(figsize=(15,4))
plt.title("reconstruction of all damage")
plt.plot(np.arange(0,dam3.shape[0]),recon_err_nor[0:dam3.shape[0]],color = 'b',label = 'damage level 3')
plt.plot(np.arange(dam3.shape[0],dam3.shape[0]+dam4.shape[0]),recon_err_nor[dam3.shape[0]:dam3.shape[0]+dam4.shape[0]],color = 'r',label = 'damage level 4')
plt.plot(np.arange(dam3.shape[0]+dam4.shape[0],dam3.shape[0]+dam4.shape[0]+dam5.shape[0]),recon_err_nor[dam3.shape[0]+dam4.shape[0]:dam3.shape[0]+dam4.shape[0]+dam5.shape[0]],color = 'g',label = 'damage level 5')
plt.plot(np.arange(dam3.shape[0]+dam4.shape[0]+dam5.shape[0],dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]),recon_err_nor[dam3.shape[0]+dam4.shape[0]+dam5.shape[0]:dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]],color = 'c',label = 'damage level 6')
plt.plot(np.arange(dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0],
                   dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]),
         recon_err_nor[dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]:dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]],color = 'm',label = 'damage level 7')
plt.plot(np.arange(dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0],
                   dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]+dam8.shape[0]),
         recon_err_nor[dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]:dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]+dam8.shape[0]],color = [0.5,0.5,0.5],label = 'damage level 8')
plt.plot(np.arange(dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]+dam8.shape[0],
                   dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]+dam8.shape[0]+dam9.shape[0]),
         recon_err_nor[dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]+dam8.shape[0]:dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]+dam8.shape[0]+dam9.shape[0]],color = 'y',label = 'damage level 9')
plt.plot(np.arange(dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]+dam8.shape[0]+dam9.shape[0],
                   dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]+dam8.shape[0]+dam9.shape[0]+dam10.shape[0]),
         recon_err_nor[dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]+dam8.shape[0]+dam9.shape[0]:dam3.shape[0]+dam4.shape[0]+dam5.shape[0]+dam6.shape[0]+dam7.shape[0]+dam8.shape[0]+dam9.shape[0]+dam10.shape[0]],color = [0.8,0.8,0.8],label = 'damage level 10')

plt.legend(fontsize=6)
"""
"""
plt.figure(figsize=(10,8))
#plt.title("reconstruction MSE")
plt.plot(recon_err_nor,color = 'b')
#plt.plot(recon_err_abnor,color = 'r')
#plt.subplots_adjust(hspace=1, wspace=0.4)
#plt.tight_layout()
"""

"""
plt.figure(figsize=(10,8))
plt.subplot(3,3,1)
plt.title("normal signal at 100000")
plt.plot(fftfreq(2000),(abs(fft(normal_data[100000]))))
plt.subplot(3,3,2)
plt.title("normal signal at 500000")
plt.plot(fftfreq(2000),(abs(fft(normal_data[50000]))))
plt.subplot(3,3,3)
plt.title("difference")
plt.plot(fftfreq(2000),(abs(fft(normal_data[100000])) - abs(fft(normal_data[50000]))))
plt.tight_layout()

plt.subplot(3,3,4)
plt.title("normal signal at 100000")
plt.plot(fftfreq(2000),(abs(fft(normal_data[100000]))))
plt.subplot(3,3,5)
plt.title("rainfall at 110000")
plt.plot(fftfreq(2000),(abs(fft(normal_data[110000]))))
plt.subplot(3,3,6)
plt.title("difference")
plt.plot(fftfreq(2000),(abs(fft(normal_data[100000])) - abs(fft(normal_data[110000]))))
plt.tight_layout()

plt.subplot(3,3,7)
plt.title("normal signal at 100000")
plt.plot(fftfreq(2000),(abs(fft(normal_data[100000]))))
plt.subplot(3,3,8)
plt.title("damage at 100000")
plt.plot(fftfreq(2000),(abs(fft(abnormal_data[100000]))))
plt.subplot(3,3,9)
plt.title("difference")
plt.plot(fftfreq(2000),(abs(fft(normal_data[100000])) - abs(fft(abnormal_data[100000]))))
plt.tight_layout()
"""
"""
plt.subplot(7,1,2)
plt.title("reconstruction coefficient")
plt.plot(dict_data['datatime'][286289+309553:286289+309553+len],recon_co,color = 'b')
#plt.plot(recon_err_abnor,color = 'r')
plt.subplots_adjust(hspace=1, wspace=0.4)
plt.tight_layout()
"""
"""
plt.subplot(7,1,3)
plt.title("MSE in kernel space")
plt.plot(dict_data['datatime'][286289+309553:286289+309553+len],ker_mse,color = 'b')
#plt.plot(recon_err_abnor,color = 'r')
plt.subplots_adjust(hspace=1, wspace=0.4)
plt.tight_layout()

"""


plt.show()