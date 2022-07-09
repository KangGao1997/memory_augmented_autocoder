import torch
from torch import nn

from models import MemModule

torch.set_default_dtype(torch.float64)
class AutoEncoderConv1D(nn.Module):
    def __init__(self,mem_dim,shrink_thres=0.0025):
        super(AutoEncoderConv1D,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2000,1000),
            nn.Tanh(),
            nn.Linear(1000,500),
            nn.Tanh(),
            nn.Linear(500,250),
            nn.Tanh(),
            nn.Linear(250,120),
            nn.Tanh(),
            nn.Linear(120,60),
            nn.Tanh(),
            nn.Linear(60,30),
            nn.Tanh(),
            nn.Linear(30,10),
            nn.Tanh(),
            nn.Linear(10, 3),

        )

        self.mem_rep = MemModule(mem_dim = mem_dim, fea_dim=3, shrink_thres = shrink_thres )

        self.decoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.Tanh(),
            nn.Linear(10,30),
            nn.Tanh(),
            nn.Linear(30,60),
            nn.Tanh(),
            nn.Linear(60,120),
            nn.Tanh(),
            nn.Linear(120,250),
            nn.Tanh(),
            nn.Linear(250,500),
            nn.Tanh(),
            nn.Linear(500,1000),
            nn.Tanh(),
            nn.Linear(1000,2000)
        )

    def forward(self,x):
        f = self.encoder(x)
        encoder_out = f
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        decoder_in = f
        att = res_mem['att']
        out = self.decoder(f)
        return {'output': out, 'att': att, 'decoder_in' : decoder_in, 'encoder_out':encoder_out}