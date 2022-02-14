import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.modules.loss import _Loss

class my_Loss(nn.Module):
    def __init__(self) -> None:
        super(my_Loss,self).__init__()
    def forward(self,target_speech,target_noise,input_speech):
        input_noise=target_speech+target_noise-input_speech
        return F.l1_loss(input_speech,target_speech)+F.l1_loss(input_noise,target_noise)


if __name__=="__main__":
    loss_fn=my_Loss()
    a=torch.randn(64,64000)
    b=torch.randn(64,64000)
    c=torch.randn(64,64000)
    d=loss_fn(a,b,c)
    print("the loss is :",d)