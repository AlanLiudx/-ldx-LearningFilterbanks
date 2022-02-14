import argparse
import os
import time
import gc
import datetime
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.modules.loss import _Loss
from torch.optim import lr_scheduler

from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.autograd import  Function
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CasualUNet_Dataset
import model
from model import TasNet
import numpy as np
import random
from losses import SingleSrcNegSDR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子

# class my_Loss(nn.Module):
#     def __init__(self) -> None:
#         super(my_Loss,self).__init__()
#     def forward(self,target_speech,target_noise,input_speech):
#         input_noise=target_speech+target_noise-input_speech
#         return F.l1_loss(input_speech,target_speech)+F.l1_loss(input_noise,target_noise)



#不使用MAE Loss，改用SISDR loss
# class my_Loss(nn.Module):
#     def __init__(self) -> None:
#         super(my_Loss,self).__init__()
#     def forward(self,target_speech,target_noise,input_speech):
#         input_noise=target_speech+target_noise-input_speech
#         return F.l1_loss(input_speech,target_speech)+F.l1_loss(input_noise,target_noise)

if __name__ == '__main__':
    #set argument
    # args=mySetup()
    setup_seed(20)
    #设置随机数种子
    BATCH_SIZE=4
    NUM_EPOCHS=100
    LR=5e-5
    ADAM_BETA=(0.5,0.999)
    NUM_WORKERS=8
    IF_RESUME_BREAK=False
    # NUM_MICS=torch.zeros(BATCH_SIZE).type(torch.FloatTensor) #这个地方设置成这样比较好，设置成1会导致无法多卡
    #mkdir tensorboard and sisdr path
    if not os.path.isdir('./tensorboard'):
        os.mkdir('./tensorboard')
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    #add tensorboard path
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    tensorboard_path='./tensorboard/'+time_stamp
    if not os.path.isdir(tensorboard_path):
        os.mkdir(tensorboard_path)
        #按时间把不同文件的tensorboard扔到不同的位置下
    writer=SummaryWriter(tensorboard_path)
    #print the arguments
    print("let's see the argument")
    # print(args.__dict__)
    print("BATCH_SIZE:",BATCH_SIZE)
    #load data
    print('loading data...')
    train_dataset=CasualUNet_Dataset(data_type='train')
    valid_dataset=CasualUNet_Dataset(data_type='valid')
    # test_dataset=myDataset(data_type='test')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    #get model
    model_Casual = TasNet()
    # model_TAC=model_TAC_Net_22()
    # loss_func=my_Loss()
    # valid_loss_func=my_Loss()
    # loss_func=my_Loss()
    # valid_loss_func=my_Loss()
    loss_func=SingleSrcNegSDR("sisdr",reduction="mean")
    valid_loss_func=SingleSrcNegSDR("sisdr",reduction="mean")
    ##loss直接使用sdr文件里面的sisdr loss
    if torch.cuda.is_available():
        model_Casual=model_Casual.cuda()
    print("# model_TasNet parameters:",sum(param.numel() for param in model_Casual.parameters()))
    
    model_Casual_optimizer=optim.Adam(model_Casual.parameters(),lr=LR,betas=ADAM_BETA)
    if not os.path.isdir("./models"):
            os.mkdir("./models")
    if not os.path.isdir("./models/checkpoint"):
            os.mkdir("./models/checkpoint")
    checkpoint_path="./models/checkpoint/current_checkpoint.pth"
    model_path='./models/'
    if not os.path.exists(checkpoint_path):
        IF_RESUME_BREAK=False
    start_epoch=0
    best_loss=1e6
    best_epoch=-1
    nums=0
    if IF_RESUME_BREAK:
        checkpoint=torch.load(checkpoint_path)
        model_Casual.load_state_dict(checkpoint['net'])
        model_Casual_optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint['epoch']+1
        #give scheduler information
    # scheduler=lr_scheduler.StepLR(model_TAC_optimizer,15,0.5)
    # scheduler=lr_scheduler.StepLR(model_TAC_optimizer,50,0.2)
    scheduler=lr_scheduler.ReduceLROnPlateau(model_Casual_optimizer, mode="min",factor=0.5, patience=5, verbose=True,  cooldown=0, min_lr=0, eps=1e-08)
    for epoch in range(start_epoch,NUM_EPOCHS):
        train_loss=0
        train_bar=tqdm(train_data_loader)
        model_Casual.train()
        start_time=time.time()
        nums=0
        for mixture, target_spk_batch, target_noise_batch,ref_batch in train_bar:
            nums=nums+1
            if torch.cuda.is_available():
                mixture=mixture.cuda()
                target_spk_batch=target_spk_batch.cuda()
                target_noise_batch=target_noise_batch.cuda()
                ref_batch=ref_batch.cuda()
            
            output_batch=model_Casual(mixture)
            # output_spk1=output_batch[:,0:1,:]
            # output_spk2=output_batch[:,1:2,:]
            target_speech=target_spk_batch[:,0,:]
            target_noise=target_noise_batch[:,0,:]
            # my_sisdrloss=loss_func(target_speech,target_noise,output_batch)
            my_sisdrloss=loss_func(output_batch,target_speech)
            train_loss+=my_sisdrloss.detach().item()
            model_Casual.zero_grad()
            my_sisdrloss.backward()
            #梯度裁剪
            clip_grad_norm_(model_Casual.parameters(),max_norm=5.0,norm_type=2)
            #梯度裁剪结束，这段代码放在loss backward之后，model optimizer之前
            model_Casual_optimizer.step()
            #scheduler step
            
            train_bar.set_description(
                'Training Epoch {}:sisdr LOSS {:.12f}:Train_Loss_sum{:.12f}'
                    .format(epoch,my_sisdrloss.item(),train_loss,))
                    #scheduler.get_last_lr() is not available in reduce plateau
        print("training finished this epoch")
        
        train_loss=train_loss/nums
        print("[Trainning] [%d/%d], Elipse Time: %4f, Train Loss: %4f" % (
                epoch , NUM_EPOCHS, time.time() - start_time, train_loss))
        checkpoint = {
                    "net": model_Casual.state_dict(),
                    'optimizer': model_Casual_optimizer.state_dict(),
                    "epoch": epoch
                }
        torch.save(checkpoint, checkpoint_path)
        # del input_real,input_imag,output_real,output_imag
        gc.collect()
        #time for cross validation
        valid_loss=0
        valid_bar=tqdm(valid_data_loader)
        model_Casual.eval()
        start_time=time.time()
        nums=0
        with torch.no_grad():
            for mixture, target_spk_batch, target_noise_batch,ref_batch in valid_bar:
                nums=nums+1
                # print("val_input_real:",val_input_real)
                # time.sleep(5)
                # print("val_input_imag:",val_input_imag)
                # time.sleep(5)
                # print("val_target_real:",val_target_real)
                # time.sleep(5)
                # print("val_target_imag:",val_target_imag)
                # time.sleep(5)
                # print("val_target_ene:",torch.sum(val_target_real**2))
                # time.sleep(10)
                if torch.cuda.is_available():
                    mixture=mixture.cuda()
                    target_spk_batch=target_spk_batch.cuda()
                    target_noise_batch=target_noise_batch.cuda()
                    ref_batch=ref_batch.cuda()
                # model_TAC.zero_grad()
                output_batch=model_Casual(mixture)
                target_speech=target_spk_batch[:,0,:]
                target_noise=target_noise_batch[:,0,:]
                # valid_sisdrloss=valid_loss_func(target_speech,target_noise,output_batch)
                valid_sisdrloss=loss_func(output_batch,target_speech)
                valid_loss+=valid_sisdrloss.detach().item()
                #my_mseloss.backward()
                #model_TAC_optimizer.step()
                valid_bar.set_description(
                    'Validation Epoch {}:MSELOSS {:.12f}:Valid_Loss_sum{:.12f}'
                        .format(epoch,valid_sisdrloss.item(),valid_loss))
            print("validation finished this epoch")
            valid_loss=valid_loss/nums
            scheduler.step(valid_loss)
            print("[validation] [%d/%d], Elipse Time: %4f, valid Loss: %4f" % (
                epoch , NUM_EPOCHS, time.time() - start_time, valid_loss))
        if(valid_loss<=best_loss):
            best_loss=valid_loss
            best_epoch=epoch
            best_unet_path = os.path.join(model_path, 'best-valid.pkl' )
            best_unet = model_Casual.state_dict()
            # print("nest_unet：",best_unet)
            # del input_real,input_imag,output_real,output_imag
            print('Find better valid model, Best  model loss : %.4f\n' % ( valid_loss))
            torch.save(best_unet, best_unet_path)
        gc.collect()
        if(epoch%5==0):
            five_unet_path = os.path.join(model_path, 'epoch_%d.pkl' % (
                     epoch))
            five_unet = model_Casual.state_dict()
            torch.save(five_unet, five_unet_path)
        print("the best epoch is:",best_epoch)
        #tensorboard add scalar(loss)
        writer.add_scalar("train loss",train_loss,epoch)
        writer.add_scalar("valid loss",valid_loss,epoch)
        gc.collect()
        #finish tensorboard add scalar
    print("the training process all done")  

