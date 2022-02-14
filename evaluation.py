import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.modules.loss import _Loss
from model import TasNet
import model
import numpy as np
from tqdm import tqdm
from pesq import pesq as pesq_inner # https://github.com/ludlows/python-pesq
from pystoi.stoi import stoi as stoi_fn # https://github.com/mpariente/pystoi
from torch.utils.data import Dataset, DataLoader
from dataset import CasualUNet_Dataset

model_path="/home/liudx/Project/LearningFilterbanks/models/best-valid.pkl" # we might change the path of our model in different condition
#load data
test_dataset=CasualUNet_Dataset(data_type='test')
test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)
data_num=len(test_dataset)
print("data num:",data_num)
# NUM_MICS=6

def calc_sisdr(sig_cln, sig_proc, eps=1e-8):
    '''
    calculate Si-SDR for one mono audio
    '''
    def _norm(x):
        return np.sum(x ** 2)
    sig_proc = np.asarray(sig_proc, dtype=np.float32)
    sig_cln = np.asarray(sig_cln, dtype=np.float32)
    if np.max(sig_cln) > 2:
        sig_cln /= (2**15)
        sig_proc /= (2**15)
    #  sig_cln = sig_cln - np.mean(sig_cln)
    #  sig_proc = sig_proc - np.mean(sig_proc)
    sig_tar = np.sum(sig_cln * sig_proc) * sig_cln / (_norm(sig_cln) + eps)
    upp = _norm(sig_tar)
    low = _norm(sig_proc - sig_tar)
    return 10 * np.log10(upp) - 10 * np.log10(low)

def calc_pesq(clean_speech, processed_speech, fs):
    if fs == 8000:
        pesq_mos = pesq_inner(fs,clean_speech, processed_speech, 'nb')
        pesq_mos = 46607/14945 - (2000*np.log(1/(pesq_mos/4 - 999/4000) - 1))/2989 #remap to raw pesq score

    elif fs == 16000:
        pesq_mos = pesq_inner(fs,clean_speech, processed_speech, 'wb')
    elif fs >= 16000:
        numSamples=round(len(clean_speech)/fs*16000)
        pesq_mos = pesq_inner(fs,resample(clean_speech, numSamples),
                              resample(processed_speech, numSamples), 'wb')
    else:
        numSamples=round(len(clean_speech)/fs*8000)
        pesq_mos = pesq_inner(fs,resample(clean_speech, numSamples),
                              resample(processed_speech, numSamples), 'nb')
        pesq_mos = 46607/14945 - (2000*np.log(1/(pesq_mos/4 - 999/4000) - 1))/2989 #remap to raw pesq score

    return pesq_mos
def calc_estoi(clean,processed,fs):
    estoi = stoi_fn(clean, processed, fs, extended=True)
    return estoi

if __name__ == "__main__":
    sisdr_list=[]
    pesq_list=[]
    estoi_list=[]
    mix_sisdr_list=[]
    mix_pesq_list=[]
    mix_estoi_list=[]
    model_Casual = TasNet()
    model_Casual.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model_Casual=model_Casual.cuda()
    model_Casual.eval()
    index=0
    for mixture,target_spk_batch,target_noise_batch,ref_batch in test_data_loader:
        index=index+1
        print("show the index _______________________________________:")
        print("the index is:",index)
        if torch.cuda.is_available():
                mixture=mixture.cuda()
                target_spk_batch=target_spk_batch.cuda()
                target_noise_batch=target_noise_batch.cuda()
                ref_batch=ref_batch.cuda()
        output_batch=model_Casual(mixture)
        # print("shape mixture:",mixture.shape)
        mixture_ref=mixture[0,0,:]
        target_speech=target_spk_batch[0,:]
        target_noise=target_noise_batch[0,:]
        mixture_speech=mixture_ref.cpu().detach().numpy().copy()
        output_speech=output_batch.cpu().detach().numpy().copy()
        target_speech=target_speech.cpu().detach().numpy().copy()
        target_noise=target_noise.cpu().detach().numpy().copy()
        # print("shape mixture:",mixture_speech.shape)
        # print("shape 2:",output_speech.shape)
        # print("shape 3:",target_speech.shape)
        # print("shape 4:",target_noise.shape)
        wav_length=output_speech.shape[1]
        # print("wav length:",wav_length)
        #reshape to 1 dim
        mixture_speech=mixture_speech.reshape(wav_length)
        output_speech=output_speech.reshape(wav_length)
        target_speech=target_speech.reshape(wav_length)
        target_noise=target_noise.reshape(wav_length)
        # print("shape mixture:",mixture_speech.shape)
        # print("shape 2:",output_speech.shape)
        # print("shape 3:",target_speech.shape)
        # print("shape 4:",target_noise.shape)
        #sisdr
        sisdr_mix=calc_sisdr(target_speech,mixture_speech)
        print("sisdr mix:",sisdr_mix)
        mix_sisdr_list.append(sisdr_mix)
        sisdr_enhanced=calc_sisdr(target_speech,output_speech)
        print("sisdr out:",sisdr_enhanced)
        sisdr_list.append(sisdr_enhanced)
        #pesq
        pesq_mix=calc_pesq(target_speech,mixture_speech,fs=16000)
        print("pesq mix:",pesq_mix)
        mix_pesq_list.append(pesq_mix)
        pesq_enhanced=calc_pesq(target_speech,output_speech,fs=16000)
        print("pesq out:",pesq_enhanced)
        pesq_list.append(pesq_enhanced)
        #estoi
        estoi_mix=calc_estoi(target_speech,mixture_speech,fs=16000)
        print("estoi mix:",estoi_mix)
        mix_estoi_list.append(estoi_mix)
        estoi_out=calc_estoi(target_speech,output_speech,fs=16000)
        print("estoi out:",estoi_out)
        estoi_list.append(estoi_out)

        #output
    print("the mixture statistics:")
    print("sisdr mixture overall:",sum(mix_sisdr_list)/data_num)
    print("pesq mixture overall:",sum(mix_pesq_list)/data_num)
    print("estoi mixture overall:",sum(mix_estoi_list)/data_num)

    print("the output statistics:")
    print("sisdr output overall:",sum(sisdr_list)/data_num)
    print("pesq output overall:",sum(pesq_list)/data_num)
    print("estoi output overall:",sum(estoi_list)/data_num)
    


        # output_spk1=output_batch[:,0,:]
        # output_spk2=output_batch[:,1,:]
        # target_wav1=target_spk1.numpy().copy()
        # target_wav2=target_spk2.numpy().copy()
        # output_wav1=output_spk1.numpy().copy()
        # output_wav2=output_spk2.numpy().copy()
        # target_wav1=target_wav1.view(-1,)
        # target_wav2=target_wav2.view(-1,)
        # output_wav1=output_wav1.view(-1,)
        # output_wav2=output_wav2.view(-1,)
        # if((calc_sisdr(target_wav1,output_wav1)+calc_sisdr(target_wav2,output_wav2))<=(calc_sisdr(target_wav1,output_wav2)+calc_sisdr(target_wav2,output_wav1))):
        #     temp=output_wav1.copy()
        #     output_wav1=output_wav2.copy()
        #     output_wav2=temp.copy()
        # sisdr_spk1=calc_sisdr(target_wav1,output_wav1)
        # print("sisdr_spk1:",sisdr_spk1)
        # sisdr_list.append(sisdr_spk1)
        # sisdr_spk2=calc_sisdr(target_wav2,output_wav2)
        # print("sisdr_spk2:",sisdr_spk2)
        # sisdr_list.append(sisdr_spk2)
        # pesq_spk1=calc_pesq(target_wav1,output_wav1,fs=16000)
        # print("pesq spk1:",pesq_spk1)
        # pesq_list.append(pesq_spk1)
        # pesq_spk2=calc_pesq(target_wav2,output_wav2,fs=16000)
        # print("pesq spk2:",pesq_spk2)
        # pesq_list.append(pesq_spk2)
        # estoi_spk1=calc_estoi(target_wav1,output_wav1,fs=16000)
        # print("estoi spk1:",estoi_spk1)
        # estoi_list.append(estoi_spk1)
        # estoi_spk2=calc_estoi(target_wav2,output_wav2,fs=16000)
        # print("estoi spk2:",estoi_spk2)
        # estoi_list.append(estoi_spk2)
        # all_sisdr=sum(sisdr_list)
        # all_pesq=sum(pesq_list)
        # all_estoi=sum(estoi_list)
        # mean_sisdr=all_sisdr/(2*data_num)
        # print("the sisdr:",mean_sisdr)
        # mean_pesq=all_pesq/(2*data_num)
        # print("the pesq:",mean_pesq)
        # mean_estoi=all_estoi/(2*data_num)
        # print("the estoi:",mean_estoi)
        #只计算指标，暂时不考虑输出音频


