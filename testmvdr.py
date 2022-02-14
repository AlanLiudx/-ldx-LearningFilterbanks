import torch
import torch.nn as nn
import numpy as np



class MVDR_Beamforming(nn.Module):
    def __init__(self):
        super(MVDR_Beamforming,self).__init__()
    


    def forward(self,x_stft,mask_stft):
        batch_size=x_stft.size(0)
        n_mics=x_stft.size(1)
        frequency_bins=x_stft.size(2)
        time_frame=x_stft.size(3)
        y_stft=x_stft*mask_stft
        MVDR_output=torch.zeros(batch_size,frequency_bins,time_frame)
        for batch_idx in range(batch_size):
            for frequency_idx in range(frequency_bins):
                M_sumtime=0
                Phiss_sumtime_unnormalized=0
                Phixx_sumtime=0
                for frame_idx in range(time_frame):
                    #the torch matrix calculation is for real value,perhaps we should maunally modify it for complex 
                    M_i=mask_stft[batch_idx,:,frequency_idx,frame_idx]
                    M_i_dot=M_i.dot(M_i.conj())
                    M_sumtime=M_sumtime+M_i_dot
                    Y_i=y_stft[batch_idx,:,frequency_idx,frame_idx]
                    X_i=x_stft[batch_idx,:,frequency_idx,frame_idx]
                    Phiss_i_unnormalized=torch.einsum('i,j->ij',Y_i,Y_i.conj())
                    Phiss_sumtime_unnormalized=Phiss_sumtime_unnormalized+Phiss_i_unnormalized
                    Phixx_i=torch.einsum('i,j->ij',X_i,X_i.conj())
                    Phixx_sumtime=Phixx_sumtime+Phixx_i
                #get correlation matrix of noisy(xx),speech(ss) and noisy(nn) in each freq bin
                Phixx=Phixx_sumtime/time_frame
                Phiss=Phiss_sumtime_unnormalized/M_sumtime
                Phinn=Phixx-Phiss
                Phinn_inv=torch.inverse(Phinn)
                (_,steering_vector,_)=torch.pca_lowrank(Phiss)
                numerator=torch.matmul(Phinn_inv,steering_vector)
                denominator=torch.matmul(steering_vector.conj().t().contiguous(),numerator)
                F_MVDR=numerator/denominator
                for frame_idx in range(time_frame):
                    MVDR_output[batch_idx,frequency_idx,frame_idx]=F_MVDR.dot(y_stft[batch_idx,:,frequency_idx,frame_idx])
                
                return MVDR_output


if __name__=="__main__":
    a=torch.randn(1,6,257,251)

    b=torch.randn(1,6,257,251)

    x_stft=a+1j*b

    mask_stft=b-2j*a

    output=MVDR_Beamforming(x_stft,mask_stft)

    print("output shape:",output.shape)