import torch
import torch.nn as nn
import time


class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, last=False):
        super(DepthConv1d, self).__init__()
        
        self.skip = skip
        self.last = last
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
                                 groups=hidden_channel,
                                 padding=self.padding)
        
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        
        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)
        if not (self.skip and self.last):
            self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        output = self.reg2(self.nonlinearity2(self.dconv1d(output)))

        if (self.skip and self.last):
            skip = self.skip_out(output)
            return skip
        elif (self.skip and not self.last):
            residual = self.res_out(output)
            skip = self.skip_out(output)
            return residual, skip
        else:
            residual = self.res_out(output)
            return residual


class TCN(nn.Module):

    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, kernel=3, skip=True, 
                 dilated=True):
        super(TCN, self).__init__()
        
        # input is a sequence of features of shape (B, N, L)
        
        # normalization
        self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)
        
        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated
        
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip)) 
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip))   
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
        
        self.skip = skip

        # modify last module
        if self.skip:
            if self.dilated:
                self.TCN[-1] = DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**(layer-1), padding=2**(layer-1), skip=skip, last=True)
            else:
                self.TCN[-1] = DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, last=True)
                    
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))
        
        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                   )
        
    def forward(self, input):
        
        # input shape: (B, N, L)
        
        # normalization
        output = self.BN(self.LN(input))
        
        # pass to TCN
        if self.skip:
            skip_connection = torch.zeros_like(output)
            for i in range(len(self.TCN) - 1):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
            skip = self.TCN[-1](output)
            skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual
            
        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        
        return output


# Conv-TasNet
class TasNet(nn.Module):

    def __init__(self,
                 enc_dim=512,
                 feature_dim=128,
                 win=20,
                 layer=8,
                 stack=3,
                 kernel=3,
                 num_spk=2,
                 in_dim=1):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk

        self.in_dim = in_dim
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        
        self.win = win
        self.stride = self.win // 2
        
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        
        # input encoder
        self.encoder = nn.Conv1d(self.in_dim, self.enc_dim, self.win, bias=False, stride=self.stride)
        
        # TCN separator
        self.TCN = TCN(self.enc_dim, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
                       self.layer, self.stack, self.kernel)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        in_dim = input.size(1)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = torch.autograd.Variable(torch.zeros(batch_size, in_dim, rest)).type_as(input)
            input = torch.cat([input, pad], 2)
        
        pad_aux = torch.autograd.Variable(torch.zeros(batch_size, in_dim, self.stride)).type_as(input)
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input):
        
        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)
        
        # waveform encoder
        enc_output = self.encoder(output)  # B, N, L

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, T
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, T
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T
        
        return output


if __name__ == "__main__":
    # x = torch.rand(2, 32000)
    x=torch.randn(8,1,64000)
    nnet=TasNet()
    out=nnet(x)
    print("x shape:",x.shape)
    print("out shape:",out.shape)
    # nnet = TasNet()
    # x = nnet(x)
    # s1 = x[0]
    # print(s1.shape)
    # print(x.shape)