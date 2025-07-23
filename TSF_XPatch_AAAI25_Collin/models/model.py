import torch
import torch.nn as nn

from layers.decomp import DECOMP
from layers.revin import RevIN


# 2. Non-linear block
# 2.1 Patching layer
# Input: Seasonal Component  s: [Batch and Channel , Input] ->  output: [Batch and Channel, Patch_num, Patch_len]
class PathcingLayer(nn.Module):

    def __init__(self, seq_len:int, patch_len:int, stride:int, padding_patch:str):
        super().__init__()
        self.seq_len = seq_len # 序列长度
        self.patch_len = patch_len # 每个patch的长度
        self.stride = stride # 滑动窗口的长度
        self.padding_patch = padding_patch # 用于在patch中进行padding的符号
        self.dim = patch_len * patch_len # patchembedding dim
        self.patch_num = (seq_len - patch_len) // stride + 1 # patch的数量
        if padding_patch == 'end':
            self.padding_layer = nn.ReplicationPad1d((0,stride)) #  ReplicationPad 的意思是“复制填充”，它会用序列边界的值来填充。具体来说，ReplicationPad1d 会复制序列的最后一个元素来填充右侧。
            self.patch_num += 1

    def forward(self,x):


        # step 3: Patching  [Batch*Channel, Input] -> [Batch*Channel, Patch_num,Patch_len]
        if self.padding_patch == 'end':
            x = self.padding_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) ##滑动窗口操作，将输入序列分成多个patch

        return  x # [Batch*Channel, Patch_num,Patch_len]

# 2.2 PatchingEmbedding Layer
# S:[Batch*Channel, Patch_num,Patch_len] -> [Batch*Channel, Patch_num, patch_len * patch_len]
class PatchingEmbedding(nn.Module):

    def __init__(self, patch_len:int, patch_num:int)->None:
        super().__init__()
        self.dim = patch_len * patch_len

        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm1d(patch_num)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.fc1(x) # [Batch*Channel, Patch_num,Patch_len] * [Batch*Channel, patch_len,patch_len * patch_len] -> [Batch*Channel, Patch_num, patch_len * patch_len]
        x = self.gelu(x)
        x = self.bn1(x)
        return x  # output is  [Batch*Channel, Patch_num, patch_len * patch_len]

# 2.3 Residual Layer
# [Batch*Channel, Patch_num, patch_len * patch_len] * [Batch*Channel, patch_len * patch_len, patch_len] -> [Batch*Channel, Patch_num, patch_len]
class ResidualLayer(nn.Module):
    def __init__(self, patch_len:int )->None:
        super().__init__()
        self.dim = patch_len * patch_len
        self.fc2 = nn.Linear(self.dim, patch_len)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.fc2(x)
        return x   # this is the residual stream output

# 2.4 Depthwise CNN and GELU and BatchNorm
class DepthwiseConv(nn.Module):
    def __init__(self,patch_num:int, patch_len:int)->None:
        super().__init__()

        self.conv1=nn.Conv1d(patch_num, patch_num, patch_len, patch_len, groups=patch_num)
        self.gelu = nn.GELU()
        self.bn2 = nn.BatchNorm1d(patch_num)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.bn2(x)
        return x

# 2.5 Residual Connection
class ResidualBlock(nn.Module):
    def __init__(self, patch_num:int, patch_len:int )->None:
        """
        Initialize the residual block
        Args:
            patch_num: number of patches (N)
            patch_len: length of patches (P)
        """
        super().__init__()
        # Instantiate the two parallel layers/streams that will be combined
        self.residual_layer=ResidualLayer(patch_len)
        self.depthwise_conv=DepthwiseConv(patch_num, patch_len)

    def forward(self,x:torch.Tensor)->torch.Tensor:

        # 1.Calculate the output of the residual strem
        residual_out=self.residual_layer(x)

        # 2.Calculate the output of the main depthwise convolution stream
        depthwise_out=self.depthwise_conv(x)

        # 3. Add the outputs of the two streams together
        output = residual_out + depthwise_out
        return output

# 2.6 Pointwise CNN
class PointwiseConv(nn.Module):
    def __init__(self,patch_num:int)->None:
        super().__init__()

        self.conv2 = nn.Conv1d(patch_num,patch_num,1,1)
        self.gelu = nn.GELU()
        self.bn3 = nn.BatchNorm1d(patch_num)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.bn3(x)
        return x

# 2.7 MLP Flatten layer
class MLPFlatten(nn.Module):
    def __init__(self,patch_num:int,patch_len:int,pred_len:int)->None:
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(patch_num*patch_len, pred_len*2)
        self.gelu = nn.GELU()
        self.fc4 = nn.Linear(pred_len*2, pred_len)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.flatten(x)
        x = self.fc3(x)
        x = self.gelu(x)
        x = self.fc4(x)
        return x

# 2.6 Non-Linear block
class NonLinearBlock(nn.Module):
    def __init__(self,seq_len:int, patch_len:int, stride:int, pred_len:int)->None:
        super().__init__()
        self.patchLayer=PathcingLayer(seq_len,patch_len,stride,padding_patch='end')
        patch_num = self.patchLayer.patch_num

        self.patchEmbedding=PatchingEmbedding(patch_len,patch_num)
        self.residualBlock=ResidualBlock(patch_num,patch_len)
        self.pointwiseConv=PointwiseConv(patch_num)
        self.mlpFlatten=MLPFlatten(patch_num,patch_len,pred_len)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.patchLayer(x)
        x = self.patchEmbedding(x)
        x = self.residualBlock(x)
        x = self.pointwiseConv(x)
        x = self.mlpFlatten(x)
        return x

# 3. Linear block
# 3.1 Fully-connected and Average Pool and LayerNorm
class LinearBlock(nn.Module):
    def __init__(self,seq_len:int,pred_len:int)->None:
        super().__init__()
        self.fc5 = nn.Linear(seq_len,pred_len*4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len*2)

        self.fc6 = nn.Linear(pred_len*2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len//2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.fc5(x)
        x = self.avgpool1(x)
        x = self.ln1(x)
        x = self.fc6(x)
        x = self.avgpool2(x)
        x = self.ln2(x)
        x = self.fc7(x)
        return x


class XPatch(nn.Module):

    def __init__(self,seq_len:int, pred_len:int, enc_in:int, patch_len:int, stride:int,alpha:float, ma_type:str="ema", beta:float=0.3, revin:bool=True)->None:
        super().__init__()
        self.pred_len=pred_len

        # 1. Instantiate pre-processing layers
        self.revin=revin
        self.revin_layer= RevIN(num_features=enc_in,affine=True,subtract_last=False)
        self.decomposition = DECOMP(ma_type=ma_type,alpha=alpha,beta=beta)

        # 2. Instantiate the two main network blocks
        self.nonlinear_block = NonLinearBlock(seq_len,pred_len,stride,pred_len)
        self.linear_block = LinearBlock(seq_len,pred_len)

        # 3. Streams Concatination
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self,x)->torch.Tensor:
        # x: [Batch, Input, Channel]

        # 1. Apply RevIN normalization if enabled
        if self.revin:
            x = self.revin_layer(x,'norm')

        # 2. Decompose into seasonal and trend components. Input is 3D [B, L, C].
        s,t = self.decomposition(x)

        # 3. Permute for channel independence: [B,L,C] -> [B,C,L]
        s = s.permute(0,2,1)
        t = t.permute(0,2,1)

        # 4. Reshape for channel independence: [B,C,L] -> [B*C, L]
        B, C, L = s.shape
        s = s.reshape(B*C, L)
        t = t.reshape(B*C, L)

        # 5. Process through respective streams
        s_pre = self.nonlinear_block(s) # -> [B*C, pred_len]
        t_pre = self.linear_block(t)

        # 6. Concatenate stream outputs
        x_out = torch.cat((s_pre,t_pre),dim=1) #->[B*C, pred_len*2]

        # 7. Final Projection
        x_out = self.fc8(x_out) # -> [B*C, pred_len]

        # 8. Reshape back to include channels: [B*C, pred_len] -> [B,C,pred_len]
        x_out = x_out.reshape(B,C,self.pred_len)

        # 9. Permute back to original format: [B,C,pred_len] -> [B,pred_len,C]
        x_out = x_out.permute(0,2,1)

        # 10. Apply RevIn denormalization to get the final prediction
        if self.revin:
            x_out = self.revin_layer(x_out,'denorm')
        return x_out



