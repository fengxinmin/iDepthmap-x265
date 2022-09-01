#**********************************************
# improved depth prediction model (from luma CTU to depth map)
# authorized by Feng Xinmin
#**********************************************

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as f

class BasicBlock(nn.Module):
    """改变了feature map的尺寸"""
    def __init__(self,in_channel,out_channel,stride=2,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel,3,1,1,bias=False)
        self.pooling1 = nn.MaxPool2d(2)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.conv2=nn.Conv2d(out_channel,out_channel,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.identity_conv1 = nn.Conv2d(in_channel,out_channel,3,stride,1,bias=False)
        self.identity_bn = nn.BatchNorm2d(out_channel)
        self.downsample=downsample
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(x)  
        else:
            identity = self.identity_conv1(x)
            identity = self.identity_bn(identity)  

        out=self.relu(self.bn1(self.pooling1(self.conv1(x))))
        out=self.bn2(self.conv2(out))
        out+=identity  
        out=self.relu(out)
        return out

class IdentityBlock(nn.Module):
    """仅改变通道数"""
    def __init__(self,in_channel,out_channel,downsample=None):
        super(IdentityBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel,3,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.conv2=nn.Conv2d(out_channel,out_channel,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

        self.identity_conv = nn.Conv2d(in_channel,out_channel,1,1)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        identity=self.identity_conv(x)
        if self.downsample is not None:
            identity=self.downsample(x)     
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=identity  
        out=self.relu(out)
        return out

class Sub_Net1(nn.Module):
    def __init__(self):
        super(Sub_Net1,self).__init__()
        self.conv1 = nn.Conv2d(1,64,5,1,2)
        self.basic_block1 = BasicBlock(64,64)
        self.basic_block2 = BasicBlock(64,64)
        self.identity_block = IdentityBlock(64,64)
    def forward(self,x):
        # input 1*64*64 output 64*16*16
        out = self.conv1(x)
        out = self.basic_block1(out)
        out = self.basic_block2(out)
        out = self.identity_block(out)
        return out

class Sub_Net2(nn.Module):
    def __init__(self):
        super(Sub_Net2,self).__init__()
        self.basic_block1 = BasicBlock(64,64)
        self.identity_block1 = IdentityBlock(64,32)
        self.identity_block2 = IdentityBlock(32,32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
    def forward(self,x):
        # input 64*16*16 output 128*8*8
        out = self.basic_block1(x)
        out = self.identity_block1(out)
        out8x8 = self.identity_block2(out)  # 8x8
        out4x4 = self.pool2(out8x8)
        out2x2 = self.pool2(out4x4)
        out1x1 = self.pool2(out2x2)
        out = torch.cat([ f.interpolate(out1x1,scale_factor=8), f.interpolate(out2x2,scale_factor=4), f.interpolate(out4x4,scale_factor=2),out8x8],dim=1)
        return out

class Sub_Net3(nn.Module):
    def __init__(self):
        super(Sub_Net3,self).__init__()
        self.identity_block1 = IdentityBlock(128,64)
        self.identity_block2 = IdentityBlock(64,32)
        self.identity_block3 = IdentityBlock(32,16)
        self.identity_block4 = IdentityBlock(16,8)
        self.conv = nn.Conv2d(8,1,1,1)
    def forward(self,x):
        # input 128*8*8 output 1*8*8
        out = self.identity_block1(x)
        out = self.identity_block2(out)
        out = self.identity_block3(out)
        out = self.identity_block4(out)
        out = torch.sigmoid(self.conv(out))
        return out
        
class Sub_Net4(nn.Module):
    def __init__(self):
        super(Sub_Net4,self).__init__()
        self.identity_block1 = IdentityBlock(128,64)
        self.identity_block2 = IdentityBlock(64,32)
        self.identity_block3 = IdentityBlock(32,16)
        self.identity_block4 = IdentityBlock(16,8)
        self.conv = nn.Conv2d(8,1,1,1)
    def forward(self,x,identity):
        # input 128*8*8 output 1*8*8
        out = self.identity_block1(x)
        out = self.identity_block2(out)
        out = torch.mul(out,identity)
        out = self.identity_block3(out)
        out = self.identity_block4(out)
        out = torch.sigmoid(self.conv(out))
        return out

class Sub_Net5(nn.Module):
    def __init__(self):
        super(Sub_Net5,self).__init__()
        self.identity_block1 = IdentityBlock(64,64)
        self.basic_block1 = BasicBlock(64,32,2)
        self.identity_block2 = IdentityBlock(32,16)
        self.identity_block3 = IdentityBlock(16,8)
        self.conv = nn.Conv2d(8,1,1,1)
    def forward(self,x,identity):
        # input 64*16*16 output 1*8*8
        out = self.identity_block1(x)
        out = self.basic_block1(out)
        out = torch.mul(out,identity)
        out = self.identity_block2(out)
        out = self.identity_block3(out)
        out = torch.sigmoid(self.conv(out))
        return out    

class XOR_Net(nn.Module):
    def __init__(self):
        super(XOR_Net,self).__init__()
        self.fc1 = nn.Linear(12,4)
        self.fc2 = nn.Linear(4,1)
    def forward(self,x):
        # input B*1*4*4  output B*1
        out = torch.flatten(x,dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = f.tanh(out)
        return out

# class Weak_Checker_Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self,map_1,map_2,map_3):
#         # input B*1*8*8 output B*1
#         map_2 = map_2 - map_1
#         map_3 = map_3 - map_2
#         dev_8x8 = torch.std(map_1.view(map_1.shape[0],-1),unbiased=False,dim=1) # B*1
#         dev_4x4,dev_2x2 = torch.zeros(map_2.shape[0],4),torch.zeros(map_2.shape[0],16)
#         xmap_4x4 = torch.zeros(map_2.shape[0],4,16)
#         xmap_2x2 = torch.zeros(map_2.shape[0],16,4)
#         # map_2 B 1 8 8   
#         for idx,submap in enumerate([map_2[:,:,x:x+4,y:y+4] for x in [0,4] for y in [0,4]]):
#             xmap_4x4[:,idx] = submap.reshape(submap.shape[0],-1)
#             dev_4x4[:,idx] = torch.std(xmap_4x4[:,idx],unbiased=False,dim=1)

#         for idx,submap in enumerate([map_3[:,:,x:x+2,y:y+2] for x in [0,2,4,6] for y in [0,2,4,6]]):
#             xmap_2x2[:,idx] = submap.reshape(submap.shape[0],-1)
#             dev_2x2[:,idx] = torch.std(xmap_2x2[:,idx],unbiased=False,dim=1)
   
#         return (dev_8x8+(dev_4x4.sum().item() / 4) + (dev_2x2.sum().item() / 16) )/3  # return B*1

class Weak_Checker_Net(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,map_1,map_2,map_3):
        # input B*1*8*8 output B*1
        map_2 = map_2 - map_1
        map_3 = map_3 - map_2
        dev_8x8 = torch.max(map_1) - torch.min(map_1) # B*1
        dev_4x4,dev_2x2 = torch.zeros(map_2.shape[0],4),torch.zeros(map_2.shape[0],16)
        xmap_4x4 = torch.zeros(map_2.shape[0],4,16)
        xmap_2x2 = torch.zeros(map_2.shape[0],16,4)
        # map_2 B 1 8 8   
        for idx,submap in enumerate([map_2[:,:,x:x+4,y:y+4] for x in [0,4] for y in [0,4]]):
            xmap_4x4[:,idx] = submap.reshape(submap.shape[0],-1)
            dev_4x4[:,idx] = torch.max(xmap_4x4[:,idx]) - torch.min(xmap_4x4[:,idx])

        for idx,submap in enumerate([map_3[:,:,x:x+2,y:y+2] for x in [0,2,4,6] for y in [0,2,4,6]]):
            xmap_2x2[:,idx] = submap.reshape(submap.shape[0],-1)
            dev_2x2[:,idx] = torch.max(xmap_2x2[:,idx]) - torch.min(xmap_2x2[:,idx])
   
        return (dev_8x8+(dev_4x4.sum().item() / 4) + (dev_2x2.sum().item() / 16) )/3  # return B*1


class Strong_Checker_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.weak_checker = Weak_Checker_Net()
        self.pool = nn.MaxPool2d(2)
        self.xor_net = XOR_Net()
    def forward(self,map_1,map_2,map_3):
        # input B*1*8*8
        local_loss1 = self.weak_checker(map_1,map_2,map_3)
        local_loss2 = 0
        for map in [map_2,map_3]:
            map = torch.round(self.pool(map))
            map = nn.ReLU(map - 1)
            for sub_map in [map_2[:,:,x:x+2,:,:,y:y+2] for x in [0,2] for y in [0,2]]:
                local_loss2 += self.xor_net(torch.flatten(sub_map,dim=2))  # B*1*4*4
        return local_loss1+local_loss2
            
class IDP_Net(nn.Module):
    def __init__(self,strong_checker=False):
        super().__init__()
        self.net1 = Sub_Net1()
        self.net2 = Sub_Net2()
        self.net3 = Sub_Net3()
        self.net4 = Sub_Net4()
        self.net5 = Sub_Net5()
        if strong_checker == True:
            self.checker = Strong_Checker_Net()
        else:
            self.checker = Weak_Checker_Net()
    def forward(self,x):
        # input B*1*64*64
        out1 = self.net1(x)
        out2 = self.net2(out1)
        out3 = self.net3(out2) # B*1*8*8, Depth = 1
        out4 = self.net4(out2,out3) + out3 # Depth = 2
        out5 = self.net5(out1,out4) + out4 # Depth = 3
        invalidity_score = self.checker(out3,out4,out5)
        return torch.cat((out5,out4,out3),dim=1),invalidity_score.mean(0)  # depth = 3,2,1, (B,3,8,8), (1,)

    # TODO 检查strong checker
    # TODO 检查weak checker