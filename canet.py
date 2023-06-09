#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from pandas import concat
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import dataset

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # 权值初始化，kaiming正态分布：此为0均值的正态分布，N～ (0,std)，其中std = sqrt(2/(1+a^2)*fan_in)
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.ReLU6):
            pass
        else:
            m.initialize()


# resnet组件
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * 4))

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./resnet50-19c8e357.pth'), strict=False)


""" Channel Attention Module """
class CALayer(nn.Module):
    def __init__(self, in_ch_left, in_ch_down):
        super(CALayer, self).__init__()
        self.conv0 = nn.Conv2d(in_ch_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_ch_down, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256
        down = down.mean(dim=(2, 3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        ca=left * down
        return ca

    def initialize(self):
        weight_init(self)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
    def initialize(self):
        weight_init(self)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    def initialize(self):
        weight_init(self)





""" Body_Aggregation1 Module """
class BA1(nn.Module):
    def __init__(self, in_ch_left, in_ch_down, in_ch_right):
        super(BA1, self).__init__()
        #self.CoordAttention = CoordAttention(256, 256,16)
        self.conv0 = nn.Conv2d(in_ch_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_ch_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_ch_right, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        # self.conv4 = nn.Conv2d(top, 256, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)

        #The above ops are used to reduce channels.left:low down:high right:global

        self.conv_d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_r = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(256 * 3, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)



    def forward(self, left, down, right):
        
        #top = F.relu(self.bn4(self.conv4(top)), inplace=True)  # 256 channels  
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels     
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256

        down_1 = self.conv_d(down)
        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_1 * left, inplace=True)

        # z3
        right_1 = self.conv_r(right)
        if right_1.size()[2:] != left.size()[2:]:
            right_1 = F.interpolate(right_1, size=left.size()[2:], mode='bilinear')
        z3 = F.relu(right_1 * left, inplace=True)



        out = torch.cat((z1, z2, z3), dim=1)       

        out=F.relu(self.bn3(self.conv3(out)), inplace=True)

        #top_1=self.CoordAttention(top)
       
        #out=out*top_1+out      
 
        return out               
        

    def initialize(self):
        weight_init(self)


""" Body_Aggregation2 Module """
class BA2(nn.Module):
    def __init__(self, in_ch):
        super(BA2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)

    def initialize(self):
        weight_init(self)


class ConvBn(nn.Sequential):
    """
    Cascade of 2D convolution and batch norm.
    """

    def __init__(self, in_ch, out_ch, kernel_size, padding=0, dilation=1):
        super(ConvBn, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

    def initialize(self):
        weight_init(self)
        


class SPPLayer(nn.Module):
    def __init__(self):
        super(SPPLayer, self).__init__()

    def forward(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_4 = F.max_pool2d(x, 13, stride=1, padding=6)
        out = torch.cat((x_1, x_2, x_3, x_4),dim=1)
        return out


class ASPPPooling(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBn(in_ch, out_ch, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = F.adaptive_avg_pool2d(x, 1)
        h = F.relu(self.conv(h))
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h

    def initialize(self):
        weight_init(self)


class ASPP(nn.Module):
    def __init__(self, in_ch, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        self.conv1 = ConvBn(in_ch, out_channels, 1)
        self.conv_aspp1 = ConvBn(in_ch, out_channels, 3, atrous_rates[0], atrous_rates[0])
        self.conv_aspp2 = ConvBn(in_ch, out_channels, 3, atrous_rates[1], atrous_rates[1])
        self.conv_aspp3 = ConvBn(in_ch, out_channels, 3, atrous_rates[2], atrous_rates[2])
        self.conv_pool = ASPPPooling(in_ch, out_channels)
        self.conv2 = ConvBn(5 * out_channels, in_ch, 1)

    def forward(self, x):
        res = []
        res.append(F.relu(self.conv1(x)))
        res.append(F.relu(self.conv_aspp1(x)))
        res.append(F.relu(self.conv_aspp2(x)))
        res.append(F.relu(self.conv_aspp3(x)))
        res.append(F.relu(self.conv_pool(x)))
        out = torch.cat([a for a in res], dim=1)
        out = F.relu(self.conv2(out))

        out = F.dropout(out, p=0.5, training=self.training)
        return out

    def initialize(self):
        weight_init(self)


class ContextNet(nn.Module):
    def __init__(self, cfg):
        super(ContextNet, self).__init__()
        self.aspp = ASPP(256, [1, 2, 4])
        self.cfg = cfg
        self.bkbone = ResNet()
        #self.sync_bn = torch.nn.SyncBatchNorm(num_features, eps=1e-05, momentum=0.1, affine=True, 
        #                          track_running_stats=True)
        
        self.ca45 = CALayer(2048,2048)
        self.ca35 = CALayer(2048,2048)
        self.ca25 = CALayer(2048,2048)
        


        self.ba1_45 = BA1(1024, 256, 256)
        self.ba1_34 = BA1(512, 256, 256)
        self.ba1_23 = BA1(256, 256, 256)
    
  
        self.ba2_4 = BA2(256)
        self.ba2_3 = BA2(256)
        self.ba2_2 = BA2(256)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)


        
        self.conv_o = nn.Conv2d(2048, 256, 1)
        #self.conv_p = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_q = nn.Conv2d(256, 2048, 1)
        self.conv5_c1= nn.Conv2d(2048,100, 1)
        self.conv4_c1= nn.Conv2d(1024,400, 1)
        # self.conv3_c1= nn.Conv2d(512,1600, 1)

        self.conv5_c2= nn.Conv2d(512, 256, 1)
        self.conv4_c2= nn.Conv2d(512,256, 1)
        # self.conv3_c2= nn.Conv2d(512,256,1)



        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn2048 = nn.BatchNorm2d(2048)
        self.bn100 = nn.BatchNorm2d(100)
        self.bn400 = nn.BatchNorm2d(400)
        # self.bn1600 = nn.BatchNorm2d(1600)

        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5_ = self.bkbone(x)

        out5_c= F.relu(self.bn100(self.conv5_c1(out5_))) 
        out5_c=out5_c.reshape(out5_c.shape[0],out5_.shape[2]*out5_.shape[3],-1)# transfer N C H*W
  
  
        out4_c= F.relu(self.bn400(self.conv4_c1(out4)))        
        out4_c=out4_c.reshape(out4.shape[0],out4.shape[2]*out4.shape[3],-1)# transfer N C H*W
         
     
 
        out5_f = F.relu(self.conv_o(out5_))  # 256 1*1
        out5_fp = self.aspp(out5_f)  # 256
        out5_f = self.bn2048(self.conv_q(out5_fp))  # 1*1

        out5_f = F.sigmoid(out5_f)
        out5_ = out5_ + out5_f * out5_

        # CA
        out4_a = self.ca45(out5_, out5_)
        out3_a = self.ca35(out5_, out5_)
        out2_a = self.ca25(out5_, out5_)

     
    

        out5 = out5_fp
        out5_c2=out5.reshape(out5.shape[0],out5.shape[2]*out5.shape[3],-1)       
        out5_c2=torch.bmm(F.sigmoid(out5_c),out5_c2)      
        out5_c2=out5_c2.reshape(out5_c2.shape[0],out5_c2.shape[2],out5.shape[2],-1)#out5 
        out5= torch.cat((out5, out5_c2), dim=1)  
        out5=F.relu(self.bn1(self.conv5_c2(out5)))


        out4 = self.ba1_45(out4, out5, out4_a)
        out4_c2=out4.reshape(out4.shape[0],out4.shape[2]*out4.shape[3],-1)
        out4_c2=torch.bmm(F.sigmoid(out4_c),out4_c2)
        out4_c2=out4_c2.reshape(out4_c2.shape[0],out4_c2.shape[2],out4.shape[2],-1)
        out4= torch.cat((out4, out4_c2), dim=1)   
        out4=F.relu(self.bn2(self.conv4_c2(out4)))
   

        out3 = self.ba1_34(out3, out4, out3_a)
        out3 = self.ba2_3(out3)
        out2 = self.ba2_2(self.ba1_23(out2, out3, out2_a))      
        
      
        
        out5_1 = self.linear5(out5)
        out4_1 = self.linear4(out4)
        out3_1 = self.linear3(out3)
        out2_1 = self.linear2(out2)        
              

        out5 = F.interpolate(out5_1, size=x.size()[2:], mode='bilinear')
        out4 = F.interpolate(out4_1, size=x.size()[2:], mode='bilinear')
        out3 = F.interpolate(out3_1, size=x.size()[2:], mode='bilinear')
        out2 = F.interpolate(out2_1, size=x.size()[2:], mode='bilinear')
     

        out5_c=out5_c.unsqueeze(1)
        out4_c=out4_c.unsqueeze(1)
       
        return out2, out3, out4, out5,out5_c,out4_c

    def initialize(self):
        if self.cfg.snapshot:  # 监控snapshot状态
            try:
                self.load_state_dict(torch.load(self.cfg.snapshot,map_location='cuda:0'))
            except:
                print("Warning: please check the snapshot file:", self.cfg.snapshot)
                pass
        else:
            weight_init(self)


