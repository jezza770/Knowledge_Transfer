import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Fx
import torch.nn.init as init
from operator import itemgetter


class weak_mxh64_1024(nn.Module):

    def __init__(self,nclass,glplfn,featType):
        super(weak_mxh64_1024,self).__init__() 
        self.globalpool = glplfn
        val=featType.split('layer')[-1]
        val=int(val)+1
        if(val>1):
            self.layer1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU())
        if(val>2):
            self.layer2 = nn.Sequential(nn.Conv2d(16,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU())
        if(val>3):
            self.layer3 = nn.MaxPool2d(2)

        if(val>4):
            self.layer4 = nn.Sequential(nn.Conv2d(16,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU())
        if(val>5):
            self.layer5 = nn.Sequential(nn.Conv2d(32,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU())
        if(val>6):
            self.layer6 = nn.MaxPool2d(2)

        if(val>7):
            self.layer7 = nn.Sequential(nn.Conv2d(32,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        if(val>8):
            self.layer8 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        if(val>9):
            self.layer9 = nn.MaxPool2d(2)

        if(val>10):
            self.layer10 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        if(val>11):
            self.layer11 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        if(val>12):
            self.layer12 = nn.MaxPool2d(2)

        if(val>13):
            self.layer13 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        if(val>14):
            self.layer14 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        if(val>15):
            self.layer15 = nn.MaxPool2d(2) #

        if(val>16):
            self.layer16 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.ReLU())
        if(val>17):
            self.layer17 = nn.MaxPool2d(2) # 
        
        if(val>18):
            self.layer18 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=2),nn.BatchNorm2d(1024),nn.ReLU()) # F1
        if(val>19):
            self.layer19 = nn.Sequential(nn.Conv2d(1024,nclass,kernel_size=1),nn.Sigmoid()) # F2
        return

    # def forward(self,x,featType):
        # val=featType.split('layer')[-1]
        # val=int(val)+1
        # if(val>1):
            # out = self.layer1(x)
        # if(val>2):
            # out = self.layer2(out)
        # if(val>3):
            # out = self.layer3(out)
        # if(val>4):
            # out = self.layer4(out)
        # if(val>5):
            # out = self.layer5(out)
        # if(val>6):
            # out = self.layer6(out)
        # if(val>7):
            # out = self.layer7(out)
        # if(val>8):
            # out = self.layer8(out)
        # if(val>9):
            # out = self.layer9(out)
        # if(val>10):
            # out = self.layer10(out)
        # if(val>11):
            # out = self.layer11(out)
        # if(val>12):
            # out = self.layer12(out)
        # if(val>13):
            # out = self.layer13(out)
        # if(val>14):
            # out = self.layer14(out)
        # if(val>15):
            # out = self.layer15(out)
        # if(val>16):
            # out = self.layer16(out)
        # if(val>17):
            # out = self.layer17(out)
        # if(val>18):
            # out = self.layer18(out)
        # if(val>19):
            # out = self.layer19(out)
        # out = self.globalpool(out,kernel_size=out.size()[2:])
        # out = out.view(out.size(0),-1)
        # return out #,out1

    


               
    
    
    
            
            
