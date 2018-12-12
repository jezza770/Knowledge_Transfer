import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Fx
import torch.nn.init as init
from operator import itemgetter

class featExtractor(nn.Module):

    def __init__(self,model,loadlayer,trainlayer):
        super(featExtractor,self).__init__()
        for k, mod in model._modules.items():
            self.add_module(k,mod)
            if(trainlayer==k):
                for param in self.parameters():
                    param.requires_grad = False
        self.featLayer = loadlayer
        return

    def forward(self,x,outputs=None):
        if outputs!=None:
            returns=[0]*len(outputs)
        i=0
        for nm, module in self._modules.items():
            x = module(x)
            # print(nm)
            # if nm.split('-')[0] == self.featLayer:
                # out = x
            if outputs==None:
                if nm.split('-')[0] == self.featLayer:
                    returns = x
            elif nm in outputs:
                returns[i] = x
                i+=1
            # if nm == 'layer19':
                # out2 = x
        return returns#,out2

        
    def add_layer(self,layer,loadlayer,isFeat=False):
        for k, mod in layer._modules.items():
            self.add_module(loadlayer+'-'+k,mod)
        if(isFeat):
            self.featLayer = loadlayer
        return