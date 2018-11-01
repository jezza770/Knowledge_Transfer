import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Fx
import torch.nn.init as init
from operator import itemgetter

class featExtractor(nn.Module):

    def __init__(self,model,layer_name):
        super(featExtractor,self).__init__()
        for k, mod in model._modules.items():
            self.add_module(k,mod)
            if(layer_name==k):
                for param in self.parameters():
                    param.requires_grad = False
        self.featLayer = layer_name
        return

    def forward(self,x):
        for nm, module in self._modules.items():
            x = module(x)
            if nm.split('-')[0] == self.featLayer:
                out = x
        return out

        
    def add_layer(self,layer,layer_name,isFeat=False):
        for k, mod in layer._modules.items():
            self.add_module(layer_name+'-'+k,mod)
        if(isFeat):
            self.featLayer = layer_name
        return