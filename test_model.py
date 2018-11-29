from feat_extractor import get_class_dict,get_pred,get_inpt
import numpy as np
import torch
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',help='name/path of model file to load',required=True)
    parser.add_argument('--sample', help='path to sample file to get prediction for', required=True)
    args = parser.parse_args()
    
    classes=get_class_dict()
    esc_csv = pd.read_csv('esc50.csv')
    inpt=get_inpt(args.sample)
    trained_model = torch.load(args.model)
    inpt = np.reshape(inpt,(1,inpt.shape[0],inpt.shape[1],inpt.shape[2]))
    pred=get_pred(inpt,trained_model)
    print(pred, esc_csv['category'][pred])