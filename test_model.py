from feat_extractor import get_class_dict,get_pred,get_inpt,compare_weights,get_model,two_stage_model
import numpy as np
import torch
import argparse
import pandas as pd
import torch.nn.functional as Fx
import os,glob
import cv2
import matplotlib.pyplot as plt
globalpoolfn = Fx.max_pool2d # can use max also
np.set_printoptions(threshold=np.nan)
np.core.arrayprint._line_width = 250
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn',help='name/path of NN model file to load',required=True)
    parser.add_argument('--svm',help='name/path of SVM model file to load',required=True)
    parser.add_argument('--sample', help='path to folder to get predictions for', required=True)
    args = parser.parse_args()
    
    samples=args.sample+os.sep+'*'+os.sep+'*.wav'
    classes=get_class_dict()
    confusion=np.zeros((50,50))
    esc_csv = pd.read_csv('esc50.csv')
    trained_model = torch.load(args.nn) #get_model('layer19','layer19')#
    SVM=cv2.ml.SVM_load(args.svm)
    #print(compare_weights(trained_model,trained_model))
    for file in glob.glob(samples):
        data_class=file.split(os.sep)[-2]
        data_class=data_class.split('-')[-1]
        category=classes[data_class]
        inpt=get_inpt(file)
        inpt = np.reshape(inpt,(1,inpt.shape[0],inpt.shape[1],inpt.shape[2]))
        pred=two_stage_model(inpt,trained_model,SVM,globalpoolfn)
        confusion[category][pred]+=1
    correct_count=0
    for i in range(50):
        correct_count += confusion[i][i]
    accuracy=correct_count/np.sum(confusion)
    print(accuracy)
    plt.imshow(confusion,cmap='gray')
    plt.show()