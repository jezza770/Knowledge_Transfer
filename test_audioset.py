from feat_extractor import get_class_dict,get_pred,get_inpt,get_model
import numpy as np
import torch
import argparse
import pandas as pd
import glob,os
import pickle
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D

def read_class_labels():
    with open('classes_id_name.txt','r') as f:
        labels=f.readlines()
    return labels

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--sample', help='path to sample file to get prediction for', required=True)
    # args = parser.parse_args()
    if(not os.path.exists('confusion.pickle')):
        trained_model=get_model('layer19','layer19')
        confusion=[[]]*50
        test_class=get_class_dict()
        labels=read_class_labels()  
        for folder in glob.glob('C:'+os.sep+'ESC50-sorted'+os.sep+'esc50-1'+os.sep+'train'+os.sep+'*'): # Path to ESC-50 sorted data, test fold 1
            f=(folder.split(os.sep)[-1]).split('-')[-1]
            for sample in glob.glob(folder+os.sep+'*.wav'):
                inpt=get_inpt(sample)
                inpt = np.reshape(inpt,(1,inpt.shape[0],inpt.shape[1],inpt.shape[2]))
                pred=get_pred(inpt,trained_model)
                labels_split=labels[pred].split(' ')
                pred_label=labels_split[0]+' '+' '.join(labels_split[2:])[0:-1]
                if(confusion[test_class[f]]==[]):
                    confusion[test_class[f]]=[[pred_label,1]]
                else:
                    for i in range(len(confusion[test_class[f]])+1):
                        if(i==len(confusion[test_class[f]])):
                            confusion[test_class[f]].append([pred_label,1])
                            break
                        p,c=confusion[test_class[f]][i]
                        if(p==pred_label):
                            confusion[test_class[f]][i][1]+=1
                            break
                        
            print(f)
            for p,c in confusion[test_class[f]]: #p=prediction, c=count
                print('    ',p,':',c)
        pickle.dump(confusion,open('confusion.pickle','wb'))
    else:
        confusion=pickle.load(open('confusion.pickle','rb'))
    conf_mat=np.zeros((50,527))
    #conf_mat.fill(-32)
    conf_lab=[]
    max_vals=[0]*527
    for i in range(50):
        for p,c in confusion[i]:
            for j in range(len(conf_lab)+1):
                if(j==len(conf_lab)):
                    conf_lab.append(p)
                    conf_mat[i][j]=c
                    break
                lab=conf_lab[j]
                if(lab==p):
                    conf_mat[i][j]=c
                    break
    
    conf_mat=conf_mat[:,:len(conf_lab)]
    sorted=np.zeros(len(conf_lab))
    max_sum=0
    exclude=[5,7,13,53]
    for i in range(len(conf_lab)):
        if(i in exclude):
            continue
        max_sum+=np.max(conf_mat[:,i])
        sorted[i]=np.argmax(conf_mat[:,i])
    conf_mat2=np.zeros((50,len(conf_lab)-len(exclude)))
    col=0
    for i in range(50):
        maxes=np.where(sorted==i)[0]
        for j in maxes:
            if(j in exclude):
                continue
            conf_mat2[:,col]=conf_mat[:,j]
            col+=1
    print(max_sum/np.sum(conf_mat2))
    print(conf_mat.shape,conf_mat2.shape)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    # fake data
    x = np.arange(len(conf_lab))
    y = np.arange(50)
    xx, yy = np.meshgrid(x, y)
    x, y = xx.ravel(), yy.ravel()

    top = np.reshape(conf_mat,-1)
    bottom = np.zeros_like(top)
    width = depth = 1

    # ax1.bar3d(x, y, bottom, width, depth, top)
    plt.imshow(conf_mat2,cmap='viridis')
    plt.colorbar()
    plt.show()
    