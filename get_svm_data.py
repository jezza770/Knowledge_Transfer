from feat_extractor import get_class_dict,get_pred,get_inpt,compare_weights,get_model,do_pooling
import numpy as np
import torch
import argparse
import pandas as pd
import torch.nn.functional as Fx
import os,glob
globalpoolfn = Fx.max_pool2d # can use max also
np.set_printoptions(threshold=np.nan)
np.core.arrayprint._line_width = 250
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',help='name/path of model file to load',required=True)
    parser.add_argument('--sorted', help='path to ESC-50 sorted directory', required=True)
    parser.add_argument('--testfold',help='which fold of ESC-50 will be for testing (1-5)',choices=['1','2','3','4','5'],required=True)
    args = parser.parse_args()
    test_fold=args.testfold
    data_dir = os.path.join(os.path.abspath(args.sorted),'esc50-'+test_fold) 
    classes=get_class_dict()
    esc_csv = pd.read_csv('esc50.csv')
    #inpt=get_inpt(args.sample)
    trained_model = torch.load(args.model) #get_model('layer19','layer19')
    #print(compare_weights(trained_model,trained_model))
    trained_model.eval()
    for mode in ['train','val']:
        samples=data_dir+os.sep+mode+os.sep+'*'+os.sep+'*.wav'
        num_train=len(glob.glob(samples))
        train_data1=np.zeros((num_train,1024),dtype=np.float32)
        train_data2=np.zeros((num_train,527),dtype=np.float32)
        train_data=[train_data1,train_data2]
        train_class=np.zeros(num_train,dtype=np.int32)
        i=0
        for file in glob.glob(samples):
            data_class=file.split(os.sep)[-2]
            data_class=data_class.split('-')[-1]
            train_class[i]=classes[data_class]
            inpt=get_inpt(file)
            inpt = np.reshape(inpt,(1,inpt.shape[0],inpt.shape[1],inpt.shape[2]))
            # pred=get_pred(inpt,trained_model)
            # print(pred, esc_csv['category'][pred])
            out = trained_model(inpt,['layer18','layer19'])
            #print( pred.size())
            for j in range(len(out)):
                layer=do_pooling(out[j],globalpoolfn)
                train_data[j][i,:]=layer.detach().numpy()
            i+=1
            
            # print(gout1)
            # print(gout2)
        np.save('results2'+os.sep+mode+'_data_'+test_fold+'_f1.npy',train_data[0])
        np.save('results2'+os.sep+mode+'_data_'+test_fold+'_f2.npy',train_data[1])
        np.save('results2'+os.sep+mode+'_class_'+test_fold+'.npy',train_class)