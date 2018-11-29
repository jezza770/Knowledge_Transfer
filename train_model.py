from feat_extractor import train_model,get_model,get_inpt,get_class_dict,get_pred
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from torchvision import datasets
import os
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='path to ESC-50 directory', required=True)
    parser.add_argument('--sorted', help='path to ESC-50 sorted directory', required=True)
    parser.add_argument('--testfold',help='which fold of ESC-50 will be for testing (1-5)',choices=['1','2','3','4','5'],required=True)
    args = parser.parse_args()
    
    loadlayer = 'layer18' # the layer to load up to
    trainlayer= 'layer17' # layers below this layer will not be updated (inclusive)
    
    esc_csv = pd.read_csv(os.path.join(os.path.abspath(args.source), 'meta', 'esc50.csv')) # File contains labels for each sample in ESC-50
    audio_path = os.path.join(os.path.abspath(args.source), 'audio') # Folder contains ESC-50 samples in original form, easier for testing
    classes=get_class_dict()
    confusion=np.zeros((50,50))
    test_fold=args.testfold
    if(os.path.exists('trained_model-'+test_fold+'.mdl')):
        print('Model file already exists')
        exit()
    # ESC-50 samples as processed by prepare_esc50.py for the appropriate test fold
    data_dir = os.path.join(os.path.abspath(args.sorted),'esc50-'+test_fold) 
    # Load the Audioset weights and modify to create N1 structure
    feat_extractor=get_model(loadlayer,trainlayer)
    # Add F_T layer
    layer19 = nn.Sequential(nn.Conv2d(1024,50,kernel_size=1),nn.ReLU())
    feat_extractor.add_layer(layer19,'layer19',True)
    criterion = nn.CrossEntropyLoss()
    # Unsure of what parameters to set here
    optimizer_conv = optim.SGD(feat_extractor.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1) # Decay LR by a factor of 0.1 every 7 epochs
    # Train the appropriate layers of the loaded model
    trained_model = train_model(feat_extractor, criterion, optimizer_conv, exp_lr_scheduler, data_dir, num_epochs=50)
    #save model
    torch.save(trained_model,'trained_model-'+test_fold+'.mdl')
    # Generate confusion matrix by passing each sample in the test fold through the trained model
    for i in range(len(esc_csv['filename'])):
        if(int(esc_csv['fold'][i])!=int(test_fold)):
            continue
        inpt=get_inpt(audio_path+os.sep+esc_csv['filename'][i])
        inpt = np.reshape(inpt,(1,inpt.shape[0],inpt.shape[1],inpt.shape[2]))
        pred=get_pred(inpt,trained_model)
        category=classes[esc_csv['category'][i]]
        confusion[category][pred]+=1
    np.save('confusion-'+test_fold,confusion)
    # trained_model = torch.load('trained_model-'+test_fold+'.mdl')
    # confusion = np.load('confusion-'+test_fold+'.npy')
    correct_count=0
    for i in range(50):
        correct_count += confusion[i][i]
    accuracy=correct_count/np.sum(confusion)
    print(accuracy)
    plt.imshow(confusion,cmap='gray')
    plt.show()
    # np.set_printoptions(threshold=np.nan)
    # np.core.arrayprint._line_width = 250
    # print(confusion)
