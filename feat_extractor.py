import librosa as lib
import numpy as np
import network_architectures as netark
import torch.nn.functional as Fx
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from torchvision import datasets
from torch.autograd import Variable
import sys,os
from collections import OrderedDict
import extractor as exm
import pandas as pd
import time
import copy
import matplotlib.pyplot as plt
usegpu = False


n_fft = 1024
hop_length = 512
n_mels = 128
trainType = 'weak_mxh64_1024'
pre_model_path = 'weak_feature_extractor-master'+os.sep+'mx-h64-1024_0d3-1.17.pkl'
featType = 'layer17' # or layer 19 -  layer19 might not work well
globalpoolfn = Fx.avg_pool2d # can use max also
netwrkgpl = Fx.avg_pool2d # keep it fixed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(netx,modpath,featType):
    #load through cpu -- safest
    state_dict = torch.load(modpath,map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    val=featType.split('layer')[-1]
    val=int(val)
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k[7:]
        else:
            name = k
        id=name.split('.')[0]
        id=id.split('layer')[-1]
        id=int(id)
        if(id>val):
            break
        new_state_dict[name] = v
    netx.load_state_dict(new_state_dict)
    return

def getFeat(extractor,indata):
    # return pytorch tensor 
    extractor.eval()
    #indata = Variable(torch.Tensor(inpt))#,volatile=True)
    if usegpu:
        indata = indata.cuda()

    pred = extractor(indata)
    #print( pred.size())
    if len(pred.size()) > 2:
        gpred = globalpoolfn(pred,kernel_size=pred.size()[2:])
        gpred = gpred.view(gpred.size(0),-1)

    return gpred

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # Pool segments into one global value
                    if len(outputs.size()) > 2:
                        gpred = globalpoolfn(outputs,kernel_size=outputs.size()[2:])
                        gpred = gpred.view(gpred.size(0),-1)
                    _, preds = torch.max(gpred, 1)
                    loss = criterion(gpred, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def get_model():
    netType = getattr(netark,trainType,featType)
    netx = netType(527,netwrkgpl,featType)
    load_model(netx,pre_model_path,featType)
    #
    
    if usegpu:
        netx.cuda()
    
    feat_extractor = exm.featExtractor(netx,featType)
    layer18 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=2),nn.BatchNorm2d(1024),nn.ReLU())
    feat_extractor.add_layer(layer18,'layer18')
    layer19 = nn.Sequential(nn.Conv2d(1024,50,kernel_size=1),nn.Sigmoid())
    feat_extractor.add_layer(layer19,'layer19',True)
    return feat_extractor

#keep sample rate 44100, no need 
def get_inpt(filename):#,srate=44100):
    srate=44100
    try:
        y, sr = lib.load(filename,sr=None)
    except:
        raise IOError('Give me an audio  file which I can read!!')
    
    if len(y.shape) > 1:
        print( 'Mono Conversion') 
        y = lib.to_mono(y)

    if sr != srate:
        print ('Resampling to {}'.format(srate))
        y = lib.resample(y,sr,srate)

        
    mel_feat = lib.feature.melspectrogram(y=y,sr=srate,n_fft=n_fft,hop_length=hop_length,n_mels=128)
    inpt = lib.power_to_db(mel_feat).T
    #plt.imshow(inpt)
    #plt.show()
    #quick hack for now
    if inpt.shape[0] < 128:
        inpt = np.concatenate((inpt,np.zeros((128-inpt.shape[0],n_mels))),axis=0)
    inpt = np.reshape(inpt,(1,inpt.shape[0],inpt.shape[1]))
    #inpt2=np.concatenate((inpt,np.zeros((2,inpt.shape[1],inpt.shape[2]))))
    # input needs to be 4D, batch_size X 1 X inpt_size[0] X inpt_size[1]
    #inpt = np.reshape(inpt,(1,1,inpt.shape[0],inpt.shape[1]))
    #print (inpt2.shape)
    indata = Variable(torch.Tensor(inpt))
    return indata
    
def get_pred(inpt,model):    
    pred = getFeat(model,inpt)
    #numpy arrays
    feature = pred.data.cpu().numpy()
    #print('Feature:',feature)
    return np.where(feature==np.max(feature))[-1][0]
    # prediction for each segment in each column
    #return feature

def print_weights():
    #model=get_model()
    model = torch.load('trained_model-3.mdl')
    for param in model.parameters():
        print(param.data)
    return
    
def get_class_dict():
    return {'dog':0,'rooster':1,'pig':2,'cow':3,'frog':4,'cat':5,'hen':6,'insects':7,'sheep':8,'crow':9,
    'rain':10,'sea_waves':11,'crackling_fire':12,'crickets':13,'chirping_birds':14,'water_drops':15,'wind':16,'pouring_water':17,'toilet_flush':18,'thunderstorm':19,
    'crying_baby':20,'sneezing':21,'clapping':22,'breathing':23,'coughing':24,'footsteps':25,'laughing':26,'brushing_teeth':27,'snoring':28,'drinking_sipping':29,
    'door_wood_knock':30,'mouse_click':31,'keyboard_typing':32,'door_wood_creaks':33,'can_opening':34,'washing_machine':35,'vacuum_cleaner':36,'clock_alarm':37,'clock_tick':38,'glass_breaking':39,
    'helicopter':40,'chainsaw':41,'siren':42,'car_horn':43,'engine':44,'train':45,'church_bells':46,'airplane':47,'fireworks':48,'hand_saw':49}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='path to ESC-50 directory', required=True)
    parser.add_argument('--sorted', help='path to ESC-50 sorted directory', required=True)
    args = parser.parse_args()
    
    esc_csv = pd.read_csv(os.path.join(os.path.abspath(args.source), 'meta', 'esc50.csv'))
    audio_path = os.path.join(os.path.abspath(args.source), 'audio')
    classes=get_class_dict()
    confusion=np.zeros((50,50))
    
    test_fold=str(5)
    if(not os.path.exists('trained_model-'+test_fold+'.mdl')):
        data_dir = os.path.join(os.path.abspath(args.sorted),'esc50-'+test_fold)
        image_datasets = {x: datasets.DatasetFolder(os.path.join(data_dir, x), get_inpt,['wav']) for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        
        feat_extractor=get_model()
        criterion = nn.CrossEntropyLoss()
        optimizer_conv = optim.SGD(feat_extractor.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1) # Decay LR by a factor of 0.1 every 7 epochs
        trained_model = train_model(feat_extractor, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=50)
        #save model
        torch.save(trained_model,'trained_model-'+test_fold+'.mdl')
    else:
        trained_model = torch.load('trained_model-'+test_fold+'.mdl')
    if(not os.path.exists('confusion-'+test_fold+'.npy')):
        for i in range(len(esc_csv['filename'])):
            if(int(esc_csv['fold'][i])!=int(test_fold)):
                continue
            inpt=get_inpt(audio_path+os.sep+esc_csv['filename'][i])
            inpt = np.reshape(inpt,(1,inpt.shape[0],inpt.shape[1],inpt.shape[2]))
            pred=get_pred(inpt,trained_model)
            category=classes[esc_csv['category'][i]]
            confusion[category][pred]+=1
        np.save('confusion-'+test_fold,confusion)
    else:
        confusion = np.load('confusion-'+test_fold+'.npy')
    correct_count=0
    for i in range(50):
        correct_count += confusion[i][i]
    accuracy=correct_count/np.sum(confusion)
    print(accuracy)
    plt.imshow(confusion)
    plt.show()
    # np.set_printoptions(threshold=np.nan)
    # np.core.arrayprint._line_width = 250
    # print(confusion)
