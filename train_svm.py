import numpy as np
import cv2
import glob,os,sys
import csv
from PIL import Image
classes=[]
with open('model.csv','r') as csvfile:
    fr=csv.reader(csvfile)
    for row in fr:
        classes=classes+row
classes.sort()
train_folder="S:\\ShortTimeSignals\\MIVIA Audio Events Dataset\\MIVIA_DB4_dist\\training\\mod_features"
test_folder="S:\\ShortTimeSignals\\MIVIA Audio Events Dataset\\MIVIA_DB4_dist\\test\\mod_features"


img_size=[25,25]
num_classes=50



def generate_data():
# if((not os.path.isfile('train_data.npy')) and (not os.path.isfile('train_labels.npy'))):
    print('Generating training data')
    num_train=len(glob.glob(train_folder+os.sep+'*'+os.sep+'*.png'))
    train_images=np.zeros((num_train,img_size[0]*img_size[1]),dtype=np.float32)
    train_class=np.zeros(num_train,dtype=np.int32)
    i=0
    for file in glob.glob(train_folder+os.sep+'*'+os.sep+'*.png'):
        img_class=file.split(os.sep)[-2]
        #class_count[classes.index(img_class)]+=1
        train_class[i]=classes.index(img_class)
        im=Image.open(file)
        temp=np.array(im.getdata())
        for pixel in range(len(temp)):
            train_images[i][pixel]=temp[pixel][0]
        i+=1
        if(i%10==0):
            print('Image',i,'of',num_train,end='\r')
    np.save('train_data.npy',train_images)
    np.save('train_labels.npy',train_class)
    print()
# else:
    # print('Loading training data')
    

# if((not os.path.isfile('test_data.npy')) and (not os.path.isfile('test_labels.npy'))):  
    # print('Generating test data')
    # num_test=len(glob.glob(test_folder+os.sep+'*'+os.sep+'*.png')) 
    # test_images=np.zeros((num_test,img_size[0]*img_size[1]),dtype=np.float32)
    # test_class=np.zeros(num_test,dtype=np.int32)
    # i=0
    # for file in glob.glob(test_folder+os.sep+'*'+os.sep+'*.png'):
        # img_class=file.split(os.sep)[-2]
        # test_class[i]=classes.index(img_class)
        # im=Image.open(file)
        # temp=np.array(im.getdata())
        # for pixel in range(len(temp)):
            # test_images[i][pixel]=temp[pixel][0]
        # i+=1
        # if(i%10==0):
            # print('Image',i,'of',num_test,end='\r')
    # np.save('test_data.npy',test_images)
    # np.save('test_labels.npy',test_class)
    # print()
# else:
    # print('Loading test data')
    # test_images=np.load('test_data.npy')
    # test_class=np.load('test_labels.npy')

if __name__ == '__main__':
    test_fold='1'
    layer='f2'
    folder='results2'+os.sep+'N2-max'+os.sep
    train_data=np.load(folder+'train_data_'+test_fold+'_'+layer+'.npy')
    train_class=np.load(folder+'train_class_'+test_fold+'.npy')
    test_data=np.load(folder+'val_data_'+test_fold+'_'+layer+'.npy')
    test_class=np.load(folder+'val_class_'+test_fold+'.npy')

    if(not os.path.isfile(folder+'svm_model-'+test_fold+'_'+layer+'.dat')):
        print('Training SVM')
        SVM=cv2.ml.SVM_create()
        SVM.setKernel(cv2.ml.SVM_LINEAR)
        # SVM.setDegree(0.0)
        #SVM.setGamma(1e-6)
        # SVM.setCoef0(0.0)
        SVM.setC(3.0)
        # SVM.setNu(0.0)
        # SVM.setP(0.0)
        # SVM.setClassWeights(None)
        SVM.train(train_data,cv2.ml.ROW_SAMPLE,train_class)
        SVM.save(folder+'svm_model-'+test_fold+'_'+layer+'.dat')
    else:
        print('Loading SVM')
        SVM=cv2.ml.SVM_load(folder+'svm_model-'+test_fold+'_'+layer+'.dat')
    preds=SVM.predict(test_data)[1]
    confusion=np.zeros((num_classes,num_classes))
    class_count=[0]*num_classes
    pred_count=[0]*num_classes
    for i in range(len(test_class)):
        class_count[test_class[i]]+=1
        if(preds[i]==test_class[i]):
            pred_count[int(preds[i])]+=1
        confusion[test_class[i]][int(preds[i])]+=1
            
    #print(classes)
    print(pred_count)
    print(class_count)
    print('Accuracy:',np.sum(pred_count)/np.sum(class_count))
    print(confusion)
        