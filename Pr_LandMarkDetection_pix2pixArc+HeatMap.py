# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:44:34 2018

@author: Moha-Thinkpad
"""

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow.keras

import argparse
import tensorflow as tf


from tensorflow.keras import backend as K
#cfg = K.tf.ConfigProto()
#cfg.gpu_options.allow_growth = True
#K.set_session(K.tf.Session(config=cfg))


####################################
########################################################################
####################################


def custom_loss_seg (y_true, y_pred):
    
    
    #A = tensorflow.keras.losses.mean_squared_error(y_true, y_pred)
    B = tensorflow.keras.losses.mean_absolute_error(y_true, y_pred)
    
    return(B)


from tensorflow.keras.layers import Lambda
sum_dim_channel = Lambda(lambda xin: K.sum(xin, axis=3))



def lrelu(x): #from pix2pix code
    a=0.2
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2

    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def lrelu_output_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)

layer_lrelu=Lambda(lrelu, output_shape=lrelu_output_shape)


def PreProcess(InputImages):
    
    #output=np.zeros(InputImages.shape,dtype=np.float)
    InputImages=InputImages.astype(np.float)
    for i in range(InputImages.shape[0]):
        try:
            InputImages[i,:,:,:]=InputImages[i,:,:,:]/np.max(InputImages[i,:,:,:])
#            output[i,:,:,:] = (output[i,:,:,:]* 2)-1
        except:
            InputImages[i,:,:]=InputImages[i,:,:]/np.max(InputImages[i,:,:])
#            output[i,:,:] = (output[i,:,:]* 2) -1
            
    return InputImages


####################################
########################################################################
####################################

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test", "export"])
parser.add_argument("--input_dir",  help="path to folder containing images")
parser.add_argument("--target_dir",  help="where to")
parser.add_argument("--checkpoint",  help="where to ")
parser.add_argument("--output_dir",  help="where to p")
parser.add_argument("--landmarks",  help=" -,-,-")
parser.add_argument("--lr",  help="adam learning rate")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")

# export options
a = parser.parse_args()



a.batch_size=40
a.max_epochs_seg=1
a.lr_seg=0.0001
a.beta1=0.5
a.ngf=64


#a.seed=1


# a.mode="train"
# a.input_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/temp_train_png/'
# a.target_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/temp_train_lm/'
# a.checkpoint='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/Models_lm/'
# a.output_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/Models_lm/'
# a.landmarks='43,43,43'


#a.mode="test"
#a.batch_size=1
#a.input_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/temp_test_png/'
#a.target_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/temp_test_lm/'
#a.checkpoint='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/Models_lm/'
#a.output_dir='C:\\Users\\User\\Desktop\\Example_LoSoCo_Inputs_3_large_heatmaps/Models_lm/'
#a.landmarks='43,43,43'



######## ------------ Config 

#Ind_impo_landmarks_matlab=np.array([5, 6, 15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,41])
#Ind_impo_landmarks_python=Ind_impo_landmarks_matlab-1
#Num_landmarks=25

# 33,23,16 - 29,15, - 30,20,26 - 5,18,21 - 44,17,41 - 28,22,34, - 27,43,37


StrLandmarks=a.landmarks
StrLandmarks=StrLandmarks.split(",")
Ind_impo_landmarks_matlab=np.array([0,0,0])
Ind_impo_landmarks_matlab[0]=int(StrLandmarks[0])
Ind_impo_landmarks_matlab[1]=int(StrLandmarks[1])
Ind_impo_landmarks_matlab[2]=int(StrLandmarks[2])
Ind_impo_landmarks_python=Ind_impo_landmarks_matlab-1
Num_landmarks=3




print('============================')
print('============================')
print(datetime.datetime.now())
print('============================')
print('============================')


#########----------------------DATA

from os import listdir
ImageFileNames=[]
FileNames=listdir(a.input_dir)
for names in FileNames:
    if names.endswith(".png"):
        ImageFileNames.append(names)
#LMFileNames=listdir(a.target_dir)
from skimage import io as ioSK
from numpy import genfromtxt

Images=np.zeros((len(ImageFileNames),256,256,3),dtype=np.uint8)    
#Images_seg=np.zeros((len(ImageFileNames),256,256),dtype=np.uint8)    
LandmarkLocations=np.zeros((len(ImageFileNames),2,44),dtype=np.uint8)



for i in range(len(ImageFileNames)):
    Image = ioSK.imread(a.input_dir+'/'+ImageFileNames[i])
    Images[i,:,:,:]=Image    
    
    FileName=ImageFileNames[i]
    FileName=FileName[:-4]
    
#    Image = ioSK.imread(a.target_dir_seg+'/'+ImageFileNames[i])
#    Images_seg[i,:,:]=Image
            
    Landmarks0 = genfromtxt(a.target_dir+'/'+FileName+'.csv', delimiter=',')    
    Landmarks0 = Landmarks0.astype(int)    
    LandmarkLocations[i,0,:]=Landmarks0[:,0]
    LandmarkLocations[i,1,:]=Landmarks0[:,1]
    
    #Landmarks = np.flip(Landmarks0, axis=1)

#plt.figure()
#plt.imshow(Images[100,:,:,:])
#plt.scatter(LandmarkLocations[100,0,:],LandmarkLocations[100,1,:])



X_train = PreProcess(Images) 
del Images
import gc
gc.collect()

LandmarkLocations_row=LandmarkLocations[:,0,:]
LandmarkLocations_col=LandmarkLocations[:,1,:]
LandmarkLocations_row=LandmarkLocations_row[:,Ind_impo_landmarks_python]
LandmarkLocations_col=LandmarkLocations_col[:,Ind_impo_landmarks_python]
    
from scipy.ndimage import gaussian_filter

Images_HeatMaps=np.zeros((X_train.shape[0],X_train.shape[1],X_train.shape[2],Num_landmarks),dtype=np.float)

Image_heatmap=np.zeros((256,256),dtype=np.float)
for i in range(X_train.shape[0]):
  for k in range(Num_landmarks):
      
#        h=np.argwhere(Images_seg[i,:,:]==2*Ind_impo_landmarks_matlab[k])    
      lms_1=LandmarkLocations_row[i,k]
      lms_2=LandmarkLocations_col[i,k]
      Image_heatmap[:,:]=0
      Image_heatmap[lms_2,lms_1]=1
      Image_heatmap=gaussian_filter(Image_heatmap, sigma=10)
      Image_heatmap=(Image_heatmap/np.max(Image_heatmap))
      Images_HeatMaps[i,:,:,k]=Image_heatmap
        

gc.collect()
      
#plt.figure()
#plt.imshow(np.squeeze(Images_HeatMaps[2,:,:,5]), cmap='gray')
#plt.imshow(Images[2,:,:,:],cmap='jet', alpha=0.5)
#plt.show()    

Y_train_heatmap = PreProcess(Images_HeatMaps) 
del Images_HeatMaps
gc.collect()
#    del Images_seg    



import os
if not os.path.exists(a.checkpoint):
    os.makedirs(a.checkpoint)
    
if not os.path.exists(a.output_dir):
    os.makedirs(a.output_dir)

if a.mode=='test':
    
    checkpoint_model_file=a.checkpoint+'LandMarkModel'

    from tensorflow.keras.models import load_model
    

    print('loading model ...')
    model_final=load_model(checkpoint_model_file+'_weights.h5', custom_objects={ 
                                                                                    'custom_loss_seg': custom_loss_seg, 
                                                                                'layer_lrelu':layer_lrelu, 
                                                                                'lrelu':lrelu, 
                                                                                'lrelu_output_shape':lrelu_output_shape,
                                                                                'tf': tf}) 

    print('model is loaded ')
    Images=np.zeros((len(ImageFileNames),256,256,3),dtype=np.float)            
    newLandmarks=np.zeros((Num_landmarks,2),dtype=np.float16)
    
    Y_test_heatmap=Y_train_heatmap
    X_test=X_train
        
    #    fig = plt.figure()
    #    plt.imshow(X_train[0,:,:,:],cmap='gray', alpha=0.95)
    #    plt.imshow(Y_train_heatmap[0,:,:,:],cmap='jet', alpha=0.5)
    #    plt.grid(True)
        
    pred_example_heatmaps=model_final.predict(X_test[:,:,:,:])  
    print('writing results ...')
    for i in range(len(ImageFileNames)):
        # print(i)
        FileName=ImageFileNames[i]
        FileName=FileName[:-4]        

        lms_pred_all=np.zeros((Num_landmarks,2),dtype=np.int)
        lms_True_all=np.zeros((Num_landmarks,2),dtype=np.int)
        for k in range(Num_landmarks):
    #        plt.figure()
    #        plt.imshow(example_segmentation[0,:,:,i], cmap='gray')
    #        plt.imshow(Y_train_heatmap[0,:,:,:],cmap='jet', alpha=0.5)
    #        plt.show()
    
           
            True_chan=np.squeeze(Y_test_heatmap[i,:,:,k])
            lms_True=np.unravel_index(np.argmax(True_chan, axis=None), True_chan.shape)
            lms_True_all[k,:]=lms_True
            
            Pred_chan=np.squeeze(pred_example_heatmaps[i,:,:,k])
            lms_pred=np.unravel_index(np.argmax(Pred_chan, axis=None), Pred_chan.shape)
            lms_pred_all[k,:]=lms_pred
            
            
#            fig, ax = plt.subplots(1, 2)
#            ax[0].imshow(Y_test_heatmap[i,:,:,i])        
#            ax[1].imshow(pred_example_heatmaps[i,:,:,i])
#            plt.show()
    
        np.savetxt(a.output_dir+FileName+'_pred.csv', 
           lms_pred_all , delimiter=",", fmt='%i')

        np.savetxt(a.output_dir+FileName+'_true.csv', 
           lms_True_all , delimiter=",", fmt='%i')
    
        fig = plt.figure()
        plt.imshow(X_test[i,:,:,:],cmap='jet', alpha=0.9)
        plt.scatter(lms_True_all[:,1],lms_True_all[:,0], marker='+', color='red')
        plt.scatter(lms_pred_all[:,1],lms_pred_all[:,0], marker='x', color='blue')
#        plt.grid(True)
        fig.savefig(a.output_dir+FileName+'.png')
        plt.close(fig)  
        
        


        

if a.mode=='train':
    
    
#    plt.figure()
#    plt.imshow(X_train[90,:,:,:])
#    plt.figure()
#    plt.imshow(Y_train_heatmap[90,:,:,4])
    

    
    
    
    try: # continue training
        checkpoint_model_file=a.checkpoint+'LandMarkModel'

        from tensorflow.keras.models import load_model

        print('======== loading model ...')
        model_4_heatmap=load_model(checkpoint_model_file+'_weights.h5', custom_objects={ 
                                                                                    'custom_loss_seg': custom_loss_seg, 
                                                                                'layer_lrelu':layer_lrelu, 
                                                                                'lrelu':lrelu, 
                                                                                'lrelu_output_shape':lrelu_output_shape,
                                                                                'tf': tf}) 
        print('======== continue training ...')
    except:  # new training
        print('======== new training ...')
        checkpoint_model_file=a.output_dir+'LandMarkModel'
            
        ########### network    
        kernelSize=(4,4)
        InputLayer=tensorflow.keras.layers.Input(shape=(256,256,3))
        e_1=tensorflow.keras.layers.Conv2D(a.ngf, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(InputLayer)
        
        e_2=layer_lrelu(e_1)
        e_2=tensorflow.keras.layers.Conv2D(a.ngf * 2, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_2)
        e_2=tensorflow.keras.layers.BatchNormalization()(e_2)
        
        e_3=layer_lrelu(e_2)
        e_3=tensorflow.keras.layers.Conv2D(a.ngf * 4, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_3)
        e_3=tensorflow.keras.layers.BatchNormalization()(e_3)

        e_4=layer_lrelu(e_3)
        e_4=tensorflow.keras.layers.Conv2D(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_4)
        e_4=tensorflow.keras.layers.BatchNormalization()(e_4)

        e_5=layer_lrelu(e_4)
        e_5=tensorflow.keras.layers.Conv2D(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_5)
        e_5=tensorflow.keras.layers.BatchNormalization()(e_5)

        e_6=layer_lrelu(e_5)
        e_6=tensorflow.keras.layers.Conv2D(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_6)
        e_6=tensorflow.keras.layers.BatchNormalization()(e_6)

        e_7=layer_lrelu(e_6)
        e_7=tensorflow.keras.layers.Conv2D(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_7)
        e_7=tensorflow.keras.layers.BatchNormalization()(e_7)

        e_8=layer_lrelu(e_7)
        e_8=tensorflow.keras.layers.Conv2D(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_8)
        e_8=tensorflow.keras.layers.BatchNormalization()(e_8)

        
        
        
        d_8=e_8
        d_8=tensorflow.keras.layers.Activation('relu')(d_8)
        d_8=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_8)
        d_8=tensorflow.keras.layers.BatchNormalization()(d_8)
        d_8=tensorflow.keras.layers.Dropout(0.5)(d_8)
        
        d_7=tensorflow.keras.layers.concatenate(inputs=[d_8, e_7], axis=3)
        d_7=tensorflow.keras.layers.Activation('relu')(d_7)
        d_7=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_7)
        d_7=tensorflow.keras.layers.BatchNormalization()(d_7)
        d_7=tensorflow.keras.layers.Dropout(0.5)(d_7)  
        
        d_6=tensorflow.keras.layers.concatenate(inputs=[d_7, e_6], axis=3)
        d_6=tensorflow.keras.layers.Activation('relu')(d_6)
        d_6=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_6)
        d_6=tensorflow.keras.layers.BatchNormalization()(d_6)
        d_6=tensorflow.keras.layers.Dropout(0.5) (d_6)
        
        d_5=tensorflow.keras.layers.concatenate(inputs=[d_6, e_5], axis=3)
        d_5=tensorflow.keras.layers.Activation('relu')(d_5)
        d_5=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_5)
        d_5=tensorflow.keras.layers.BatchNormalization()(d_5)
        d_5=tensorflow.keras.layers.Dropout(0.5) (d_5)
        
        d_4=tensorflow.keras.layers.concatenate(inputs=[d_5, e_4], axis=3)
        d_4=tensorflow.keras.layers.Activation('relu')(d_4)
        d_4=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 4, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_4)
        d_4=tensorflow.keras.layers.BatchNormalization()(d_4)
        
        d_3=tensorflow.keras.layers.concatenate(inputs=[d_4, e_3], axis=3)
        d_3=tensorflow.keras.layers.Activation('relu')(d_3)
        d_3=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 2, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_3)
        d_3=tensorflow.keras.layers.BatchNormalization()(d_3)
        
        d_2=tensorflow.keras.layers.concatenate(inputs=[d_3, e_2], axis=3)
        d_2=tensorflow.keras.layers.Activation('relu')(d_2)
#        d_2=tensorflow.keras.layers.Conv2DTranspose(a.ngf, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_2)
        d_2=tensorflow.keras.layers.Conv2DTranspose(a.ngf, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_2)
        d_2=tensorflow.keras.layers.BatchNormalization()(d_2)
        
        
        d_1=tensorflow.keras.layers.concatenate(inputs=[d_2, e_1], axis=3)
        d_1=tensorflow.keras.layers.Activation('relu')(d_1)
        d_1=tensorflow.keras.layers.Conv2DTranspose(Num_landmarks, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_1)        
        HeatMaps=tensorflow.keras.layers.Activation('sigmoid', name='last_layer_of_decoder')(d_1)
                
        model_4_heatmap=Model(inputs=InputLayer, outputs=HeatMaps)
       

    ###########Train
        
    print('trainable_count =',int(np.sum([K.count_params(p) for p in set(model_4_heatmap.trainable_weights)])))
    print('non_trainable_count =', int(np.sum([K.count_params(p) for p in set(model_4_heatmap.non_trainable_weights)])))   
       
    # fix random seed for reproducibility
    seed = 1    
    import random    
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
    #### compile and train the model
    UsedOptimizer=optimizers.Adam(lr=a.lr_seg, beta_1=a.beta1)
    model_4_heatmap.compile(loss=custom_loss_seg, optimizer=UsedOptimizer)        
    History=model_4_heatmap.fit(X_train, Y_train_heatmap,
            batch_size=a.batch_size, shuffle=True, validation_split=0.05,
        epochs=a.max_epochs_seg,
            verbose=1)
    
    
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.grid()
    plt.savefig(a.output_dir+'History_'+str(a.lr)+'.png')
    plt.close()
    
    import pickle        
    Dict={'History_loss_train':History.history['loss'],
          'History_loss_val':History.history['val_loss'],}
    pickle.dump( Dict, open(a.output_dir+'History_'+str(a.lr)+'.pkl', "wb" ) )
    
    
    # show an exemplary result    
    Num_example_train=0
    pred_example_heatmaps=model_4_heatmap.predict(X_train[Num_example_train:Num_example_train+1,:,:,:])    
    lms_pred_all=np.zeros((Num_landmarks,2),dtype=np.int)
    lms_True_all=np.zeros((Num_landmarks,2),dtype=np.int)
    for i in range(Num_landmarks):
#        plt.figure()
#        plt.imshow(example_segmentation[0,:,:,i], cmap='gray')
#        plt.imshow(X_train[0,:,:,:],cmap='jet', alpha=0.5)
#        plt.show()
        Pred_chan=np.squeeze(pred_example_heatmaps[0,:,:,i])
        lms_pred=np.unravel_index(np.argmax(Pred_chan, axis=None), Pred_chan.shape)
        lms_pred_all[i,:]=lms_pred
       
        True_chan=np.squeeze(Y_train_heatmap[Num_example_train,:,:,i])
        lms_True=np.unravel_index(np.argmax(True_chan, axis=None), True_chan.shape)
        lms_True_all[i,:]=lms_True
        
        
#        fig, ax = plt.subplots(1, 2)
#        ax[0].imshow(Y_train_heatmap[Num_example_train,:,:,i])        
#        ax[1].imshow(pred_example_heatmaps[0,:,:,i])
#        plt.show()

    fig = plt.figure()
    plt.imshow(X_train[Num_example_train,:,:,:],cmap='jet', alpha=0.9)
    plt.scatter(lms_True_all[:,1],lms_True_all[:,0], marker='+', color='red')
    plt.scatter(lms_pred_all[:,1],lms_pred_all[:,0], marker='x', color='blue')
    plt.grid(True)
#    fig.savefig('scatter-result'+str(i)+'_pred.png')
    plt.close(fig) 
    
    
    
    print('===========training done=================')
    print('============================')
    print(datetime.datetime.now())
    print('============================')
    print('============================')
                
        
    print('Saving model ...')
    model_4_heatmap.save(checkpoint_model_file+'_weights.h5')



