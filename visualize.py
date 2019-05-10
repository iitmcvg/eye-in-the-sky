import keras
import numpy as np 
from osgeo import gdal
import cv2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import glob
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing.image import *
from keras.callbacks import *
from keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
from evaluate import classes, convert_to_binary, compare



def unet(pretrained_weights = None,input_size = (64,64,4)):
    inputs = keras.layers.Input(input_size)
    noise=keras.layers.GaussianNoise(0.1)(inputs)
    conv1 = keras.layers.Conv2D(64, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(noise)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(64, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#32x32x32
    conv2 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'elu' ,padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128,3,activation = 'elu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128,3,activation = 'elu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#64x16x16
    conv3 = Conv2D(256, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#128x8x8
    conv4 = Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
#512x16x16
    up6 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv4))
    merge6 = concatenate([conv3,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)

    conv6 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)
#256x32x32
    up7 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv2,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation ='elu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

#64x64
    up8 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv1,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'elu' , padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 3, activation = 'elu' , padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, 3, activation = 'elu',  padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, 3, activation = 'elu',  padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, 3, activation = 'elu',  padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(1, 3, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv8) 
    


    model = keras.models.Model(inputs = inputs, outputs = conv8)  


    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model






def shallowunet(pretrained_weights = None,input_size = (64,64,4)):
    inputs = keras.layers.Input(input_size)
    noise=keras.layers.GaussianNoise(0.1)(inputs)
    conv1 = keras.layers.Conv2D(64, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(noise)
    #conv1 = keras.layers.BatchNormalization()(conv1)
    #conv1 = keras.layers.Conv2D(64, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#32x32x32
    conv2 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'elu' ,padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128,3,activation = 'elu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv2)
    conv2 = BatchNormalization()(conv2)
    #conv2 = Conv2D(128,3,activation = 'elu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv2)
    #conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#64x16x16
    conv3 = Conv2D(256, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    #conv3 = Conv2D(256, 3,activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv3)
    #conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#128x8x8
    conv4 = Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    #conv4 = Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #conv4 = BatchNormalization()(conv4)
#512x16x16
    up6 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv4))
    merge6 = concatenate([conv3,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)

    #conv6 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv6)
    #conv6 = BatchNormalization()(conv6)
#256x32x32
    up7 = Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv2,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation ='elu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

#64x64
    up8 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv1,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'elu' , padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    #conv8 = Conv2D(64, 3, activation = 'elu' , padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv8)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, 3, activation = 'elu',  padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)
    #conv8 = Conv2D(32, 3, activation = 'elu',  padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv8)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, 3, activation = 'elu',  padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(1, 3, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv8) 
    


    model = keras.models.Model(inputs = inputs, outputs = conv8)  
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def image_seg( img, crop_shape,stride,model ):


    '''Function that cuts image into out_shape shaped sub-images 
    and reflect pads excess bits'''

    img_original = np.array(img)
    out_shape = crop_shape
    img_shape = img_original.shape
    img = np.pad(img, ((0,out_shape[0]),(0,out_shape[1]),(0,0)),'reflect')
    x_limit = img_shape[0]/stride
    y_limit = img_shape[1]/stride

    if x_limit != int(x_limit):
        x_limit = int(x_limit) + 1
    else:
        x_limit = int( x_limit )
    if y_limit != int(y_limit):
        y_limit = int(y_limit) + 1
    else:
        y_limit = int( y_limit)

    segmented = np.zeros( (img_shape[0]+out_shape[0], img_shape[1]+out_shape[1],1) )

    for x_ind in range(x_limit):
        for y_ind in range(y_limit):
            x_t = x_ind*stride
            y_t = y_ind*stride
            crop = img[ x_t: x_t + out_shape[0],y_t: y_t + out_shape[1],:]
            out=model.predict(np.expand_dims(crop,axis=0))
            segmented[ x_t: x_t+out_shape[0] , y_t:y_t +out_shape[1]] = np.maximum(segmented[ x_t: x_t+out_shape[0] , y_t:y_t +out_shape[1]] ,out[0])
				
    segmented = segmented[ :img_shape[0], :img_shape[1]]
    segmented=cv2.GaussianBlur(segmented,(5,5),1)
    kernel=np.ones((5,5),np.uint8)
    #segmented=cv2.morphologyEx(segmented,cv2.MORPH_OPEN,kernel)

    # ret,segmented=cv2.threshold(segmented,0.68,1,cv2.THRESH_BINARY)
    # kernel=np.ones((5,5))
    # segmented=cv2.morphologyEx(segmented,kernel,cv2.MORP)
    # cv2.imshow('predicted',segmented )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return segmented


def tiff_to_np(filename,n=4):
''' Converts the tif file into a numpy array '''
	ds = gdal.Open(filename, gdal.GA_ReadOnly)
	arys=[]
	for i in range(1, ds.RasterCount+1):
	    arys.append(ds.GetRasterBand(i).ReadAsArray())
	arys = np.asarray(arys)
	print(arys.shape)
	w = arys.shape[1]
	h = arys.shape[2]
	img = np.zeros((w,h,n))
	img[:,:,0] = arys[0,:,:]/255.0
	img[:,:,1] = arys[1,:,:]/255.0
	img[:,:,2] = arys[2,:,:]/255.0
	if n==4 :
		img[:,:,3]=arys[3,:,:]/255.0
	return img

def equalize2(img,n=4):
'''' Function for histogram equalization. The channel number can be varied for 4-channel or 3 -channel images '''
	equ = np.zeros(img.shape)
	x = np.zeros(img[:,:,0].shape)
	x = img[:,:,0]
	x=np.uint8(cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX))
	equ[:,:,0] = cv2.equalizeHist(x)
	x = img[:,:,1]
	x=np.uint8(cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX))
	equ[:,:,1] = cv2.equalizeHist(x)
	x = img[:,:,2]
	x=np.uint8(cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX))
	equ[:,:,2] = cv2.equalizeHist(x)
	equ = equ/255.0
	if n==4:
		equ[:,:,3]=img[:,:,3]/255.0
	return equ

def convert_to_labels(img):
''' '''
    dims=np.shape(img)
    result=np.zeros((dims[0],dims[1],3))
    result=cv2.normalize(result,None,0,255,cv2.NORM_MINMAX)
    for i in range(dims[0]):
        for j in range(dims[1]):
            ind=np.argmax(img[i,j])
            
            #print(ind)
            if ind==0:
                result[i,j]=np.array([0,255,0])
            if ind==1:
                result[i,j]=np.array([0,0,0])
            if ind==2:
                result[i,j]=np.array([255,255,0])
            if ind==3:
                result[i,j]=np.array([150,80,0])
            if ind==4:
                result[i,j]=np.array([255,255,255])
    return result

#hardrail2 is with threshold 15 
#hardrail is threshold 1
#roadweighted was done with pos_weight=1.5
#roadweighted2 pos_weight=1.25

def metric_eval(i,mode,model_name,model_weight,thresh,stride,index):
	"""Funtion that evaluates a metric for the given image """
	x=tiff_to_np("./The-Eye-in-the-Sky-dataset/sat/%d.tif"%i)
	model=model_name(model_weight)
	x=equalize2(x)
	gt = tiff_to_np("./The-Eye-in-the-Sky-dataset/gt/%d.tif"%i,3)
	#print(gt.shape,gt.dtype)
	gt=np.uint8(gt*255)
	y_pred = image_seg(x,(64,64),stride,model)
	#print(y_pred.shape)

	y_pred = np.uint8(y_pred>thresh)
	#plt.imshow(y_pred,cmap='gray')
	#plt.show()

	print(compare(mode,y_pred,gt,index))


def visualize(model_name,model_weight_name,train_mode,thresh,stride,index):
	"""
	Function that can evaluate metrics on train data and visualize on test. 
		model_name : unet (54 layers) 
					shallowunet (38 layers) 
		model_weight_name : name of the weight file 
		train_mode : 0 - visualize test
					 1 - visualize and evaluate metrics on train data 
		thresh - threshold for evaluating metric/visualizing test
		stride - the image is  broken into 64 x64 patches with this stride. stride helps in getting more continuous outputs
		index : index for encoding the label matrix .Depends on class as :
			    grass 0
				trees 1
				railways 2
				buildings 3
				roads 4
				bare soil 5
				oceans 6
				swimming pool 7
	"""
	model_weight=str('./model_{}.h5'.format(model_weight_name))
	model=model_name(pretrained_weights=model_weight)

	if train_mode == 0: #test
		num = 6
		for i in range(num):
			i+=1
			x=tiff_to_np('./The-Eye-in-the-Sky-test-data/sat_test/%d.tif'%i)
			#label=tiff_to_np('./The-Eye-in-the-Sky-dataset/gt/%d.tif'%i,n=3)
			print(np.shape(x))
			x= equalize2(x)
			#label=np.uint8(label*255)
			y=image_seg(x,(64,64),stride,model)
			print("What is Y :",np.shape(y))
			y=np.uint8(y*255)
			x=x[:,:,:3]
			
			ret,y = cv2.threshold(y,np.uint8(thresh*255),255,cv2.THRESH_BINARY)
			#print(np.max(y), np.max(label))
			#plt.subplot(131)
			#plt.imshow(label)
			#print(compare(1,y,label,index))
			plt.subplot(121)	
			plt.imshow(x[:,:,::-1])
			plt.subplot(122)
			plt.imshow(np.squeeze(y),cmap='gray')
			plt.show()
			
	if train_mode == 1:
		num = 14
	
		for i in range(num):
			i+=1
			x=tiff_to_np('./The-Eye-in-the-Sky-dataset/sat/%d.tif'%i)
			label=tiff_to_np('./The-Eye-in-the-Sky-dataset/gt/%d.tif'%i,n=3)
			print(np.shape(x))
			x= equalize2(x)
			label=np.uint8(label*255)
			y=image_seg(x,(64,64),stride,model)
			#print(np.shape(y))
			y=np.uint8(y*255)
			x=x[:,:,:3]
			#ret,y = cv2.threshold(y,np.uint8(thresh*255),255,cv2.THRESH_BINARY)
			print(np.max(y), np.max(label))
			metric_eval(i,1,model_name,model_weight,thresh,stride,index)
			#plt.subplot(131)
			#plt.imshow(label)
			#print(compare(1,y,label,index))
			plt.subplot(121)	
			plt.imshow(label)
			plt.subplot(122)
			plt.imshow(np.squeeze(y),cmap='gray')
			# plt.show()
			print(np.shape(y))
			np.save('./outputs/%s/%d-stitched'%(model_weight_name,i),y)
			print("saved")
			#cv2.waitKey(0) & 0xFF
			#cv2.destroyAllWindows()


visualize(unet,'ckpt_file',1,0.6,32,3)











