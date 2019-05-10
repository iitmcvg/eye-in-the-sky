from tensorflow.python import keras as keras
import numpy as np 
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
import glob
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import *
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras import regularizers
import tensorflow as tf
from losses import bce_dice_loss, dice_coeff,dice_loss , weighted_bce_dice_loss , weighted_dice_loss 

def unet(pretrained_weights = None,input_size = (64,64,4)):
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

class TensorBoardKeras(keras.callbacks.Callback):
    def __init__(self,model,logdir):
        self.model=model
        self.logdir=logdir
        self.sess=K.get_session()

        self.img=tf.placeholder(shape=(32,32,3),dtype=tf.float32)
        tf.summary.image('out',self.img)
        self.train_loss=tf.placeholder((1),dtype=tf.float32)
        tf.summary.scalar('train_loss',self.train_loss)
        self.train_acc=tf.placeholder((1),dtype=tf.float32)
        tf.summary.scalar('train_acc',self.train_acc)
        self.val_loss=tf.placeholder((1),dtype=tf.float32)
        tf.summary.scalar('val_loss',self.val_loss)
        self.val_acc=tf.placeholder((1),dtype=tf.float32)
        tf.summary.scalar('val_acc',self.val_acc)
        self.sum=tf.summary.merge_all()
        self.write=tf.summary.FileWriter(self.logdir)



    def get_img(self,model):
        return model.get_tensor_by_name('conv2d_21',dtype=tf.float32)

    def on_epoch_end(self,epoch,model,logs):
        summary=self.sess.run(self.sum,feed_dict={self.img:self.get_img(model),self.train_acc:tf.dtypes.cast(logs['acc'],tf.float32),self.train_loss:tf.dtypes.cast(logs['loss'],tf.float32),self.val_acc:tf.dtypes.cast(logs['val_acc'],tf.float32),self.val_loss:tf.dtypes.cast(logs['val_loss'],tf.float32)})
        self.write.add_summary(summary,epoch)
        self.write.flush()



    def on_epoch_end_cb(self):
        return LambdaCallback(on_epoch_end=lambda epoch, logs:self.on_epoch_end(epoch, logs))


class My_Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.images=[]
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.images.append(self.model.get_tensor_by_name('conv2d_21'))
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return
'''
X_train=[]
Y_train=[]
for img in sorted(glob.glob("./train_soil/images/*.npy")):
    _npy=np.load(img)
    if _npy.ndim == 3:
       X_train.append(_npy)

for lbl in sorted(glob.glob('./train_soil/labels/*.npy')):
    _lbl=np.load(lbl)
    Y_train.append(_lbl)
print(len(X_train))
print(len(Y_train))
X_train=np.asarray(X_train)
Y_train=np.asarray(Y_train)
'''
X_train=np.load('./X_train.npy')
Y_train=np.load('./Y_train.npy')
#only buildings
Y_train=Y_train[:,:,:,0]
Y_train=np.expand_dims(Y_train,axis=3)
shape_train=np.shape(Y_train)
print('-train-'+str(shape_train)) 


#Y_train=Y_train.reshape((shape_train[0],shape_train[1]*shape_train[2],shape_train[3]))
#print('after-train-' +str(np.shape(Y_train)))
'''
X_val=[]
Y_val=[]
for img in sorted(glob.glob("./validation_soil/images/*.npy")):
    _npy=np.load(img)
    X_val.append(_npy)

for lbl in sorted(glob.glob('./validation_soil/labels/*.npy')):
    _lbl=np.load(lbl)
    Y_val.append(_lbl)

X_val=np.asarray(X_val)
Y_val=np.asarray(Y_val)
'''
X_val=np.load('./X_val.npy')
Y_val=np.load('./Y_val.npy')
#only railways
Y_val=Y_val[:,:,0]
Y_val=np.expand_dims(Y_val,axis=3)

#Y_val=Y_val.reshape((shape_val[0],shape_val[1]*shape_val[2],shape_val[3]))
#print('after-val-' +str(np.shape(Y_val)))
print('val-' +str(np.shape(Y_val)))


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
#summary=tf.summary.merge_all()
#summary=tf.summary.FileWriter('./logs',session.graph)
K.set_session(session)




data_gen_args = dict(featurewise_center=False,featurewise_std_normalization=False, rotation_range=70,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_val_datagen = ImageDataGenerator(**data_gen_args)
mask_val_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_datagen.fit(X_train, augment=True, seed=seed)
mask_datagen.fit(Y_train, augment=True, seed=seed)

image_datagen.fit(X_val, augment=True, seed=seed)
mask_datagen.fit(Y_val, augment=True, seed=seed)

image_generator = image_datagen.flow(X_train,seed=seed)
mask_generator = image_datagen.flow(Y_train,seed=seed)
train_generator = zip(image_generator, mask_generator)

image_val_generator = image_datagen.flow(X_val,seed=seed)
mask_val_generator = image_datagen.flow(Y_val,seed=seed)
val_generator = zip(image_generator, mask_generator)


x = unet() 
x.compile(optimizer = keras.optimizers.Nadam(lr = 2e-4), loss =bce_dice_loss, metrics = ['accuracy',dice_coeff])


tensorboard=keras.callbacks.TensorBoard(log_dir='./buildshallowlogs', write_images=True)
#custom_callback = My_Callback()
#testing for large epochs for a better mask
#earlystopper = EarlyStopping(patience=20, verbose=1)
checkpointer = ModelCheckpoint('model_buildshallow.h5', verbose=1,save_best_only=True)
results = x.fit_generator(generator=train_generator ,steps_per_epoch=132,validation_steps=40, validation_data=val_generator, epochs=100,callbacks=[ checkpointer,tensorboard])

'''
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
results = x.fit_generator(generator=train_generator ,steps_per_epoch=130,validation_steps=50, validation_data=validation_generator, epochs=50,callbacks=[earlystopper, checkpointer])
'''




