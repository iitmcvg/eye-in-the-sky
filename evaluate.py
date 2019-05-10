import numpy as np
import sklearn.metrics as m

def classes(index,pixel):
    '''
    Function for on-hot encoding of the images
    index vs class
    grass 0
    trees 1
    railways 2
    buildings 3
    roads 4
    bare soil 5
    oceans 6
    swimming pool 7

    '''
    if (pixel[0] == 150 and pixel[1] == 150 and pixel[2] == 255):
        pixel=np.array([0,0,0,0,0,0,0,1]) #swimming pool is background
    elif (pixel[0] == 0 and pixel[1] == 255 and pixel[2] == 0):
        pixel=np.array([1,0,0,0,0,0,0,0]) #grass
    elif (pixel[0] == 255 and pixel[1] ==255 and pixel[2] == 0):
        pixel=np.array([0,0,1,0,0,0,0,0]) #railway
    elif (pixel[0] == 100 and pixel[1] == 100 and pixel[2] == 100):
        pixel=np.array([0,0,0,1,0,0,0,0]) #buildings
    elif (pixel[0] == 0 and pixel[1] ==125 and pixel[2] == 0):
        pixel=np.array([0,1,0,0,0,0,0,0]) # forests
    elif (pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 150):
        pixel=np.array([0,0,0,0,0,0,1,0]) #oceans
    elif (pixel[0] == 0 and pixel[1] ==0 and pixel[2] == 0):
        pixel=np.array([0,0,0,1,0,0,0,0]) #roads
    elif (pixel[0] == 150 and pixel[1] ==80 and pixel[2] == 0):
        pixel=np.array([0,0,0,0,0,1,0,0]) #baresoil
    else :
        pixel=np.array([0,0,0,0,0,0,0,0]) #background

    return pixel[index]


def convert_to_binary(gt,index):
    w,h,c = gt.shape
    x= np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            x[i,j] = classes(index,gt[i,j,:])
    return x


def compare(mode,y_pred,y_true,index):
    #mode = 0 means confusion matrix
    y_true = convert_to_binary(y_true,index)
    #print(y_true.shape)
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    if(mode==0):
        return m.confusion_matrix(y_pred,y_true)
    if(mode==1):
        return m.cohen_kappa_score(y_pred,y_true)


