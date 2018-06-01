import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential, Model                                        
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications import *
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.optimizers import SGD, Nadam, adam

##---------------------------------------------------------------------------##
##IMPORT DATA AND DUMMY ENCODE
##---------------------------------------------------------------------------##

path = "C:/Users/Grant Feng/Google Drive/DataScienceWork/DL/Assignment-2-Dataset-Round-1/"

trainpath = path + 'train-set/'
valpath = path + 'vali-set/'

traintxt = pd.read_table(path + 'train.txt', header = None, sep = ' ')
valtxt = pd.read_table(path + 'vali.txt', header = None, sep = ' ')

##Order the traintxt lists in the same manner as the train/test lists so we can extract the labels
traintxt = traintxt.sort_values(0).reset_index().drop('index',1)
valtxt = valtxt.sort_values(0).reset_index().drop('index',1)

##Get image names
trainimg = os.listdir(trainpath)
valimg = os.listdir(valpath)

##Make path to each image
X_train = [trainpath + i for i in trainimg]
X_val = [valpath + i for i in valimg]

##Labels data
trainy = traintxt[1]
valy = valtxt[1]

##Dummy labels data
numlabels = len(pd.unique(trainy))
y_train = np_utils.to_categorical(trainy, numlabels)
y_val = np_utils.to_categorical(valy, numlabels)

##View an image
img = cv2.imread(X_train[5355])
plt.imshow(img)
plt.show()

##---------------------------------------------------------------------------##
##AUGMENTATION AND BATCH FUNCTIONS
##---------------------------------------------------------------------------##

##We need image augmentations, batch function and then we can make a conv net.
##First - Augmentations - Resize, rotate, shift, and shear seem most relevant to digit and letter images.
##Flipping images make no sense.

##Simple resize
img1 = cv2.resize(img, (224,224))
plt.imshow(img1)
plt.show()
plt.show()

##Rotate function - Most digits and characters are always either slanted to the right or completely vertical 
# Thus we will make our rotate function only rotate right - we allow there to be a chance of slight left rotates       
def rotate(img, range, random = True):
    if len(img.shape) == 3:
        rows,cols, chan = img.shape
    else:
        rows,cols = img.shape
    if random:
        R = np.random.randint(-range, range)
    else:
        R = range
    M = cv2.getRotationMatrix2D((cols/2,rows/2),R,1)
    dst = cv2.warpAffine(img,M,(cols,rows), borderMode = cv2.BORDER_CONSTANT, borderValue = (255,255,255))
    return dst

img2 = rotate(img1, 15)
plt.imshow(img2)
plt.show()

##Shift function - Most likely unnecessary as all digits/characters in dataset are centred
def Horizontalshift(img, range = 0.1):
    '''range is a float determining the percentage of current image it can maximally shift'''
    if len(img.shape) == 3:
        rows,cols, chan = img.shape
    else:
        rows,cols = img.shape
    R = np.random.randint(-range*cols, range*cols)
    M = np.float32([[1,0,R],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows), borderMode = cv2.BORDER_CONSTANT, borderValue = (255,255,255))
    return dst

def Verticalshift(img, range = 0.1):
    '''range is a float determining the percentage of current image it can maximally shift'''
    if len(img.shape) == 3:
        rows,cols, chan = img.shape
    else:
        rows,cols = img.shape
    R = np.random.randint(-range*cols, range*cols)
    M = np.float32([[1,0,0],[0,1,R]])
    dst = cv2.warpAffine(img,M,(cols,rows), borderMode = cv2.BORDER_CONSTANT, borderValue = (255,255,255))
    return dst

img2 = Horizontalshift(img1, 0.05)
img3 = Verticalshift(img2, 0.05)
plt.imshow(img2)
plt.imshow(img3)
plt.show()

##Shear function - A novel shear function = Same principle as rotation - only shear to the right
def shear(img, range = 1, random = True):
    if len(img.shape) == 3:
        rows,cols, chan = img.shape
    else:
        rows,cols = img.shape
    
    ##Choosing starting points
    c1 = cols/2 - 0.2*cols
    c2 = cols/2 + 0.2*cols
    c3 = cols/2
    r1 = 0.2* rows
    r2 = rows - 0.2*rows
    
    ##Starting position
    pts1 = np.float32([[c1,r1],[c2,r1],[c3,r2]])
    if random:
    ##For Rows
        pt1 = r1+np.random.randint(-range, range)
        pt2 = r2+np.random.randint(-range, range)
    ##For column
        pt3 = c1+np.random.randint(-range, range)
        pt4 = c2+np.random.randint(-range, range)
        pt5 = c3+np.random.randint(-range, range)
    else:
        pt1 = r1
        pt2 = r2
        ##For column
        pt3 = c1
        pt4 = c2
        pt5 = c3+range/2

    ## Move to new points
    pts2 = np.float32([[pt3,pt1],[pt4,pt1],[pt5,pt2]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows), borderMode = cv2.BORDER_CONSTANT, borderValue = (255,255,255))
    return dst

img3 = shear(img1, -20 , random = False)
plt.imshow(img3)
plt.show()

##Clipped Zoom
def zoom(img, zoom_factor, range = None, random = True):
    if random:
        zoom_factor = np.random.uniform(zoom_factor-range, zoom_factor + range)
    
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='edge')
    assert result.shape[0] == height and result.shape[1] == width
    return result


##Function For a sequential augment
def Augment(img, dim, rot, vert, hori, shr, zm):
    img = cv2.resize(img, (dim,dim))
    img = rotate(img, rot)
    img = Verticalshift(img, vert)
    img = Horizontalshift(img, hori)
    img = shear(img, shr)
    img = zoom(img, zoom_factor = 1, range = zm)
    return img/255

img1 = Augment(img, 224, 5, 0.01, 0.01 ,5, 0.1)
plt.imshow(img1)
plt.show()

##Function to generate batches
#Ordered generator with shuffle - We create batches from the list of paths to images and then read only a certain batch at once in memory
def Generator(files, labels, batchsize, img_dim):
    files,labels = shuffle(files,labels)
    while 1:
        for i in range(0, len(files), batchsize):
            batchX = files[i:i+batchsize]
            batchY = np.array(labels[i:i+batchsize][:])
            batchX = np.array([cv2.resize(cv2.imread(fname), (img_dim, img_dim))/255 for fname in batchX])
            yield(batchX, batchY)

##Generator 2 with Data Augmentation on the fly
def GeneratorAug(files, labels, batchsize, img_dim):
    files,labels = shuffle(files,labels)
    while 1:
        for i in range(0, len(files), batchsize):
            batchX = files[i:i+batchsize]
            batchY = np.array(labels[i:i+batchsize][:])
            batchX = np.array([Augment(cv2.imread(fname), img_dim, 5, 0.01, 0.01, 5, 0.1) for fname in batchX])
            yield(batchX, batchY)
