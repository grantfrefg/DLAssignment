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

##Function For a sequential augment
def Augment(img, dim, rot, vert, hori, shr):
    img = cv2.resize(img, (dim,dim))
    img = rotate(img, rot)
    img = Verticalshift(img, vert)
    img = Horizontalshift(img, hori)
    img = shear(img, shr)
    return img/255

img1 = Augment(img, 224, 5, 0.01, 0.01 ,10)
plt.imshow(img1)


##Generator 2 with Data Augmentation on the fly
def GeneratorAug(files, labels, batchsize, img_dim):
    files,labels = shuffle(files,labels)
    while 1:
        for i in range(0, len(files), batchsize):
            batchX = files[i:i+batchsize]
            batchY = np.array(labels[i:i+batchsize][:])
            batchX = np.array([Augment(cv2.imread(fname), img_dim, 10, 0.01, 0.01, 5) for fname in batchX])
            yield(batchX, batchY)

##---------------------------------------------------------------------------##
##CNN
##---------------------------------------------------------------------------##
##CNN - can use any model we want from https://keras.io/applications/
        
dim = 224

##Resnet model
basemodel = ResNet50(input_shape=(dim,dim,3), weights = 'imagenet', include_top=False)
base = basemodel.output 
base = GlobalAveragePooling2D()(base)

dense = Dense(1024, activation = 'relu')(base)
drop = Dropout(0.5)(dense) 
##Softmax layer
predictions = Dense(numlabels, activation='softmax')(drop)
model = Model(basemodel.input, predictions)
        
##USe this to freeze blocks of layers and train only last few
#for layer in model.layers[:126]:
#    layer.trainable = False
#for layer in model.layers[126:]:
#    layer.trainable = True

##Set optimizer - Can be anything in keras.optimisers
#Nadam = optimizers.Nadam(lr=0.003)   
SGD = optimizers.SGD(lr = 0.03, momentum = 0.9, nesterov = True)
model.compile(optimizer=SGD,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

##Save path
final_weights_path=path + "model_weightsResnet50-{epoch:02d}-{val_acc:.2f}.hdf5"
callbacks_list = [
    ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00003),
    EarlyStopping(monitor='val_loss', patience=30, verbose=0)
    ]    

## Fit model - Make sure that steps_per_epoch is equal to training size / batchsize - similarly with validation_Steps
classifier = model.fit_generator(GeneratorAug(X_train, y_train,16, dim),steps_per_epoch=2368, epochs=100, 
                         validation_data = Generator(X_val, y_val, 16, dim), validation_steps = 392, callbacks = callbacks_list)

##---------------------------------------------------------------------------##
##TEST TIME AUGMENTATION
##---------------------------------------------------------------------------##
##Test Time Augmentation pipeline - we used fixed augments - 
##5 test time augment (original, shearedright, rotatedright, shearedleft, rotatedleft)

final_weights_path = os.path.join(os.path.abspath(path), 'model_weightsResnet50-32-0.92.hdf5')
model.load_weights(final_weights_path)

TTA1Pred = []
TTA2Pred = []
TTA3Pred = []
TTA4Pred = []
TTA5Pred = []
for counter, i in enumerate(X_val):
    img1 = cv2.resize(cv2.imread(i), (dim,dim))
    img2 = rotate(img1, 5, random = False).reshape(1,224,224,3)
    img3 = shear(img1, 10, random = False).reshape(1,224,224,3)
    img4 = rotate(img1, -5, random = False).reshape(1,224,224,3)
    img5 = shear(img1, -10, random = False).reshape(1,224,224,3)
    pred1 = model.predict((img1/255).reshape(1,224,224,3))
    pred2 = model.predict(img2/255)
    pred3 = model.predict(img3/255)
    pred4 = model.predict(img4/255)
    pred5 = model.predict(img5/255)
    TTA1Pred.append(pred1)
    TTA2Pred.append(pred2)
    TTA3Pred.append(pred3)
    TTA4Pred.append(pred4)
    TTA5Pred.append(pred5)
    if counter % 1000 == 0:
        print(counter)

##Average over the 5 TTAs
ResNetPred = np.mean(np.hstack([TTA1Pred, TTA2Pred, TTA3Pred, TTA4Pred, TTA5Pred]), axis = 1)

##Get class label
ResNetPredC = np.argmax(ResNetPred, axis = 1)
TTA1PredC = np.argmax(TTA1Pred, axis = 2)
TTA2PredC = np.argmax(TTA2Pred, axis = 2)
TTA3PredC = np.argmax(TTA3Pred, axis = 2)
TTA4PredC = np.argmax(TTA4Pred, axis = 2)
TTA5PredC = np.argmax(TTA5Pred, axis = 2)

##Get accuracy
from sklearn.metrics import accuracy_score 
print(accuracy_score(valy, ResNetPredC))
print(accuracy_score(valy, TTA1PredC))
print(accuracy_score(valy, TTA2PredC))
print(accuracy_score(valy, TTA3PredC))
print(accuracy_score(valy, TTA4PredC))
print(accuracy_score(valy, TTA5PredC))

np.save(path + 'ResNetPred', ResNetPred)

np.save(path + 'TTA1PredRN', TTA1Pred)

np.save(path + 'TTA2PredRN', TTA2Pred)

np.save(path + 'TTA3PredRN', TTA3Pred)

np.save(path + 'TTA4PredRN', TTA4Pred)

np.save(path + 'TTA5PredRN', TTA5Pred)
