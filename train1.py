from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import load_train_data, load_test_data, load_test_mask
from matplotlib import pyplot as plt
import tensorflow as tf
import csv
import pydicom
from skimage.io import imsave, imread
from pydicom.errors import InvalidDicomError
import matplotlib.pyplot as plt
import pickle

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#img_rows = 96
#img_cols = 96
img_rows = 384
img_cols = 384

smooth = 1.

#train_data_path = 'train/'
batch_size=32


def get_train_file_names(input_file_name):
    images = sorted(os.listdir(input_file_name))
    train_id=[]
    train_mask_id=[]
    for image_name in images:
        if 'mask'  in image_name:
            #print(os.path.join(input_file_name, image_name))
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        train_id.append(os.path.join(input_file_name, image_name))
        #print(os.path.join(input_file_name, image_name))
        train_mask_id.append(os.path.join(input_file_name, image_mask_name))            
          
    
    return train_id,train_mask_id

def generator(input_file_name, batch_size):
    images = sorted(os.listdir(input_file_name))
    total = len(images) // 2
    imgs = np.ndarray((batch_size, img_rows, img_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((batch_size, img_rows, img_cols), dtype=np.uint8)
    i=0
    while True:
        for image_name in images:
            #print(i)
            if 'mask'  in image_name:
                continue
            image_mask_name = image_name.split('.')[0] + '_mask.npy'
            #img = imread(os.path.join(train_data_path, image_name), as_grey=True)
            img = np.load(os.path.join(input_file_name, image_name))
            img=np.squeeze(img, axis=0)
            #print(os.path.join(train_data_path, image_name))
            #img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)
            img_mask = np.load(os.path.join(input_file_name, image_mask_name))
            img_mask=np.squeeze(img_mask, axis=0)
            #print(img_mask.shape)
            #print(os.path.join(train_data_path, image_mask_name))
            img = np.array([img])
            img_mask = np.array([img_mask])

            imgs[i] = img
            imgs_mask[i] = img_mask
            i +=1
            if i % batch_size==0:                 
                 #print ('Hello')
                 i=0  
                 #print(i) 
                 yield imgs[..., np.newaxis], imgs_mask[..., np.newaxis]

def generator_validation(input_file_name, batch_size):
    images = sorted(os.listdir(input_file_name))
    total = len(images) // 2
    imgs = np.ndarray((batch_size, img_rows, img_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((batch_size, img_rows, img_cols), dtype=np.uint8)
    i=0
    while True:
        for image_name in images:
            #print(image_name)
            if 'mask'  in image_name:
                continue
            image_mask_name = image_name.split('.')[0] + '_mask.npy'
            #img = imread(os.path.join(train_data_path, image_name), as_grey=True)
            img = np.load(os.path.join(input_file_name, image_name))
            img=np.squeeze(img, axis=0)
            #print(os.path.join(input_file_name, image_name))
            #img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)
            img_mask = np.load(os.path.join(input_file_name, image_mask_name))
            img_mask=np.squeeze(img_mask, axis=0)
            #print(img_mask.shape)
            #print(os.path.join(train_data_path, image_mask_name))
            img = np.array([img])
            img_mask = np.array([img_mask])

            imgs[i] = img
            imgs_mask[i] = img_mask
            i +=1
            if i % batch_size==0:                 
                 #print ('Hello')
                 i=0  
                 #print(i) 
                 yield imgs[..., np.newaxis], imgs_mask[..., np.newaxis]

"""def get_validation_data(input_file_name):
    images = sorted(os.listdir(input_file_name))
    total = len(images) // 2
    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    i=0
    for image_name in images:
         #print(i)
        if 'mask'  in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        #img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img = np.load(os.path.join(input_file_name, image_name))
        img=np.squeeze(img, axis=0)
        print(os.path.join(input_file_name, image_name))
        #img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)
        img_mask = np.load(os.path.join(input_file_name, image_mask_name))
        img_mask=np.squeeze(img_mask, axis=0)
        #print(img_mask.shape)
        #print(os.path.join(train_data_path, image_mask_name))
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask
    return imgs[..., np.newaxis], imgs_mask[..., np.newaxis]   """   

def load_test_data_modifed():
    imgs_test = np.load('imgs_test_true.npy')
    imgs_id = np.load('imgs_id_test_true.npy')
    return imgs_test, imgs_id


def dice_coef(y_true, y_pred): 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice(im1, im2, empty_score=1.0):  #different function copied from net
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print (im1.sum() )
    #print (im2.sum() )
    return 2. * intersection.sum() / im_sum


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True, mode='constant')

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def preprocess1(imgs):
    imgs_p = imgs
    
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

def train():     #modified by Sibaji
    """print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess1(imgs_train)
    imgs_mask_train = preprocess1(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]"""
    #val_data,val_mask_id=get_validation_data('Validation/')
    
    train_id,train_mask_id=get_train_file_names('train/')
    #print(len(train_id)) 
    steps_per_epoch_value=len(train_id)/batch_size
    print (steps_per_epoch_value)
    validation_id,validation_mask_id=get_train_file_names('Validation/')
    #print(len(train_id)) 
    steps_per_epoch_validation_value=len(train_id)/batch_size
    print (steps_per_epoch_validation_value)
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    filepath="weights/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only= True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    """history = model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=100, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])"""
    history =model.fit_generator(generator('train/', batch_size),
                    steps_per_epoch=steps_per_epoch_value, epochs=2,validation_data=generator_validation('Validation/',batch_size),
                         validation_steps=steps_per_epoch_validation_value,
                            callbacks=[model_checkpoint])
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model accuracy')
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig('accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig('loss.png')

def predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test_true, imgs_id_test_true = load_test_data_modifed()
    imgs_test_true = preprocess(imgs_test_true)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet() 


    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-'*30)
    print('evaluting on test data...')
    print('-'*30)
    score = model.evaluate(imgs_test, imgs_test_true, verbose=1)
    print(score)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


    
def predict_modified1(test_npy,test_mask_npy):
    """print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess1(imgs_train)
    imgs_mask_train = preprocess1(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    print("mean and std")
    print(mean)
    print(std)"""
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_test_mask_true = load_test_mask(test_npy,test_mask_npy)
    imgs_test = preprocess1(imgs_test)
    imgs_test1 =imgs_test 
    imgs_test_mask_true = preprocess1(imgs_test_mask_true)

    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    imgs_test_mask_true  = imgs_test_mask_true.astype('float32')
    imgs_test_mask_true /= 255.  # scale masks to [0, 1]

    

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')
   
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test_pred.npy', imgs_mask_test)

    
    # define an empty list
    imgs_id = []

    # open file and read the content in a list
    with open('listfile.txt', 'r') as filehandle:  
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            # add item to the list
            imgs_id.append(currentPlace)

    

    #image_id1=0
    pred_dir = 'preds'
    dice_coeff_each=[]
    dice_coeff_each1=[]
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image1, image, image_true,  image_id in zip(imgs_test1, imgs_mask_test, imgs_test_mask_true, imgs_id):
        dice_coeff_each.append( dice_coef(image_true , image)) #calculate dice coefficients for each image

        image1=(image1[:, :, 0] * 1.).astype(np.uint8)
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        image_true = (image_true[:, :, 0] * 255.).astype(np.uint8)
        dice_coeff_each1.append([image_id, dice(image_true , image)]) #calculate dice coefficients for each image using new function

        #print(image1.shape)
        #print(image.shape)
        imsave(os.path.join(pred_dir, str(image_id) + '_org.png'), image1)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)
        imsave(os.path.join(pred_dir, str(image_id) + '_true.png'), image_true)
        #image_id1=image_id1+1
        #if image_id1 > 10:
            #break
    print('-'*30)
    print('evaluting on test data...')
    print('-'*30)
    score = model.evaluate(imgs_test, imgs_test_mask_true, verbose=1)
    print(score)
    del imgs_test #clear some memory
    del imgs_test1 
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #print(sess.run(dice_coeff_each))
    a=sess.run(dice_coeff_each)
    #print(dice_coeff_each1)
    sess.close()
    print('-'*30)
    print('Mean dice coefficients after averaging using default dice function...')
    print('-'*30)
    print(np.mean(a))

   
    

    with open("prediction1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(dice_coeff_each1)
    with open("prediction.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(np.column_stack((imgs_id,a)))



if __name__ == '__main__':
    #train_and_predict()

    train()
    # predict_modified1('imgs_test1.npy','imgs_mask_test1.npy')
    
