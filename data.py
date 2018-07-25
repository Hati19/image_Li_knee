from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

data_path = 'train/'
test_data_path ='test/'
image_rows = 384
image_cols = 384


def create_train_data():
    #train_data_path = os.path.join(data_path, 'train')
    train_data_path=data_path
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask'  in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
	
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)
	
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
            print(np.amax(img))
            print(np.amax(img_mask))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    #train_data_path = os.path.join(data_path, 'test')
    train_data_path ='test/'
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    img_id =0;
    for image_name in images:
        #img_id = int(image_name.split('.')[0])
        img_id = img_id + 1
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')

def create_test_data_modified():                  
    #train_data_path = os.path.join(data_path, 'test')
    train_data_path ='test_true/'
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    img_id =0;
    for image_name in images:
        #img_id = int(image_name.split('.')[0])
        img_id = img_id + 1
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test_true.npy', imgs)
    np.save('imgs_id_test_true.npy', imgs_id)
    print('Saving to .npy files done.')

def create_test_data_mask():
    #train_data_path = os.path.join(data_path, 'train')
    train_data_path=test_data_path
    images = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask'  in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
	
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)
	
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
            print(np.amax(img))
            print(np.amax(img_mask))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_mask_test.npy', imgs_mask)
    print('Saving to .npy files done.')

def create_test_data_mask1():
    #train_data_path = os.path.join(data_path, 'train')
    train_data_path=test_data_path
    images = os.listdir(train_data_path)
    total = len(images) // 2
    print(len(images))
    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = []
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in sorted(images):
        if 'mask'  in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        imgs_id.append(image_name.split('.')[0])
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
            print(np.amax(img))
            print(np.amax(img_mask))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_mask_test.npy', imgs_mask)
    print('Saving to .npy files done.')
    with open('listfile.txt', 'w') as filehandle:
        for listitem in imgs_id:
            filehandle.write('%s\n' % listitem)

def load_test_mask():
    imgs_test = np.load('imgs_test.npy')
    imgs_mask  = np.load('imgs_mask_test.npy')
    return imgs_test, imgs_mask


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id



if __name__ == '__main__':
    #create_train_data()
    
    create_test_data_mask1()
