from __future__ import print_function

import os
import numpy as np
import pydicom
from skimage.io import imsave, imread
from pydicom.errors import InvalidDicomError
import matplotlib.pyplot as plt
import pickle

data_path = 'train/'
test_data_path ='test/'

image_rows = 384
image_cols = 384
#mydir = '9003406-9279291/'
intermidiate='intermidiate/'


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

def create_train_data_modified(list_dir):
    # marge the .npy created by create_train_data_from_dicom for each .zip files and save for final training
    # 
    imgs_train=[]
    imgs_mask_train=[]
    for mydir in list_dir:
        a=np.load(intermidiate+str(mydir[:-1])+'.npy')
        print(a.shape)
        imgs_train.extend( np.load(intermidiate+str(mydir[:-1])+'.npy'))
        imgs_mask_train.extend(np.load(intermidiate+str(mydir[:-1])+'_mask.npy'))
        print('Loading done.')
    imgs_train=np.array(imgs_train)
    imgs_mask_train=np.array(imgs_mask_train)
    print(imgs_train.shape)
    print(imgs_mask_train.shape)
    np.save('imgs_train.npy', imgs_train)
    np.save('imgs_mask_train.npy', imgs_mask_train)
    print('Saving to .npy files done.')

def create_train_data_modified1(list_dir):
    # marge the .npy created by create_train_data_from_dicom for each .zip files and save for final training
    # 
    imgs_train=[]
    imgs_mask_train=[]
    for mydir in list_dir:
        a=np.load(intermidiate+str(mydir[:-1])+'.npy')
        print(a.shape)
        imgs_train.extend( np.load(intermidiate+str(mydir[:-1])+'.npy'))
        imgs_mask_train.extend(np.load(intermidiate+str(mydir[:-1])+'_mask.npy'))
        print('Loading done.')
    imgs_train=np.array(imgs_train)
    imgs_mask_train=np.array(imgs_mask_train)
    print(imgs_train.shape)
    print(imgs_mask_train.shape)
    
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]


def create_train_data_modified1(list_dir,train_dir):
    # create .npy files from intermidiate folders .npy files and save them in train folder mean subtraction is done as preprocessing
    # 
    imgs_train=[]
    imgs_mask_train=[]
    imgs_id = []
    for mydir in list_dir:
        a=np.load(intermidiate+str(mydir[:-1])+'.npy')
        print(a.shape)
        imgs_train.extend( np.load(intermidiate+str(mydir[:-1])+'.npy'))
        imgs_mask_train.extend(np.load(intermidiate+str(mydir[:-1])+'_mask.npy'))
        location=intermidiate+str(mydir[:-1])+'_id.txt'
        with open(location, 'r') as filehandle:  
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]
                image_mask_name = currentPlace.split('/')[1]+'_'+currentPlace.split('/')[2] +'_'+currentPlace.split('/')[7]
                #print(image_mask_name)
                # add item to the list
                imgs_id.append(image_mask_name)
        print('Loading done.')
    imgs_train=np.array(imgs_train)
    imgs_mask_train=np.array(imgs_mask_train)
    print(imgs_train.shape)
    print(imgs_mask_train.shape)
    
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    #print(std)
    imgs_train -= mean
    #imgs_train /= std
    
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    #train_dir='train'
    i=0;
    for  image, image_mask,  image_id in zip(imgs_train, imgs_mask_train, imgs_id):
        np.save(os.path.join(train_dir, str(image_id) + '.npy'), np.array([image]))
        np.save(os.path.join(train_dir, str(image_id) + '_mask.npy'), np.array([image_mask]))
        
        """image= np.array([image])
        image = (image [0,:,:]* 255.).astype(np.uint8)
        print(image.shape)
        image_mask= np.array([image_mask])
        image_mask = (image_mask [0,:,:]* 255.).astype(np.uint8)
        print(image_mask.shape)
        print(image_id)
        imsave(os.path.join(train_dir, str(image_id) + '.png'), image)
        imsave(os.path.join(train_dir, str(image_id) + '_mask.png'), image_mask)
        i+=1
        if i>100:
            break"""

def create_train_data_from_dicom(mydir,data_path_local):
    #read dicom files and save as .npy in 12bit format. Data are saved in intermidiate folder
    # read the mask for corrosponding slice from matlab created folders and store as .npy file
    # saves also the file order into a text file
    #train_data_path = os.path.join(data_path, 'train')
    count=0
    train_data=[]
    train_data_id=[]
    train_data_mask=[]

    for root, dirs, files in sorted(os.walk(mydir , topdown=False)):
        #print(30*"--")
        #print(dirs)
        
            
        for name in sorted(files):
            #print(os.path.join(root, name))
            
            try:
                dataset=pydicom.dcmread(os.path.join(root, name))
                if(count==0):
                    print(os.path.join(root, name))
                    count=1

                # plot the image using matplotlib
                #plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
                #plt.show()
                data=dataset.pixel_array
                train_data.append(data)
                train_data_id.append(os.path.join(root, name))
                image_mask_name = data_path_local+root.split('/')[1]+'_'+root.split('/')[2] +'_'+name+ '_mask.tif'
                img_mask = imread(image_mask_name, as_grey=True)
                #plt.imshow(img_mask, cmap=plt.cm.bone)
                #plt.show()            
                train_data_mask.append(img_mask)
                #print(type(data))
                #count=count+1
            except IOError:
                #print(os.path.join(root, name))
                #print('No such file')
                continue
            except InvalidDicomError:
                #print(os.path.join(root, name))
                #print('Invalid Dicom file')
                continue
            #if 'DICOM.zip'  in name:
                #count=count+1

        #for name in dirs:
            #print(os.path.join(root, name))
        if(count>0):
            count=0
            #break;
    
    
    train_data = np.array(train_data)
    print(train_data.shape)
    train_data.dump(intermidiate+str(mydir[:-1])+'.npy')
    train_data_mask = np.array(train_data_mask)
    print(train_data_mask.shape)
    train_data_mask.dump(intermidiate+str(mydir[:-1])+'_mask.npy')
    
    with open(intermidiate+str(mydir[:-1])+'_id.txt', 'w') as filehandle:  
        filehandle.writelines("%s\n" % place for place in train_data_id)

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

def create_test_data_mask2(list_dir,candidate_number):
     # create test.npy and test_mask.npy files from the intermidiate files
    imgs_train=[]
    imgs_mask_train=[]
    imgs_id = []
    for mydir in list_dir:
        #a=np.load(intermidiate+str(mydir[:-1])+'.npy')
        #print(a.shape)
        imgs_train.extend( np.load(intermidiate+str(mydir[:-1])+'.npy'))
        imgs_mask_train.extend(np.load(intermidiate+str(mydir[:-1])+'_mask.npy'))
        location=intermidiate+str(mydir[:-1])+'_id.txt'
        # open file and read the content in a list
        with open(location, 'r') as filehandle:  
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]
                image_mask_name = currentPlace.split('/')[1]+'_'+currentPlace.split('/')[2] +'_'+currentPlace.split('/')[7]
                #print(image_mask_name)
                # add item to the list
                imgs_id.append(image_mask_name)
        print('Loading done.')
    imgs_train=np.array(imgs_train)
    imgs_mask_train=np.array(imgs_mask_train)
    imgs_train=imgs_train[:candidate_number*160,:,:]
    imgs_mask_train=imgs_mask_train[:candidate_number*160,:,:]
    print(imgs_train.shape)
    print(imgs_mask_train.shape)
    np.save('imgs_test1.npy', imgs_train)
    np.save('imgs_mask_test1.npy', imgs_mask_train)
    print('Saving to .npy files done.')
    count=0;
    with open('listfile.txt', 'w') as filehandle:
        for listitem in imgs_id:
            filehandle.write('%s\n' % listitem)
            count=count+1
            if count >= (160*candidate_number):
                break;
                     

def load_test_mask(test_npy,test_mask_npy):
    imgs_test = np.load(test_npy)
    imgs_mask  = np.load(test_mask_npy)
    return imgs_test, imgs_mask


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id



if __name__ == '__main__':
    #create_train_data()
    
    #create_test_data_mask1()
      

    #call below function to create the training image npy in intermidiate folder
    """create_train_data_from_dicom('9309170-9496443/','train_9309170-9496443/')"""
      
    #create train.npy for final input, Give the folder names to create training images
    """list_dir=['9003406-9279291/', '9500390-9698705/', '9720535-9897397/', '9902757-9993846/']
    create_train_data_modified(list_dir)"""
    #create train.npy for final input, Give the folder names to create training images and save the .npy in train folder
    list_dir=['9003406-9279291/','9500390-9698705/', '9720535-9897397/', '9902757-9993846/']
    create_train_data_modified1(list_dir,'train1')
    # create test.npy for final input, Give the folder names to create test images
    """list_dir=['9309170-9496443/']
    create_test_data_mask2(list_dir,1)"""
