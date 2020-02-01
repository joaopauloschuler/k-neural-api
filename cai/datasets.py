"""K-CAI data set functions.
"""
from skimage import color as skimage_color
import numpy as np
import cv2
import keras
from keras.datasets import cifar10,  fashion_mnist,  mnist
from keras.preprocessing.image import img_to_array
import cai.util
import os
import urllib.request
import scipy.io
import zipfile
import requests
from sklearn.model_selection import train_test_split

class img_folder_dataset:
    def __init__(self, pbasefolder,  test_size=0.06,  Verbose=False, sizex=256,  sizey=256,  max_samples_per_class=0):
        self.basefolder = pbasefolder
        self.verbose = Verbose
        self.test_size = test_size
        self.sizex = sizex
        self.sizey = sizey
        self.max_samples_per_class = max_samples_per_class
    def load_data(self):
        image_list, label_list = [], []
        folder_list = os.listdir(f"{self.basefolder}/")
        folder_list.sort()
        label_id = 0
        #folder_list.remove('Background_without_leaves')
        for folder in folder_list:
            cnt_per_class = 0
            img_folder_name = f"{self.basefolder}/{folder}/"
            if self.verbose: print('Loading '+img_folder_name)
            img_list = os.listdir(img_folder_name)
            for img_file in img_list:
                absolute_file_name = img_folder_name+'/'+img_file;
                #print('File:'+absolute_file_name)
                if absolute_file_name.lower().endswith('.jpg'):                
                    aimage =  img_to_array(cv2.resize(cv2.imread(absolute_file_name),  tuple((self.sizex, self.sizey))), dtype='int8')
                    #print(aimage.shape)
                    image_list.append(aimage)
                    label_list.append(label_id)
                    cnt_per_class = cnt_per_class + 1
                    if (self.max_samples_per_class>0 and cnt_per_class >= self.max_samples_per_class): break
            label_id = label_id + 1
        #print(image_list)
        image_list = np.array(image_list, dtype='int8')
        label_list = np.array(label_list,  dtype='int16')
        x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=self.test_size, random_state = 17)
        return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

def print_cifar10_result(result):
    """Prints CIFAR-10 result into a readable format.
    # Arguments
        result: 10 elements float array
    """
    # cifar10 labels
    labels = np.array([
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'])
    for n in range(9):
        print("[{}] : {}%".format(labels[n], round(result[0][n]*100,2)))

def load_dataset(dataset, lab=False,  verbose=False,  bipolar=True):
    """Loads a dataset into memory.
    # Arguments
        dataset: object capable of loading the dataset.
        lab: boolean indicating CIELAB (True) color space of RGB color space (False).
        verbose: boolean value.
        bipolar: if true, inputs are given in the rage [-2, +2].
    # Returns
        x_train: array with training images.
        y_train: array with training labels.
        x_test: array with testing images.
        y_test: array with testing labels.
    """
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    if (verbose):
        print("train shape", x_train.shape)
        print("test shape", x_test.shape)
    class_cnt = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, class_cnt)
    y_test = keras.utils.to_categorical(y_test, class_cnt)
    x_train = x_train.astype('float16')
    x_test = x_test.astype('float16')
    if (lab):
        if (verbose):
            print("Converting RGB to LAB.")
        x_train /= 255
        x_test /= 255
        x_train = skimage_color.rgb2lab(x_train)
        x_test = skimage_color.rgb2lab(x_test)
        if (bipolar):
            # JP prefers bipolar input [-2,+2]
            x_train[:,:,:,0:3] /= [25, 50, 50]
            x_train[:,:,:,0] -= 2
            x_test[:,:,:,0:3] /= [25, 50, 50]
            x_test[:,:,:,0] -= 2
        else:
            x_train[:,:,:,0:3] /= [100, 200, 200]
            x_train[:,:,:,1:3] += 0.5
            x_test[:,:,:,0:3] /= [100, 200, 200]
            x_test[:,:,:,1:3] += 0.5            
    else:
        if (verbose):
            print("Loading RGB.")
        if (bipolar):
            x_train /= 64
            x_test /= 64
            x_train -= 2
            x_test -= 2
        else:
            x_train /= 255
            x_test /= 255
    if dataset is fashion_mnist or dataset is mnist:
        img_rows, img_cols = 28, 28    
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    if (verbose):
        for channel in range(0, x_train.shape[3]):
            sub_matrix = x_train[:,:,:,channel]
            print('Channel ', channel, ' min:', np.min(sub_matrix), ' max:', np.max(sub_matrix))
    return x_train, y_train, x_test, y_test

def load_cifar10_dataset(lab=False,  verbose=False,  bipolar=True):
    """Loads a CIFAR-10 into memory.
    # Arguments
        lab: boolean indicating CIELAB (True) color space of RGB color space (False).
        verbose: boolean value.
        bipolar: if true, inputs are given in the rage [-2, +2].
    # Returns
        x_train: array with training images.
        y_train: array with training labels.
        x_test: array with testing images.
        y_test: array with testing labels.
    """
    return load_dataset(cifar10, lab=False,  verbose=False,  bipolar=True)
    
def download_file(remote_url,  local_file):
    r = requests.get(remote_url, stream = True) 
    with open(local_file,"wb") as local_wb: 
        for chunk in r.iter_content(chunk_size=1024*1024): 
             # writing one chunk at a time to local file 
             if chunk: 
                 local_wb.write(chunk)

def unzip_file(local_zip_file, expected_folder_name):
    with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
        zip_ref.extractall(expected_folder_name)    
    
def download_zip_and_extract(url_zip_file, local_zip_file, expected_folder_name, Verbose=True):
    """Downloads (if required) and extracts a zip file. 
    # Arguments
        url_zip_file: remote zip file.
        local_zip_file: local zip file.
        expected_folder_name: where to deploy extracted files.
        Verbose: boolean value.
    """
    if not os.path.isfile(local_zip_file):
        if Verbose: print('Downloading: ', url_zip_file, ' to ', local_zip_file)
        download_file(url_zip_file,  local_zip_file)
    if not os.path.isdir(expected_folder_name):
        os.mkdir(expected_folder_name)
        if Verbose: print('Decompressing into: ', expected_folder_name)
        unzip_file(local_zip_file, expected_folder_name)

def load_hyperspectral_matlab_image(ImgUrl, ClassUrl, ImgProperty, ClassProperty, LocalBaseName, Verbose=True):
  """Downloads (if required) and loads hyperspectral image from matlab file.
  http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
  This function has been tested with AVIRIS sensor data stored as matlab file.
  
  #Examples
  ## AVIRIS sensor over Salinas Valley, California
  ImgUrl='http://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat', 
  ClassUrl='http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat', 
  ImgProperty='salinas', 
  ClassProperty='salinas_gt', 
  LocalBaseName='Salinas', 

  ## A small subscene of AVIRIS Salinas image, denoted Salinas-A
  ImgUrl='http://www.ehu.eus/ccwintco/uploads/d/df/SalinasA.mat', 
  ClassUrl='http://www.ehu.eus/ccwintco/uploads/a/aa/SalinasA_gt.mat', 
  ImgProperty='salinasA', 
  ClassProperty='salinasA_gt', 
  LocalBaseName='SalinasA', 

  ## AVIRIS sensor over the Indian Pines test site in North-western Indiana
  ImgUrl='http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat', 
  ClassUrl='http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat', 
  ImgProperty='indian_pines', 
  ClassProperty='indian_pines_gt', 
  LocalBaseName='indian_pines',

  ## ROSIS sensor during a flight campaign over Pavia, nothern Italy
  ImgUrl='http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat', 
  ClassUrl='http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat', 
  ImgProperty='pavia', 
  ClassProperty='pavia_gt', 
  LocalBaseName='pavia', 
  
  ##  sensor during a flight over Pavia University
  ImgUrl='http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat', 
  ClassUrl='http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat', 
  ImgProperty='paviaU', 
  ClassProperty='paviaU_gt', 
  LocalBaseName='paviaU',

  ## NASA AVIRIS over the Kennedy Space Center (KSC) 
  ImgUrl='http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat', 
  ClassUrl='http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat', 
  ImgProperty='KSC', 
  ClassProperty='KSC_gt', 
  LocalBaseName='KSC',
  
  ## NASA EO-1 satellite over the Okavango Delta, Botswana in 2001-2004
  ImgUrl='http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat', 
  ClassUrl='http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat', 
  ImgProperty='Botswana', 
  ClassProperty='Botswana_gt', 
  LocalBaseName='Botswana',

  # Arguments
    ImgUrl: url from where to download the image file.
    ClassUrl: url from where to download the ground truth.
    ImgProperty: property in the loaded file to load the image.
    ClassProperty: property in the ground truth to load classes.
    LocalBaseName: file name without extension for local storage.
    Verbose: 
  # Returns
    Image: loaded image.
    Classes: loaded ground truth.
    NumClasses: number of classes in the ground truth.
  """
  imgfile = LocalBaseName + '.mat'
  classfile = LocalBaseName + '-class.mat'
  if not os.path.isfile(imgfile):
    if Verbose: print('Downloading:', ImgUrl, ' to ', imgfile)
    urllib.request.urlretrieve(ImgUrl, imgfile)

  if not os.path.isfile(classfile):
    if Verbose: print('Downloading:', ClassUrl, ' to ', classfile)
    urllib.request.urlretrieve(ClassUrl, classfile)
    
  mat = scipy.io.loadmat(imgfile)
  Image = mat[ImgProperty]
  if Verbose: print('matlab file image shape:', Image.shape)
  
  mat = scipy.io.loadmat(classfile)
  Classes = mat[ClassProperty]
  if Verbose: print('matlab file classes shape:', Classes.shape)
  
  Image = Image / Image.max()
  NumClasses = (Classes.max() - Classes.min() + 1)
  
  return Image, Classes, NumClasses
  
def slice_image(Image, PixelClasses, NewImageSize=5):
  """Creates an array of small images from a bigger image.
  # Arguments
    Image: array with input image.
    PixelClasses: ground truth for each pixe.
    NewImageSize: new image size as NewImageSize x NewImageSize pixels.
  # Returns
    aResultImg: array of images.
    aResultClasses: array of classes.
  """
  MaxX = Image.shape[0] - NewImageSize
  MaxY = Image.shape[1] - NewImageSize
  Center = int(NewImageSize / 2)
  aResultImg = [ Image[X:X+NewImageSize, Y:Y+NewImageSize, :]                
    for X in range(0, MaxX)
      for Y in range(0, MaxY) ]
  
  aResultClasses = [ int(PixelClasses[X+Center, Y+Center])                
    for X in range(0, MaxX)
      for Y in range(0, MaxY) ]
  
  NumClasses = (PixelClasses.max() - PixelClasses.min() + 1)
  aResultClasses = keras.utils.to_categorical(aResultClasses, NumClasses)
  
  return np.array(aResultImg), aResultClasses

def create_pixel_array_from_3D_image(Image):
  """Creates a pixel array from a 3D image.
  # Arguments
    Image: array with input image.
  # Returns
    aResultImage: array of pixels.
  """
  SizeX = Image.shape[0]
  SizeY = Image.shape[1]
  Channels = Image.shape[2]
  aResultImage = Image.reshape(SizeX * SizeY, Channels)
  return aResultImage

def create_image_generator_sliced_image():
    """ image generator for sliced images (pixel classification) """
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.0,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.0,
        shear_range=0.,  # set range for random shear
        zoom_range=[1.0,1.0],  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    return datagen

def train_model_on_dataset(model, dataset,  base_model_name, plrscheduler,  batch_size = 64, 
    epochs = 300, momentum=0.9, nesterov=True, verbose=False,  lab=False,  bipolar=True,  
    datagen = cai.util.create_image_generator()):
    """Trains a given neural network model on a given dataset.
    # Arguments
        model: neural network model.
        dataset: object capable of loading the dataset. 
        base_model_name: string with file name without extension.
        plrscheduler: learning rate scheduler. 
        batch_size: integer number.
        epochs: integer number.
        momentum: float. 
        nesterov: bolean. 
        verbose: boolean value.
        lab: boolean indicating CIELAB (True) color space of RGB color space (False).
        bipolar: if true, inputs are given in the rage [-2, +2].
        datagen: a data generator
    # Returns
        fit_result: object
        model_name: h5 file name with best model.
        csv_name: string with CSV file name showing training progress.
    """
    x_train, y_train, x_test, y_test = load_dataset(dataset, verbose=verbose, lab=lab,  bipolar=bipolar)

    batches_per_epoch = int(x_train.shape[0]/batch_size)
    batches_per_validation = int(x_test.shape[0]/batch_size)    
    model_name = base_model_name+'.h5'
    csv_name = base_model_name+'.csv'
    opt = keras.optimizers.SGD(lr=0.1, momentum=momentum, nesterov=nesterov)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])
    
    fit_verbose=0
    if (verbose):
        fit_verbose=2
    
    fit_result = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=batches_per_epoch,
        validation_steps=batches_per_validation,
        validation_data=(x_test, y_test),
        verbose=fit_verbose,
        callbacks=[
            keras.callbacks.LearningRateScheduler(plrscheduler),
            keras.callbacks.ModelCheckpoint(
                filepath=model_name, 
                monitor='val_acc', 
                verbose=fit_verbose, 
                save_best_only=True, 
                save_weights_only=False, 
                mode='max', 
                period=1),
            keras.callbacks.CSVLogger(csv_name, append=False, separator=';')  
        ]
    )
    return fit_result,  model_name,  csv_name    
    
def train_model_on_cifar10(model,  base_model_name, plrscheduler,  batch_size = 64, epochs = 300, momentum=0.9, nesterov=True, verbose=False,  lab=False,  bipolar=True):
    """Trains a given neural network model on a given dataset.
    # Arguments
        model: neural network model.
        base_model_name: string with file name without extension.
        plrscheduler: learning rate scheduler. 
        batch_size: integer number.
        epochs: integer number.
        momentum: float. 
        nesterov: bolean. 
        verbose: boolean value.
        lab: boolean indicating CIELAB (True) color space of RGB color space (False).
        bipolar: if true, inputs are given in the rage [-2, +2].
    # Returns
        fit_result: object
        model_name: h5 file name with best model.
        csv_name: string with CSV file name showing training progress.
    """
    return train_model_on_dataset(model=model, dataset=cifar10,  base_model_name=base_model_name, 
    plrscheduler=plrscheduler,  batch_size=batch_size, epochs=epochs, momentum=momentum, nesterov=nesterov, 
    verbose=verbose, lab=lab, bipolar=bipolar)
