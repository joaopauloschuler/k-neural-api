import numpy as np
import os
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv

def save_2d_array_as_csv(a, filename):
    """This function saves a 2D array into the filename (second parameter)
    # Arguments
        a: 2D array
        filename: string containing the filename of the CSV file to be saved.
    """
    with open(filename, "w+") as local_csv:
        csvWriter = csv.writer(local_csv, delimiter=',')
        csvWriter.writerows(a)

    """Transforms a 3D array into a 2D array. Channels are placed side by side in a new array.
    # Arguments
        aImage: array
        NumRows: number of rows in the new array.
        NumCols: number of cols in the new array.
        ForceCellMax: all slices are normalized with MAX = 1.
    """
def slice_3d_into_2d(aImage, NumRows, NumCols, ForceCellMax = False):
  SizeX = aImage.shape[0]
  SizeY = aImage.shape[1]
  Depth = aImage.shape[2]
  NewSizeX = SizeX * NumCols
  NewSizeY = SizeY * NumRows
  aResult = np.zeros(shape=(NewSizeX, NewSizeY))
  # print(aResult.shape)
  for depthCnt in range(Depth):
    PosX = depthCnt % NumCols
    PosY = int(depthCnt / NumCols)
    # print(PosX,' ',PosY,' ',PosX*SizeX,' ',PosY*SizeY)
    if ForceCellMax:
      Slice = aImage[:,:,depthCnt]
      SliceMax = Slice.max()
      if SliceMax > 0:
        Slice /= SliceMax
      aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY] += Slice
    else:
      aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY] += aImage[:,:,depthCnt]
  return aResult

def evaluate_model_print(model, x_test, y_test):
    """Evaluates a model and prints its result.
    """
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss", scores[0])
    print("Test accuracy", scores[1])
    return scores
    
def create_folder_if_required(save_dir):
    """Creates a a folder if it doesn't exist.
    # Arguments
        save_dir: string with folder to be created.
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

def get_model_parameter_counts(model):
    """Calculates the number of parameters from a given model.
    # Arguments
        model: model to have parameters counted.
    # Returns
      trainable_count: integer number with trainable parameter count.
      non_trainable_count:  integer number with non trainable parameter count.
    """
    trainable_count = int(np.sum([backend.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([backend.count_params(p) for p in set(model.non_trainable_weights)]))
    return trainable_count, non_trainable_count 

def preprocess(img):
    """Transforms an image from [0,255] interval into bipolar [-2,+2] interval.
    """
    # JP prefers bipolar input [-2,+2]
    img /= 64
    img -= 2

# This is the default CAI Image generator with data augmentation
def create_image_generator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.2,  # set range for random shear
        zoom_range=0.3,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0
    ):
    """This is a wrapper for keras.preprocessing.image with extremely well tested K-CAI default values.
    """
    # This will do preprocessing and realtime data augmentation:
    return ImageDataGenerator(
        featurewise_center=featurewise_center,  # set input mean to 0 over the dataset
        samplewise_center=samplewise_center,  # set each sample mean to 0
        featurewise_std_normalization=featurewise_std_normalization,  # divide inputs by std of the dataset
        samplewise_std_normalization=samplewise_std_normalization,  # divide each input by its std
        zca_whitening=zca_whitening,  # apply ZCA whitening
        zca_epsilon=zca_epsilon,  # epsilon for ZCA whitening
        rotation_range=rotation_range,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=width_shift_range,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=height_shift_range,
        shear_range=shear_range,  # set range for random shear
        zoom_range=zoom_range,  # set range for random zoom
        channel_shift_range=channel_shift_range,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode=fill_mode,
        cval=cval,  # value used for fill_mode = "constant"
        horizontal_flip=horizontal_flip,  # randomly flip images
        vertical_flip=vertical_flip,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=rescale,
        # set function that will be applied on each input
        preprocessing_function=preprocessing_function,
        # image data format, either "channels_first" or "channels_last"
        data_format=data_format,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=validation_split)
