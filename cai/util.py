import numpy as np
import os
import tensorflow
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import color as skimage_color
import csv
import random

def save_2d_array_as_csv(a, filename):
    """This function saves a 2D array into the filename (second parameter)
    # Arguments
        a: 2D array
        filename: string containing the filename of the CSV file to be saved.
    """
    with open(filename, "w+") as local_csv:
        csvWriter = csv.writer(local_csv, delimiter=',')
        csvWriter.writerows(a)

def slice_3d_into_2d(aImage, NumRows, NumCols, ForceCellMax = False):
  """Transforms a 3D array into a 2D array. Channels are placed side by side in a new array.
  # Arguments
        aImage: array
        NumRows: number of rows in the new array.
        NumCols: number of cols in the new array.
        ForceCellMax: all slices are normalized with MAX = 1.
  """
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

def slice_3d_into_2d_cl(aImage, NumRows, NumCols, ForceCellMax = False):
  """Transforms a 3D array into a 2D array.
    This function assumes that Cells are placed the in the 2 Last dimension.
    Channels are placed side by side in a new array.
  # Arguments
        aImage: array
        NumRows: number of rows in the new array.
        NumCols: number of cols in the new array.
        ForceCellMax: all slices are normalized with MAX = 1.
  """
  SizeX = aImage.shape[1]
  SizeY = aImage.shape[2]
  Depth = aImage.shape[0]
  NewSizeX = SizeX * NumCols
  NewSizeY = SizeY * NumRows
  aResult = np.zeros(shape=(NewSizeX, NewSizeY))
  # print(aResult.shape)
  for depthCnt in range(Depth):
    PosX = depthCnt % NumCols
    PosY = int(depthCnt / NumCols)
    # print(PosX,' ',PosY,' ',PosX*SizeX,' ',PosY*SizeY)
    if ForceCellMax:
      Slice = aImage[depthCnt,:,:]
      SliceMax = Slice.max()
      if SliceMax > 0:
        Slice /= SliceMax
      aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY] += Slice
    else:
      aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY] += aImage[depthCnt,:,:]
  return aResult

def slice_4d_into_3d(aImage, NumRows, NumCols, ForceCellMax = False):
  """Transforms a 4D array into a 2D array. Slices are placed side by side in a new array.
  # Arguments
        aImage: array
        NumRows: number of rows in the new array.
        NumCols: number of cols in the new array.
        ForceCellMax: all slices are normalized with MAX = 1.
  """
  SizeX = aImage.shape[0]
  SizeY = aImage.shape[1]
  Depth = aImage.shape[2]
  Neurons = aImage.shape[3]
  NewSizeX = SizeX * NumCols
  NewSizeY = SizeY * NumRows
  aResult = np.zeros(shape=(NewSizeX, NewSizeY, Depth))
  # print(aResult.shape)
  for NeuronsCnt in range(Neurons):
    PosX = NeuronsCnt % NumCols
    PosY = int(NeuronsCnt / NumCols)
    # print(PosX,' ',PosY,' ',PosX*SizeX,' ',PosY*SizeY)
    # print(PosX,' ',PosY,' ',SizeX,' ',SizeY,' ',Depth)
    if ForceCellMax:
      Slice = aImage[:, :, :, NeuronsCnt]
      #print(Slice.shape)
      #print(aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY, :].shape)
      Slice = Slice - Slice.min()
      SliceMax = Slice.max()
      if SliceMax > 0:
        Slice /= SliceMax
      aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY, :] += Slice
    else:
      aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY, :] += aImage[:, :, :, NeuronsCnt]
  return aResult
  
def slice_4d_into_3d_cl(aImage, NumRows, NumCols, ForceCellMax = False):
  """Transforms a 4D array into a 3D array.
    This function assumes that Cells are placed the in the 3 Last dimensions.
    Slices are placed side by side in a new array.
  # Arguments
        aImage: array
        NumRows: number of rows in the new array.
        NumCols: number of cols in the new array.
        ForceCellMax: all slices are normalized with MAX = 1.
  """
  SizeX = aImage.shape[1]
  SizeY = aImage.shape[2]
  Depth = aImage.shape[3]
  Neurons = aImage.shape[0]
  NewSizeX = SizeX * NumCols
  NewSizeY = SizeY * NumRows
  aResult = np.zeros(shape=(NewSizeX, NewSizeY, Depth))
  # print(aResult.shape)
  for NeuronsCnt in range(Neurons):
    PosX = NeuronsCnt % NumCols
    PosY = int(NeuronsCnt / NumCols)
    # print(PosX,' ',PosY,' ',PosX*SizeX,' ',PosY*SizeY)
    # print(PosX,' ',PosY,' ',SizeX,' ',SizeY,' ',Depth)
    if ForceCellMax:
      Slice = aImage[NeuronsCnt, :, :, :]
      #print(Slice.shape)
      #print(aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY, :].shape)
      Slice = Slice - Slice.min()
      SliceMax = Slice.max()
      if SliceMax > 0:
        Slice /= SliceMax
      aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY, :] += Slice
    else:
      aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY, :] += aImage[NeuronsCnt, :, :, :]
  return aResult

def show_neuronal_patterns(aWeights, NumRows, NumCols, ForceCellMax = False):
  """Show first layer patterns. This function does a similar job to slice_4d_into_3d
  # Arguments
        aImage: array
        NumRows: number of rows in the new array.
        NumCols: number of cols in the new array.
        ForceCellMax: all slices are normalized with MAX = 1.
  """
  SizeX = aWeights.shape[0]
  SizeY = aWeights.shape[1]
  Depth = aWeights.shape[2]
  Neurons = aWeights.shape[3]
  NewSizeX = SizeX * NumCols + NumCols - 1
  NewSizeY = SizeY * NumRows + NumRows - 1
  aResult = np.zeros(shape=(NewSizeX, NewSizeY, Depth))
  # print(aResult.shape)
  for NeuronsCnt in range(Neurons):
    PosX = NeuronsCnt % NumCols
    PosY = int(NeuronsCnt / NumCols)
    # print(PosX,' ',PosY,' ',PosX + PosX*SizeX,' ',PosX + (PosX+1)*SizeX)
    # print(PosX,' ',PosY,' ',SizeX,' ',SizeY,' ',Depth)
    if ForceCellMax:
      Slice = aWeights[:, :, :, NeuronsCnt]
      # print(Slice.shape)
      # print(aResult[PosX + PosX*SizeX:PosX + (PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY, :].shape)
      Slice = Slice - Slice.min()
      SliceMax = Slice.max()
      if SliceMax > 0:
        Slice /= SliceMax
      aResult[PosX + PosX*SizeX:PosX + (PosX+1)*SizeX, PosY + PosY*SizeY:PosY + (PosY+1)*SizeY, :] += Slice
    else:
      aResult[PosX + PosX*SizeX:PosX + (PosX+1)*SizeX, PosY + PosY*SizeY:PosY + (PosY+1)*SizeY, :] += aWeights[:, :, :, NeuronsCnt]
  return aResult
  
def show_neuronal_patterns_nf(aWeights, NumRows, NumCols, ForceCellMax = False):
  """Show first layer patterns. Neurons are placed in the first dimention. This function does a similar job to show_neuronal_patterns.
  # Arguments
        aImage: array
        NumRows: number of rows in the new array.
        NumCols: number of cols in the new array.
        ForceCellMax: all slices are normalized with MAX = 1.
  """
  Neurons = aWeights.shape[0]
  SizeX = aWeights.shape[1]
  SizeY = aWeights.shape[2]
  Depth = aWeights.shape[3]
  NewSizeX = SizeX * NumCols + NumCols - 1
  NewSizeY = SizeY * NumRows + NumRows - 1
  aResult = np.zeros(shape=(NewSizeX, NewSizeY, Depth))
  # print(aResult.shape)
  for NeuronsCnt in range(Neurons):
    PosX = NeuronsCnt % NumCols
    PosY = int(NeuronsCnt / NumCols)
    # print(PosX,' ',PosY,' ',PosX + PosX*SizeX,' ',PosX + (PosX+1)*SizeX)
    # print(PosX,' ',PosY,' ',SizeX,' ',SizeY,' ',Depth)
    if ForceCellMax:
      Slice = aWeights[NeuronsCnt, :, :, :]
      # print(Slice.shape)
      # print(aResult[PosX + PosX*SizeX:PosX + (PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY, :].shape)
      Slice = Slice - Slice.min()
      SliceMax = Slice.max()
      if SliceMax > 0:
        Slice /= SliceMax
      aResult[PosX + PosX*SizeX:PosX + (PosX+1)*SizeX, PosY + PosY*SizeY:PosY + (PosY+1)*SizeY, :] += Slice
    else:
      aResult[PosX + PosX*SizeX:PosX + (PosX+1)*SizeX, PosY + PosY*SizeY:PosY + (PosY+1)*SizeY, :] += aWeights[NeuronsCnt, :, :, :]
  return aResult
  
def slice_4d_into_2d(aImage, ForceColMax = False, ForceRowMax = False, ForceCellMax = False):
  """Transforms a 4D array into a 2D array. Channels are placed side by side in a new array.
  # Arguments
        aImage: array
        ForceColMax: scales according to column max.
        ForceRowMax: scales according to row max.
        ForceCellMax: scales according to the cell (slice).
  """
  Images = aImage.shape[0] 
  SizeX = aImage.shape[1]
  SizeY = aImage.shape[2]
  Depth = aImage.shape[3]
  NewSizeX = SizeX * Depth
  NewSizeY = SizeY * Images
  aResult = np.zeros(shape=(NewSizeX, NewSizeY))
  #print('aImage:', aImage.shape)
  for depthCnt in range(Depth):
    if ForceRowMax:
      SliceMin = 0 # aImage[:, :, :, depthCnt].min()
      SliceMax = aImage[:, :, :, depthCnt].max()-SliceMin
    for imgCnt in range(Images):
      PosX = depthCnt
      PosY = imgCnt
      #print(PosX,' ',PosY,' ',PosX*SizeX,' ',PosY*SizeY)
      Slice = np.copy(aImage[imgCnt, :, :, depthCnt])
      if ForceColMax:
          SliceMin = 0 # aImage[:, :, :, depthCnt].min()
          SliceMax = aImage[imgCnt, :, :, :].max()-SliceMin

      if ForceCellMax:
          SliceMin = 0 #Slice.min()
          SliceMax = Slice.max()-SliceMin
            #print(PosX,' ',PosY,' ',PosX*SizeX,' ',PosY*SizeY,' ',SliceMax)

      #print(PosX,' ',PosY,':','Min:', np.amin(Slice), 'Max:', np.amax(Slice), 'Slice Min:',SliceMin,'Slice Max:',SliceMax)
      if ForceRowMax or ForceColMax or ForceCellMax:
        Slice -= SliceMin
        if SliceMax > 0:
            Slice /= SliceMax        
      aResult[PosX*SizeX:(PosX+1)*SizeX, PosY*SizeY:(PosY+1)*SizeY] += Slice
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

def preprocess(img,  bipolar=True, tfcons=False):
    """If Bipolar, transforms an image from [0,255] interval into bipolar [-2,+2] interval.
    """
    if (bipolar):
        # JP prefers bipolar input [-2,+2]
        img /= 64
        img -= 2
    else:
        img /= 255
    if (tfcons): img = tensorflow.constant(np.array(img))

def preprocess_cp(img,  bipolar=True, tfcons=False):
    """Same as preprocess but returning a copy of the value."""
    img_result = np.copy(img)
    preprocess(img_result,  bipolar=bipolar,  tfcons=tfcons)
    return img_result

def deprocess(img,  bipolar=True, tfcast=False):
    """Opposite process from preprocess.
    """
    if (bipolar):
        img += 2
        img *= 64
    else:
        img *= 255
    if (tfcast): img = tensorflow.cast(img, tensorflow.uint8)
    
def deprocess_cp(img,  bipolar=True, tfcast=False):
    """Opposite process from preprocess_cp.
    """
    img_result = np.copy(img)
    deprocess_cp(img_result,  bipolar=bipolar,  tfcast=tfcast)
    return img_result

def rgb2monopolar(img):
    """Transforms the input image into a monopolar (0, +1) image. """
    img /= 255
    return img

def rgb2bipolar(img):
    """Transforms the input image into a bipolar (-2, +2) image. """
    img /= 64
    img -= 2
    return img

def rgb2monopolar_lab(img):
    """Transforms the input image into a monopolar (0, +1) LAB image. """
    img /= 255
    img = skimage_color.rgb2lab(img)
    img[:,:,0:3] /= [100, 200, 200]
    img[:,:,1:3] += 0.5
    return img

def rgb2bipolar_lab(img):
    """Transforms the input image into a bipolar (-2, +2) LAB image. """
    img /= 255
    img = skimage_color.rgb2lab(img)
    img[:,:,0:3] /= [25, 50, 50]
    img[:,:,0] -= 2
    return img

def rgb2black_white_25percent(img):
    """Transforms the input image into a black white image in 25% of the cases. """
    if random.randint(0, 100) < 25:
        bw_test = np.copy(img)
        bw_test[ :, :, 0] += img[ :, :, 1] + img[ :, :, 2]
        bw_test[ :, :, 0] /= 3
        bw_test[ :, :, 1] = bw_test[ :, :, 0]
        bw_test[ :, :, 2] = bw_test[ :, :, 0]
        return bw_test
    else:
        return img

def rgb2black_white_50percent(img):
    """Transforms the input image into a black white image in 50% of the cases. """
    if random.randint(0, 100) < 50:
        bw_test = np.copy(img)
        bw_test[ :, :, 0] += img[ :, :, 1] + img[ :, :, 2]
        bw_test[ :, :, 0] /= 3
        bw_test[ :, :, 1] = bw_test[ :, :, 0]
        bw_test[ :, :, 2] = bw_test[ :, :, 0]
        return bw_test
    else:
        return img

def rgb2black_white_75percent(img):
    """Transforms the input image into a black white image in 75% of the cases. """
    if random.randint(0, 100) < 75:
        bw_test = np.copy(img)
        bw_test[ :, :, 0] += img[ :, :, 1] + img[ :, :, 2]
        bw_test[ :, :, 0] /= 3
        bw_test[ :, :, 1] = bw_test[ :, :, 0]
        bw_test[ :, :, 2] = bw_test[ :, :, 0]
        return bw_test
    else:
        return img

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

def create_image_generator_no_augmentation(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=0,  # epsilon for ZCA whitening
        rescale=None, # set rescaling factor (applied before any other transformation)
        preprocessing_function=None, # set function that will be applied on each input
        data_format=None, # image data format, either "channels_first" or "channels_last"
        validation_split=0.0 # fraction of images reserved for validation (strictly between 0 and 1)
    ):
    """This is a wrapper for keras.preprocessing.image without data augmentation.
    """
    return create_image_generator(
        featurewise_center=featurewise_center,  # set input mean to 0 over the dataset
        samplewise_center=samplewise_center,  # set each sample mean to 0
        featurewise_std_normalization=featurewise_std_normalization,  # divide inputs by std of the dataset
        samplewise_std_normalization=samplewise_std_normalization,  # divide each input by its std
        zca_whitening=zca_whitening,  # apply ZCA whitening
        zca_epsilon=zca_epsilon,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.0,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.0,
        shear_range=0.0,  # set range for random shear
        zoom_range=0.0,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=rescale,
        # set function that will be applied on each input
        preprocessing_function=preprocessing_function,
        # image data format, either "channels_first" or "channels_last"
        data_format=data_format,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=validation_split
    )

def relu(adata):   
    """Calculates the rectifier linear unit.
    # Arguments
        adata: a floating point array.
    # Returns
        a floating point array.
    """
    return np.maximum(0, adata)

def reverse_sort(arraydata):  
  """Does a revert sort.
  # Arguments
        arraydata: input array
  """
  return np.array(list(reversed(np.sort(arraydata))))

def get_class_position(pclass, predictions):
  """Identifies the position of the class in the predictions array. When 
  get_class_position returns 0, it means that the prediction is correct.
  # Arguments
        pclass: is an integer number identifying the class.
        predictions: array with predictions per class.
  """
  predicted_probability = predictions[pclass]
  predictions_sorted = reverse_sort(predictions)
  return np.where(predictions_sorted == predicted_probability)[0][0]

def get_max_acceptable_common_divisor(a, b, max_acceptable=1000000):
  """
  This is an inefficient max acceptable common divisor implementation to be improved.
    # Arguments
        a: is an integer.
        b: is an integer.
        max_acceptable: maximum acceptable common divisor.
  """
  divisor = max(1, min(a, b, max_acceptable))
  while divisor > 0:
      if a % divisor == 0 and b % divisor == 0:
          return divisor
          break
      divisor -= 1
