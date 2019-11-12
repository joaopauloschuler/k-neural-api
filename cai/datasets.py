import skimage
import numpy as np
import keras
from keras.datasets import cifar10,  fashion_mnist,  mnist
import cai.util

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
    
def print_cifar10_result(result):
    for n in range(9):
        print("[{}] : {}%".format(labels[n], round(result[0][n]*100,2)))

def load_dataset(dataset, lab=False,  verbose=False,  bipolar=True):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    if (verbose):
        print("train shape", x_train.shape)
        print("test shape", x_test.shape)
    class_cnt = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, class_cnt)
    y_test = keras.utils.to_categorical(y_test, class_cnt)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if (lab):
        if (verbose):
            print("Converting RGB to LAB.")
        x_train /= 255
        x_test /= 255
        x_train = skimage.color.rgb2lab(x_train)
        x_test = skimage.color.rgb2lab(x_test)
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
    if (verbose):
        for channel in (0,1,2):
            sub_matrix = x_train[:,:,:,channel]
            print('Channel ', channel, ' min:', np.min(sub_matrix), ' max:', np.max(sub_matrix))
    if dataset is fashion_mnist or dataset is mnist:
        img_rows, img_cols = 28, 28    
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    return x_train, y_train, x_test, y_test

def load_cifar10_dataset(lab=False,  verbose=False,  bipolar=True):
    return load_dataset(cifar10, lab=False,  verbose=False,  bipolar=True)

def train_model_on_dataset(model, dataset,  base_model_name, plrscheduler,  batch_size = 64, epochs = 300, momentum=0.9, nesterov=True, verbose=False,  lab=False,  bipolar=True):
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

    datagen = cai.util.create_image_generator()
    
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
    return train_model_on_dataset(model=model, dataset=cifar10,  base_model_name=base_model_name, 
    plrscheduler=plrscheduler,  batch_size=batch_size, epochs=epochs, momentum=momentum, nesterov=nesterov, 
    verbose=verbose, lab=lab, bipolar=bipolar)
