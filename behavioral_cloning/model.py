from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, BatchNormalization, MaxPooling2D, Lambda, Dropout, Flatten, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import csv

# retrieve the samples (e.g. image file names, rotation angle)
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# split the samples to train and validation sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

data_folder_name = './data/IMG/'

# define the generator method which loads images in a batches
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                center_name = data_folder_name + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                
                correction = 0.6 # this is a parameter to tune
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                
                left_name = data_folder_name + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                
                right_name = data_folder_name + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# method which builds the actual model
# in order to improve the pipeline, I would rather do all the data preprocessing separately and then feed the processed data to the network
def build_model():
    model = Sequential()
    # data pre-processing
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x/127.5)-1.))
    
    # convolutions with maxpooling and batchnorm
    model.add(Conv2D(24, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(36, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(48, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    # flatten and add fully connected layers
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(BatchNormalization())
   # model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(1))
    
    # compile with Adam optimizer and mean squared error as the loss function
    model.compile(optimizer='adam', loss='mse')
    
    return model

# trains the model
# defined 2 callbacks: early stopping and checkpoint to save the model if the validation loss has been improved
def train_model(model, train_generator, validation_generator, epochs=3):
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,             nb_val_samples=len(validation_samples), nb_epoch=epochs, callbacks=[early_stopping_callback, checkpoint_callback])


    
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = build_model()

train_model(model, train_generator, validation_generator, 3)

model.save('model.h5')





              