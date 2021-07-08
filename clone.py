import numpy as np
import csv
import cv2
import os

from keras.models import Sequential 
from keras.layers import Cropping2D, Dropout
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# Import the Data
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

# Preprocess the Data
images = []
steering_measurements = []
throttle_measurements = []
brake_measurements = []

for line in lines:
    path = line[0]
    file_name = path.split('/')[-1]
    current_path = './data/IMG/' + file_name
    img = cv2.imread(current_path)
    images.append(img)
    steering_measurement = float(line[3])
    steering_measurements.append(steering_measurement)
    throttle_measurement = float(line[4])
    throttle_measurements.append(throttle_measurement)
    brake_measurement = float(line[5])
    brake_measurements.append(brake_measurement)

augmented_images = []
augmented_steering = []

for image in images:
    augmented_images.append(np.fliplr(image))

for steering in steering_measurements:
    augmented_steering.append(-steering)
   
images.extend(augmented_images)
steering_measurements.extend(augmented_steering)

X_train = np.array(images)
y_train = np.array(steering_measurements)
    
# Create the Neural Network
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: ((x/255.0)-0.5)))
model.add(Convolution2D(6,(5,5), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,(5,5), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(4,(3,3), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(3,(2,2), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(3,(2,2), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1200))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=4)

model.save('model.h5')