#!/usr/bin/env python

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array

# Importing the Keras libraries and packages
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


import cv2

import numpy as np
import glob

def prepare_classifier(output_dim):
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (96, 96, 3), activation = 'relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = output_dim, activation = 'softmax'))
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

def prepare_classifier2(output_dim):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (96, 96, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(units = 64, activation = 'relu'))

    model.add(Dropout(0.8))

    model.add(Dense(units = output_dim, activation = 'softmax'))
    # Compiling the CNN
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


def get_model(labels):
    classifier = prepare_classifier2(len(labels))
    # Initialising the CN    # Part 2 - Fitting the CNN to the images
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            rotation_range=20,
            fill_mode='nearest',
            horizontal_flip = False)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory('labels_train',
            target_size = (96, 96),
            batch_size = 16,
            class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('labels_test',
            target_size = (96, 96),
            batch_size = 16,
            class_mode = 'categorical')
    print training_set.class_indices
    classifier.fit_generator(training_set,
            steps_per_epoch = 2000 // 16,
            epochs = 25,
            validation_data = test_set,
            validation_steps = 800 // 16)
    return classifier

def get_labels_and_images(directory):
    dirs = glob.glob(directory+'/*')
    data = {}
    for d in dirs:
        label = d.split('/')[1]
        images = []
        for imgfile in glob.glob(directory+'/'+label+'/*'):
            img = load_img(imgfile, target_size=(224,224))
            img = img_to_array(img)
            images.append(img)

        data[label] = images
    return data



class label_classifier:
    def __init__(self):
        self.labels = sorted([d.split("/")[1] for d in glob.glob('labels_test/*')])
        self.classifier = init_label_classifier()

    def predict_image(self, img):
        img = cv2.resize(img, dsize=(96,96), interpolation=cv2.INTER_CUBIC)
        img = img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        prediction = self.classifier.predict(img)[0]
        return self.labels[np.argmax(prediction)]

def init_label_classifier():
    model_file = "labels_model.h5"
    if(glob.glob(model_file)):
        classifier = load_model(model_file)
    else:
        data = get_labels_and_images('labels_test')
        classifier = get_model(data.keys())
        classifier._make_predict_function()
        classifier.save(model_file)
    return classifier

if __name__=="__main__":
    data = get_labels_and_images('labels')
    print data

    # load model from file or train new
    classifier = init_label_classifier()

    from keras.preprocessing import image
    for i,d in enumerate(sorted(glob.glob('labels_test/*'))):
        label = d.split('/')[1]
        imgfiles = glob.glob(d+'/*')
        print label 
        count = 0
        for imgfile in imgfiles:
            test_image = image.load_img(imgfile, target_size = (96, 96))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            count += classifier.predict(test_image)[0][i]
        print count, "/", len(imgfiles)
