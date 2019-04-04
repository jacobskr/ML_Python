# Convolutional Neural Network

# Don't need to do any preprocessing because of how the data is set up in folders.

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
clf = Sequential()

    # Step 1 - Convolution
clf.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    # Step 2 - Pooling
clf.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Step 3 - Flattening
clf.add(Flatten())

    # Step 4 - Full Connection
        #Our rule for output_dim from ANN, is annoying to calculate for this, so we chose 128 for now.
clf.add(Dense(units=128, activation='relu'))
clf.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
    # Image augmentation to prevent overfitting - compensates for only 8,000 samples
    # Takes random augmentations of images and trains on those as well (rotate, shear, flip, etc)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(directory='Data\\training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(directory='Data\\test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

# steps_per_epoch = number of train samples / batch_size
# validation_steps = number of validation samples / batch_size
clf.fit_generator(training_set,
                    steps_per_epoch=250,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=63)

# =============================================================================
#  Make these results better by making the CNN deeper
# =============================================================================
 
# Initializing the CNN
clf = Sequential()

    # Step 1 - Convolution
clf.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    # Step 2 - Pooling
clf.add(MaxPooling2D(pool_size=(2, 2), strides=2))

#### ADDED STEP ####
#A Adding a second convolutional layer
    # Don't need inpuit shape parameter because it already has layer going into it
clf.add(Convolution2D(32, (3, 3), activation='relu'))
clf.add(MaxPooling2D(pool_size=(2, 2), strides=2))


    # Step 3 - Flattening
clf.add(Flatten())

    # Step 4 - Full Connection
        #Our rule for output_dim from ANN, is annoying to calculate for this, so we chose 128 for now.
clf.add(Dense(units=128, activation='relu'))
clf.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(directory='Data\\training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(directory='Data\\test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

# steps per epoch = samples_per_epoch set size/ batch_size
clf.fit_generator(training_set,
                    steps_per_epoch=250,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=63)


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Data/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = clf.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'