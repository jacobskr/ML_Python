# Convolutional Neural Network

# Don't need to do any preprocessing because of how the data is set up in folders.

# Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

    # Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    # Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Step 3 - Flattening
classifier.add(Flatten())

    # Step 4 - Full Connection
        #Our rule for output_dim from ANN, is annoying to calculate for this, so we chose 128 for now.
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
classifier.fit_generator(training_set,
                         steps_per_epoch=250,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=63)
