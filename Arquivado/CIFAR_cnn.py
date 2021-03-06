from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

batch_size = 8
num_classes = 5
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
training_set_directory = os.path.join(
    os.getcwd(), 'dataset_turtles\\data_set')

# Convert class vectors to binary class matrices.

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(80, 80, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    training_set = train_datagen.flow_from_directory(
        training_set_directory, (80, 80),
        subset="training",
        class_mode="categorical")

    validation_set = train_datagen.flow_from_directory(
        training_set_directory, (80, 80),
        subset="validation",
        class_mode="categorical")

    model.fit_generator(training_set,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=validation_set,
                        shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
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
        validation_split=0.2)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    training_set = datagen.flow_from_directory(
        training_set_directory, (80, 80),
        batch_size=batch_size,
        subset="training",
        class_mode="categorical")

    validation_set = datagen.flow_from_directory(
        training_set_directory, (80, 80),
        batch_size=batch_size,
        subset="validation",
        class_mode="categorical")

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(training_set,
                        epochs=epochs,
                        validation_data=validation_set,
                        workers=4)

# Save model and weights
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate_generator(validation_set, verbose=1, steps=200)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


predictions = model.predict_generator(validation_set, steps=100)
predicted_classes = np.argmax(predictions, axis=1)

print('Confusion Matrix')
print(confusion_matrix(validation_set.classes, predicted_classes))

true_classes = validation_set.classes
class_labels = list(validation_set.class_indices.keys())
report = classification_report(
    true_classes, predicted_classes, target_names=class_labels)
print(report)
