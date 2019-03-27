# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os


data_augmentation = False
batch_size = 32
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'
training_set_directory = os.path.join(os.getcwd(), 'dataset_turtles\\training_set')
test_set_directory = os.path.join(os.getcwd(), 'dataset_turtles\\test_set')

# Inicializando as camadas da Rede neural
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (80, 80, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 5, activation = 'softmax'))

#  Método de compilação.
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

# Part 2 - Fitting the CNN to the images
if not data_augmentation:
    print('Sem data augmentation.')
    train_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(training_set_directory,
                                                 target_size = (80, 80),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

    test_datagen = ImageDataGenerator(rescale = 1./255)

    test_set = test_datagen.flow_from_directory(test_set_directory,
                                            target_size = (80, 80),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')

    # classifier.fit_generator(training_set,
    #                         steps_per_epoch = 8000,
    #                         epochs = 25,
    #                         validation_data = test_set,
    #                         validation_steps = 2000)

else:
    print('Usando data augmentation.')
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    rotation_range=90,
                                    horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('./dataset_turtles/training_set',
                                                    target_size = (80, 80),
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory('./dataset_turtles/test_set',
                                                target_size = (80, 80),
                                                batch_size = batch_size,
                                                class_mode = 'categorical')

classifier.fit_generator(training_set,
                        steps_per_epoch = 8000/batch_size,
                        epochs = 90,
                        validation_data = test_set,
                        validation_steps = 2000/batch_size,
                        workers=12)


scores = classifier.evaluate_generator(generator=test_set,steps=500,verbose=1)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
classifier.save(model_path)
print('Saved trained model at %s ' % model_path)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('./dataset_turtles/single_pred/eretmochelysimbricata/main_eretmochelysimbricata_2.jpg', target_size = (80, 80))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'dog'
# else:
#     prediction = 'cat'