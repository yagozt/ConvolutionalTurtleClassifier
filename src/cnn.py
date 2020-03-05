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
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os


data_augmentation = True
batch_size = 32
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model75.h5'
training_set_directory = os.path.join(
    os.getcwd(), 'dataset_turtles\\training_set')
test_set_directory = os.path.join(os.getcwd(), 'dataset_turtles\\test_set')
data_set_directory = os.path.join(os.getcwd(), 'new_dataset')

# Inicializando as camadas da Rede neural
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(80, 80, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(1))
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(1))
classifier.add(Dense(units=8, activation='softmax'))

#  Método de compilação.
classifier.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
if not data_augmentation:
    print('Sem data augmentation.')
    train_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(training_set_directory,
                                                     target_size=(80, 80),
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_set = test_datagen.flow_from_directory(test_set_directory,
                                                target_size=(80, 80),
                                                batch_size=batch_size,
                                                class_mode='categorical')

    # classifier.fit_generator(training_set,
    #                         steps_per_epoch = 8000,
    #                         epochs = 25,
    #                         validation_data = test_set,
    #                         validation_steps = 2000)

else:
    print('Usando data augmentation.')
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       rotation_range=90,
                                       horizontal_flip=True,
                                       validation_split=0.2)

    training_set = train_datagen.flow_from_directory(data_set_directory,
                                                     target_size=(80, 80),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     subset="training")

    test_set = train_datagen.flow_from_directory(data_set_directory,
                                                 target_size=(80, 80),
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 subset="validation")

classifier.fit_generator(training_set,
                         steps_per_epoch=300,
                         epochs=5,
                         validation_steps=50,
                         validation_data=test_set)


scores = classifier.evaluate_generator(
    generator=test_set, steps=250, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


predictions = classifier.predict_generator(test_set)
predicted_classes = np.argmax(predictions, axis=1)

print('Confusion Matrix')
print(confusion_matrix(test_set.classes, predicted_classes))

true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())
report = classification_report(
    true_classes, predicted_classes, target_names=class_labels)
print(report)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
classifier.save(model_path)
print('Saved trained model at %s ' % model_path)
# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('./dataset_turtles/single_pred/lepidochelysolivacea/lepidochelysolivacea_4_6.jpg', target_size = (80, 80))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'dog'
# else:
#     prediction = 'cat'
