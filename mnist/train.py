import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

import log

IMAGE_SIZE = 28
NUM_DIGITS = 10
EPOCHS = 5
BATCH_SIZE = 100
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MNIST_MODEL = os.path.join(MODELS_DIR, 'mnist.h5')


def train(model_path=None):
    np.random.seed(1)
    (x_train, _y_train), (x_test, _y_test) = mnist.load_data()

    # Channels (depth) last - Grayscale image so depth == 1
    x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
    x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)

    # Structure data
    x_train = x_train.astype('float32') / 255  # Normalise as floats between 0 and 1
    x_test = x_test.astype('float32') / 255
    y_train = np_utils.to_categorical(_y_train, NUM_DIGITS)
    y_test = np_utils.to_categorical(_y_test, NUM_DIGITS)

    if not model_path:
        # Structure the convolutional neural network
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=input_shape, data_format='channels_last'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_DIGITS, activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
        )
    else:
        model = load_model(model_path)  # Load saved model from file

    # Train
    model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    model.save(MNIST_MODEL)

    # Show success against test set
    score = model.evaluate(x_test, y_test)
    log.info(f'Test set score: {score[1] % 100}%')


def main():
    train(MNIST_MODEL)


if __name__ == '__main__':
    main()

