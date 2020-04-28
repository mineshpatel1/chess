import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers

import log
from connect_4.game import Connect4

REGULARISATION_CONSTANT = 0.0001
GRID_SHAPE = (6, 7)
INPUT_SHAPE = (2, 6, 7)
NUM_ACTIONS = 42
LEARNING_RATE = 0.1
MOMENTUM = 0.9
HIDDEN_CNN_LAYERS = [{'filters': 75, 'kernel_size': (4, 4)}] * 6


def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    p_i = y_true

    zero = tf.zeros(shape=tf.shape(p_i), dtype=tf.float32)
    where = tf.equal(p_i, zero)

    negatives = tf.fill(tf.shape(p_i), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=p_i, logits=p)
    return loss


def conv_layer(input_block, filters, kernel_size, ):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        data_format="channels_first",
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(REGULARISATION_CONSTANT)
    )(input_block)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    return x


def residual_layer(input_block, filters, kernel_size):
    x = conv_layer(input_block, filters, kernel_size)

    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        data_format="channels_first",
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(REGULARISATION_CONSTANT)
    )(x)

    x = BatchNormalization(axis=1)(x)
    x = add([input_block, x])
    x = LeakyReLU()(x)
    return x


def value_head(input_block):
    """Fork of the neural network that evaluates the board position."""
    x = Conv2D(
        filters=1,
        kernel_size=(1,1),
        data_format="channels_first",
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(REGULARISATION_CONSTANT)
    )(input_block)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)

    x = Dense(
        20,
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(REGULARISATION_CONSTANT)
    )(x)

    x = LeakyReLU()(x)

    x = Dense(
        1,
        use_bias=False,
        activation='tanh',
        kernel_regularizer=regularizers.l2(REGULARISATION_CONSTANT),
        name='value_head'
    )(x)
    return x


def policy_head(input_block):
    """
    Fork of the neural network that chooses which move to make.
    """

    x = Conv2D(
        filters=2,
        kernel_size=(1, 1),
        data_format="channels_first",
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(REGULARISATION_CONSTANT)
    )(input_block)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)

    x = Dense(
        NUM_ACTIONS,
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(REGULARISATION_CONSTANT),
        name='policy_head'
    )(x)

    return x


def build_model():
    """
    AlphaZero style convolutional model for board evaluation and move selection.

    Resources:
        * https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188
        * https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0
        * https://www.nature.com/articles/nature24270
    """
    main_input = Input(shape=INPUT_SHAPE, name='main_input')
    x = conv_layer(main_input, HIDDEN_CNN_LAYERS[0]['filters'], HIDDEN_CNN_LAYERS[0]['kernel_size'])
    for layer in HIDDEN_CNN_LAYERS[1:]:
        x = residual_layer(x, layer['filters'], layer['kernel_size'])

    vh = value_head(x)
    ph = policy_head(x)

    model = Model(inputs=[main_input], outputs=[vh, ph])
    model.compile(
        loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
        optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM),
        loss_weights={'value_head': 0.5, 'policy_head': 0.5},
    )
    return model


class NeuralNet:
    def __init__(self):
        self.model = build_model()

    def predict(self, board: Connect4):
        return self.model.predict(np.array([board.model_input]))
