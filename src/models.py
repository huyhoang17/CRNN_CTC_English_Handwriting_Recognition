from keras import backend as K
from keras.layers import Input, Dense, Activation, Reshape, Lambda
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU

import src.config as cf
from src.utils import ctc_lambda_func


def CRNN_model():
    act = 'relu'
    input_data = Input(name='the_input', shape=cf.INPUT_SHAPE, dtype='float32')
    inner = Conv2D(cf.CONV_FILTERS, cf.KERNEL_SIZE, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(cf.POOL_SIZE, cf.POOL_SIZE), name='max1')(
        inner)
    inner = Conv2D(cf.CONV_FILTERS, cf.KERNEL_SIZE, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(cf.POOL_SIZE, cf.POOL_SIZE), name='max2')(
        inner)

    conv_to_rnn_dims = (cf.IMG_W // (cf.POOL_SIZE ** 2),
                        (cf.IMG_H // (cf.POOL_SIZE ** 2)) * cf.CONV_FILTERS)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(cf.TIME_DENSE_SIZE, activation=act, name='dense1')(inner)

    gru_1 = GRU(cf.RNN_SIZE, return_sequences=True,
                kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(cf.RNN_SIZE, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(cf.RNN_SIZE, return_sequences=True,
                kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(cf.RNN_SIZE, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(cf.NO_LABELS, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[cf.MAX_LEN_TEXT], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # loss function
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )

    model = Model(inputs=[input_data, labels,
                          input_length, label_length], outputs=loss_out)

    y_func = K.function([input_data], [y_pred])

    return model, y_func
