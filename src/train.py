from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Activation, Reshape, Lambda
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU
from keras.optimizers import SGD

import src.config as cf
from src.data_generator import TextSequenceGenerator
from src.log import get_logger

logger = get_logger(__name__)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def train(pretrained=False):
    # Input Parameters
    img_h = 32
    img_w = 128
    img_size = (img_w, img_h)
    max_text_len = 32
    batch_size = 16

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    downsample_factor = pool_size ** 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    train_set = TextSequenceGenerator(
        cf.WORDS_TRAIN,
        img_size=img_size, max_text_len=max_text_len,
        downsample_factor=downsample_factor
    )
    test_set = TextSequenceGenerator(
        cf.WORDS_TEST,
        img_size=img_size, max_text_len=max_text_len,
        downsample_factor=downsample_factor,
        shuffle=False, data_aug=False
    )

    no_samples = 115319
    no_train_set = int(no_samples * 0.95)
    no_val_set = no_samples - no_train_set
    logger.info("No train set: %d", no_train_set)
    logger.info("No val set: %d", no_val_set)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2),
                        (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    gru_1 = GRU(rnn_size, return_sequences=True,
                kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True,
                kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(80, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[32], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # loss function
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    # model = load_model('../models/tmp_model.h5', compile=False)
    model = Model(inputs=[input_data, labels,
                          input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    y_func = K.function([input_data], [y_pred])
    ckp = ModelCheckpoint(
        cf.MODEL_CHECKPOINT, monitor='val_loss',
        verbose=1, save_best_only=True, save_weights_only=True
    )
    earlystop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'
    )

    model.fit_generator(generator=train_set,
                        steps_per_epoch=no_train_set // batch_size,
                        epochs=5,
                        validation_data=test_set,
                        validation_steps=no_val_set // batch_size,
                        callbacks=[ckp, earlystop])

    return model, y_func


if __name__ == '__main__':
    model, test_func = train()

    model_json = model.to_json()
    with open(cf.CONFIG_MODEL, 'w') as f:
        f.write(model_json)

    model.save_weights(cf.WEIGHT_MODEL)
