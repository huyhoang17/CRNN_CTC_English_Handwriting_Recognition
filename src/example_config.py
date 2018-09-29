from keras import backend as K


# params
MAX_LEN_TEXT = 32
IMAGE_SIZE = (128, 32)
IMG_W, IMG_H = IMAGE_SIZE
NO_CHANNELS = 1

if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (NO_CHANNELS, IMG_W, IMG_H)
else:
    INPUT_SHAPE = (IMG_W, IMG_H, NO_CHANNELS)

BATCH_SIZE = 16
CONV_FILTERS = 16
KERNEL_SIZE = (3, 3)
POOL_SIZE = 2
DOWNSAMPLE_FACTOR = POOL_SIZE ** 2
TIME_DENSE_SIZE = 32
RNN_SIZE = 512

# paths
WORDS_DATA = 'path-to-words.txt'
WORDS_TRAIN = 'path-to-words_train.txt'
WORDS_TEXT = 'path-to-words_test.txt'
CONFIG_MODEL = 'path-to-model-config'
WEIGHT_MODEL = 'path-to-save-model'
MODEL_CHECKPOINT = 'path-to-save-checkpoint'
LOGGING = 'path-to-log-file'

# naming
WORDS_FOLDER = "path-to-words-folder"

"""
data
├── words
│   ├── a01
│   ├── a02
│   ├── a03
│   ├── a04
│   ├── a05
...
"""
