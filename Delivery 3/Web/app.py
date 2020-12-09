
from flask import Flask, render_template, request
import base64
from binascii import a2b_base64
import numpy as np

import cv2

app = Flask(__name__, template_folder='html')
UPLOAD_FOLDER = 'uploads/'
app.secret_key = "$F!@sKpR0jeCt$"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# model imports
import cv2
#import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import random
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as tf_keras_backend

tf_keras_backend.set_image_data_format('channels_last')
tf_keras_backend.image_data_format()

import time
from tensorflow.keras.callbacks import Callback
from datetime import datetime

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras



def get_items(image):
    with open("images/save_image", "w+") as fp:
        fp.write(image)

    print("IMAGE OBTAINED")

@app.route('/', defaults={'path': 'html'})
@app.route('/upload', methods=['GET', 'POST'])
def upload_file(path):
    image = None
    if request.method == 'POST':
        names = request.get_json()
        if "image" in names:
            image = base64.decodebytes(names['image'])
            get_items(image)

    return render_template('find.html')


@app.route('/imageupload', methods=['POST'])
def uplloadimage():
    image = request.data;
    print("upload image request")
    print(image);
    strimage = str(image)
    print(strimage)
    base64image = str(image)[len("b'data:image/octet-stream;base64,"):-1]
    print(base64image)
    binimage = a2b_base64(base64image)
    with open("predict_this.png", "wb") as fp:
        fp.write(binimage);

    img = preprocess_image('predict_this.png', 128, 64)
    plt.imshow(img)
    plt.savefig('processed_image.png')
    my_model=load_model()

    value=predict('predict_this.png', my_model)

    if len(value) >= 2 and value[0] == 'A' or value[0] == 'U' or value[0] == '#':

        if len(value) >= 3 and value[1] == 'A' or value[1] == 'U':
            return (value[2:])
        else:
            return (value[1:])


    return value


def decode_text(prediction):
    classes = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    prediction = prediction[prediction != -1]
    result = ''.join(list(map(lambda x: classes[int(float(x))], prediction)))
    return result

def predict(filename, my_model):
  #filepath='/content/sample_data/IAM/Images/'+filename
  sample_processed_image=[]
  sample_processed_image.append((preprocess_image(filename, 128, 64)).T)
  sample_processed_image=np.array(sample_processed_image)
  sample_processed_image = sample_processed_image.reshape(1, 128, 64, 1)
  prediction_trail = my_model.predict(x=sample_processed_image)
  prediction_decode = tf_keras_backend.get_value(tf_keras_backend.ctc_decode(prediction_trail,
                                                                                 input_length = np.ones(prediction_trail.shape[0])*prediction_trail.shape[1],
                                                                                  greedy=True)[0][0])

  return decode_text(prediction_decode)

def load_model():
    input_data = layers.Input(name='InputFormat', shape=(128,64,1), dtype='float32')  # (None, 128, 64, 1)

    # Convolution layer (VGG)
    model_layers = layers.Conv2D(64, (3, 3), padding='same', name='Conv2DLayer1', kernel_initializer='he_normal')(input_data)
    model_layers = layers.BatchNormalization()(model_layers)
    model_layers = layers.Activation('relu')(model_layers)
    model_layers = layers.MaxPooling2D(pool_size=(2, 2), name='MaxPoolLayer1')(model_layers)  # (None,64, 32, 64)

    model_layers = layers.Conv2D(128, (3, 3), padding='same', name='Conv2DLayer2', kernel_initializer='he_normal')(model_layers)
    model_layers = layers.BatchNormalization()(model_layers)
    model_layers = layers.Activation('relu')(model_layers)
    model_layers = layers.MaxPooling2D(pool_size=(2, 2), name='MaxPoolLayer2')(model_layers)

    model_layers = layers.Conv2D(256, (3, 3), padding='same', name='Conv2DLayer3', kernel_initializer='he_normal')(model_layers)
    model_layers = layers.BatchNormalization()(model_layers)
    model_layers = layers.Activation('relu')(model_layers)
    model_layers = layers.Conv2D(256, (3, 3), padding='same', name='Conv2DLayer4', kernel_initializer='he_normal')(model_layers)
    model_layers = layers.BatchNormalization()(model_layers)
    model_layers = layers.Activation('relu')(model_layers)
    model_layers = layers.MaxPooling2D(pool_size=(1, 2), name='MaxPoolLayer3')(model_layers)  # (None, 32, 8, 256)

    model_layers = layers.Conv2D(512, (3, 3), padding='same', name='Conv2DLayer5', kernel_initializer='he_normal')(model_layers)
    model_layers = layers.BatchNormalization()(model_layers)
    model_layers = layers.Activation('relu')(model_layers)
    model_layers = layers.Conv2D(512, (3, 3), padding='same', name='Conv2DLayer6')(model_layers)
    model_layers = layers.BatchNormalization()(model_layers)
    model_layers = layers.Activation('relu')(model_layers)
    model_layers = layers.MaxPooling2D(pool_size=(1, 2), name='MaxPoolLayer4')(model_layers)

    model_layers = layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='Conv2DLayer7')(model_layers)
    model_layers = layers.BatchNormalization()(model_layers)
    model_layers = layers.Activation('relu')(model_layers)

    # CNN to RNN
    model_layers = layers.Reshape(target_shape=((32, 2048)), name='reshape')(model_layers)
    model_layers = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(model_layers)

    # RNN layer
    model_gru_layer1 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='GRULayer1')(model_layers)
    model_gru_layer1b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='GRULayer1B')(model_layers)
    model_reversed_gru_layer1b = layers.Lambda(lambda inputTensor: tf_keras_backend.reverse(inputTensor, axes=1)) (model_gru_layer1b)

    gru1Merged = layers.add([model_gru_layer1, model_reversed_gru_layer1b])
    gru1Merged = layers.BatchNormalization()(gru1Merged)

    model_gru_layer2 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='GRULayer2')(gru1Merged)
    model_gru_layer2b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='GRULayer2B')(gru1Merged)
    model_reversed_gru_layer2b= layers.Lambda(lambda inputTensor: tf_keras_backend.reverse(inputTensor, axes=1)) (model_gru_layer2b)

    gru2Merged = layers.concatenate([model_gru_layer2, model_reversed_gru_layer2b])
    gru2Merged = layers.BatchNormalization()(gru2Merged)

    # transforms RNN output to character activations:
    model_layers = layers.Dense(80, kernel_initializer='he_normal', name='dense2')(gru2Merged)
    model_outputs = layers.Activation('softmax', name='softmax')(model_layers)

    labels = layers.Input(name='the_labels', shape=[16], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    my_model = Model(inputs=input_data, outputs=model_outputs)
    #my_model.summary()
    #my_model.load_weights(filepath='HandWrittenTextPredictor.h5')

    return my_model


@app.route('/found')
def found():
    items = get_items()
    return render_template('found.html', image_names = items)


def padding(image, old_width, old_height, new_width, new_height):
    h1, h2 = int((new_height - old_height) / 2), int((new_height - old_height) / 2) + old_height
    w1, w2 = int((new_width - old_width) / 2), int((new_width - old_width) / 2) + old_width
    image_pad = np.ones([new_height, new_width, 3]) * 255
    image_pad[h1:h2, w1:w2, :] = image
    return image_pad


def set_size(image, target_w, target_h):
    h, w = image.shape[:2]
    if w < target_w and h < target_h:
        image = padding(image, w, h, target_w, target_h)
    elif w >= target_w and h < target_h:
        new_width = target_w
        new_height = int(h * new_width / w)
        new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image = padding(new_image, new_width, new_height, target_w, target_h)
    elif w < target_w and h >= target_h:
        new_height = target_h
        new_width = int(w * new_height / h)
        new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image = padding(new_image, new_width, new_height, target_w, target_h)
    else:
        """w>=target_w and h>=target_h """
        ratio = max(w / target_w, h / target_h)
        new_width = max(min(target_w, int(w / ratio)), 1)
        new_height = max(min(target_h, int(h / ratio)), 1)
        new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image = padding(new_image, new_width, new_height, target_w, target_h)
    return image


def preprocess_image(path, image_w, image_h):
    """ Pre-processing image for predicting """
    image = cv2.imread(path)
    image = set_size(image, image_w, image_h)

    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32)
    image /= 255
    return image


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
