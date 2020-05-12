import numpy as np
import matplotlib.pyplot as plt
import re

import src
from src import dataModifier as DM
from src import edaPlotDataAdapt as DA

import eda as EDA

#   импорт данных из дата сета *.csv
#dataset = get_DataSet_on_numpy()
adspends_s1 = np.genfromtxt("data/DataSet/CleanData.csv",
                            dtype='int',
                            delimiter=";",
                            skip_header=1,
                            usecols=(range(24, 35)))

fiches = np.genfromtxt("data/DataSet/CleanData.csv",
                       delimiter=";",
                       skip_header=1,
                       usecols=(5, 10, 12, 13, 15, 16, 17))

SIC_codes = np.genfromtxt("data/DataSet/CleanData.csv",
                          dtype='str',
                          delimiter=";",
                          skip_header=1,
                          usecols=(8))

sucsSIC_codes = np.genfromtxt("data/DataSet/sucsSIC_codes.csv",
                              dtype='str',
                              delimiter=";",
                              skip_header=1)

ballSic = DM.covid19SicCode(SIC_codes, sucsSIC_codes)

X1 = np.column_stack((fiches[:, :4], ballSic))
X2 = np.column_stack((adspends_s1[:, :-3], fiches[:, -3:]))
Y = adspends_s1[:, -3:]

X1 = np.asarray(X1).astype('float32')
X2 = np.asarray(X2).astype('float32')
Y = np.asarray(Y).astype('int')

X1, mean1, std1 = DM.normalization(X1)
X2, mean2, std2 = DM.normalization(X2)
#Y = DM.normalization(Y)

np.random.seed(426)

indices = DM.mixedIndex(X1)
X1 = X1[indices]
X2 = X2[indices]
Y = Y[indices]

from keras import models
from keras.models import Model
from keras import layers
from keras import Input
from keras import regularizers
from keras.models import load_model
from keras.optimizers import RMSprop

digit_input = Input(shape=(X1.shape[1], ))
dense_digit_layer_1 = layers.Dense(32, activation='relu')(digit_input)
dense_digit_layer_1 = layers.Dense(8, activation='relu')(dense_digit_layer_1)

text_input = Input(shape=(X2.shape[1], ))
dense_text_layer_2 = layers.Dense(32, activation='relu')(text_input)
dense_text_layer_2 = layers.Dense(8, activation='relu')(dense_text_layer_2)

concatenated = layers.concatenate([dense_digit_layer_1, dense_text_layer_2],
                                  axis=-1)

conc_layrs = layers.Dense(8, activation='relu')(concatenated)
conc_layrs = layers.Dense(8, activation='relu')(conc_layrs)
answer = layers.Dense(Y.shape[1])(conc_layrs)

model = Model([digit_input, text_input], answer)

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit([X1, X2],
                    Y,
                    epochs=500,
                    batch_size=64,
                    validation_split=0.4)
model.evaluate([X1, X2], Y)

#model.save_weights('Dense_model.h5')

#графики изменения качества модели
import matplotlib.pyplot as plt

acc = history.history['mae']
val_acc = history.history['val_mae']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(10, len(acc) + 1)

plt.plot(epochs, acc[9:], 'bo', label='Training acc')
plt.plot(epochs, val_acc[9:], 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss[9:], 'bo', label='Training loss')
plt.plot(epochs, val_loss[9:], 'b', label='Validation val_loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#проверим на данных
adspends_s1 = np.genfromtxt("data/DataSet/TestData.csv",
                            dtype='int',
                            delimiter=";",
                            skip_header=1,
                            usecols=(range(24, 35)))

fiches = np.genfromtxt("data/DataSet/TestData.csv",
                       delimiter=";",
                       skip_header=1,
                       usecols=(5, 10, 12, 13, 15, 16, 17))

SIC_codes = np.genfromtxt("data/DataSet/TestData.csv",
                          dtype='str',
                          delimiter=";",
                          skip_header=1,
                          usecols=(8))

ballSic = DM.covid19SicCode(SIC_codes, sucsSIC_codes)

x_test_1 = np.column_stack((fiches[:, :4], ballSic))
x_test_2 = np.column_stack((adspends_s1[:, :-3], fiches[:, -3:]))
y_test = adspends_s1[:, -3:]

x_test_1 = np.asarray(x_test_1).astype('float32')
x_test_2 = np.asarray(x_test_2).astype('float32')
y_test = np.asarray(y_test).astype('int')

x_test_1 = DM.normalization(x_test_1, mean1, std1)
x_test_2 = DM.normalization(x_test_2, mean2, std2)

y_pred = np.array(model.predict([x_test_1, x_test_2]))
y_pred = np.column_stack((adspends_s1[:, :-3], y_pred))

for i in range(len(y_test)):
    plt.plot(adspends_s1[i, :], 'b', color='r', label='test')
    plt.plot(y_pred[i, :], 'b', label='pred')
    plt.title(i)
    plt.legend()
    plt.show()
    plt.figure()

#plt.show()