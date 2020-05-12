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
                       usecols=(5, 10, 12, 13, 15, 16, 17, 18, 19, 20))

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

X = np.column_stack((fiches, ballSic, adspends_s1[:, :-3]))
Y = adspends_s1[:, -3:]

X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('int')

X, mean, std = DM.normalization(X)
#Y = DM.normalization(Y)

np.random.seed(247)
"""indices = DM.mixedIndex(X)
X = X[indices]
Y = Y[indices]"""

from keras import models
from keras import layers
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X.shape[1], )))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.05))
model.add(layers.Dense(Y.shape[1]))

model.compile(optimizer='adadelta', loss='mse', metrics=['mae'])
history = model.fit(X, Y, epochs=65, batch_size=32, validation_split=0.4)
model.evaluate(X, Y)

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
                       usecols=(5, 10, 12, 13, 15, 16, 17, 18, 19, 20))

SIC_codes = np.genfromtxt("data/DataSet/TestData.csv",
                          dtype='str',
                          delimiter=";",
                          skip_header=1,
                          usecols=(8))

ballSic = DM.covid19SicCode(SIC_codes, sucsSIC_codes)

x_test = np.column_stack((fiches, ballSic, adspends_s1[:, :-3]))
y_test = adspends_s1[:, -3:]

x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype('int')

x_test = DM.normalization(x_test, mean, std)
y_pred = np.array(model.predict(x_test))
y_pred = np.column_stack((adspends_s1[:, :-3], y_pred))

for i in range(len(y_test)):
    plt.plot(adspends_s1[i, :], 'b', color='r', label='test')
    plt.plot(y_pred[i, :], 'b', label='pred')
    plt.title(i)
    plt.legend()
    plt.show()
    plt.figure()

#plt.show()