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
                            usecols=(range(29, 35)))

fiches = np.genfromtxt("data/DataSet/CleanData.csv",
                       delimiter=";",
                       skip_header=1,
                       usecols=(5, 10, 12, 13, 15, 16, 17, 18, 19, 20))

codes = np.genfromtxt("data/DataSet/CleanData.csv",
                      dtype='str',
                      delimiter=";",
                      skip_header=1,
                      usecols=(8))

ballSic = np.zeros((codes.shape[0], 1))

sucsSIC_codes = np.genfromtxt("data/DataSet/sucsSIC_codes.csv",
                              dtype='int',
                              delimiter=";",
                              skip_header=1)

TOKEN_RE = re.compile(r'[\d]+')

for row in range(codes.shape[0]):
    sicCodes = TOKEN_RE.findall(codes[row])
    for code in sicCodes:
        if int(code) in sucsSIC_codes:
            ballSic[row] += 1

for row in fiches:
    for sampl in row:
        if sampl is None:
            print()
X = np.column_stack((fiches, ballSic, adspends_s1[:, :-1]))

Y = adspends_s1[:, -1:]

X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('float32')

X = DM.normalization(X)
#Y = DM.normalization(Y)

np.random.seed(2)

from keras import models
from keras import layers
#from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X.shape[1], )))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.05))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(X, Y, epochs=1000, batch_size=32, validation_split=0.4)

print(model.evaluate(X, Y))

y_pred = np.array(model.predict(X))

#model.save_weights('Dense_model.h5')

#графики изменения качества модели

import matplotlib.pyplot as plt

acc = history.history['mae']
val_acc = history.history['val_mae']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.show()