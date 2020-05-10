# импорт пакетов
import pandas as pd
import numpy as np

import re
import random

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

# чтение данных
df = pd.read_csv('data/DataSet/data.csv', delimiter=';', usecols=(9, 10))

for col in df.columns:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        print('created missing indicator for: {}'.format(col))
        df['{}_ismissing'.format(col)] = missing

# отбрасываем строки с большим количеством пропусков
ismissing_cols = [col for col in df.columns if 'ismissing' in col]
df['num_missing'] = df[ismissing_cols].sum(axis=1)

ind_missing = df[df['num_missing'] > 0].index
df_less_missing_rows = df.drop(ind_missing, axis=0)

cols_to_drop = [
    'employees_range_ismissing', 'employees_ismissing', 'num_missing'
]
df_less_hos_beds_raion = df_less_missing_rows.drop(cols_to_drop, axis=1)
data = np.asarray(df_less_hos_beds_raion)

TOKEN_RE = re.compile(r'[\d]+')
x_fiches = []
for row in data:
    rangEemployees = TOKEN_RE.findall(row[0])
    if len(rangEemployees) == 2:
        botRange, topRange = int(rangEemployees[0]), int(rangEemployees[1])
    else:
        botRange, topRange = int(rangEemployees[0]), int(rangEemployees[0]) * 2
    x_fiches.append([botRange, topRange])

y_target = (data[:, 1]).astype('float64')
x_fiches = np.asarray(x_fiches).astype('float64')
"""mean = x_fiches.mean(axis=0)
x_fiches -= mean
std = x_fiches.std(axis=0)
x_fiches /= std

mean = y_target.mean(axis=0)
y_target -= mean
std = y_target.std(axis=0)
y_target /= std"""
y_target = []
for x in x_fiches:
    y_target.append(random.randint(x[0], x[1]))
#ntgthm
"""
from keras import models
from keras import layers
from keras import regularizers
from keras.optimizers import RMSprop

model = models.Sequential()
model.add(layers.Dense(8, activation='relu',
                       input_shape=(x_fiches.shape[1], )))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(lr=1e-5), loss='mse', metrics=['mae'])

history = model.fit(x_fiches,
                    y_target,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.4)

print(model.evaluate(x_fiches, y_target))

target = np.array(model.predict(x_fiches))
"""
print()
