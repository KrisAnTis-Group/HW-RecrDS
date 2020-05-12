#%%
# импорт пакетов
import pandas as pd
import numpy as np
import seaborn as sns
import re
import random

import src
from src import cleanData as CD
from src import dataModifier as DM
from src import edaPlotDataAdapt as DA

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

import eda as EDA

#%%
#1 Быстро взглянем на данные
"""Прежде чем переходить к процессу очистки, посмотрим, что представляют из себя данные сейчас. 
Этот код покажет нам, что набор данных состоит из 30471 строки и 292 столбцов.
 Мы увидим, являются ли эти столбцы числовыми или категориальными признаками.
"""

CD.initMatplotlib()

# чтение данных
df = pd.read_csv('data/DataSet/data.csv', delimiter=';')

# shape and data types of the data
print(df.shape)
print(df.dtypes)
# отбор числовых колонок
CD.printNumericCols(df)

# отбор нечисловых колонок
CD.printNonNumericCols(df)

#%%
#2 Отсутствующие данные

# 2.1. Тепловая карта пропущенных значений
"""
В нашем наборе не очень много признаков, поэтому, визуализируем пропущенные значения с помощью тепловой карты.
"""

CD.HeatmapCreate(df)
plt.show()

#%%
# 2.2. Процентный список пропущенных данных
"""
Чтобы более формально оценить проблему отсутствующих данных составим список долей отсутствующих записей для каждого признака.
Этот список является полезным дополнением тепловой карты.
"""
CD.printPercentLostData(df)

#%%
#3 Восстановление пропущенных значений
#3.1. Отбрасывание записей
"""
Сперва отсеем те строки, которые содержать слишком много пропусков в данных. 
Нет смысла бороться за такие строки, если попытаться их восстановить они создадут либо дубликаты, либо шум.

Для отбрасывания можно ориентироваться на вычесленный ранее процент пропусков по столбцам, 
лишь небольшое количество строк содержат критическое значение пропусков. 
"""
df = CD.dropRow(df, 10)

#%%
#3.2. Внесение недостающих значений
#3.2.1. Замена медианой
"""
Воспользуемся заполнения пропусков медианами признака, для некоторых численных признаков, таких как: год основания кампании и показы кампании.
"""
df = CD.replacNaMedian(df, [
    'year_founded', 'V_Oct19', 'V_Nov19', 'V_Dec19', 'V_Jan20', 'V_Feb20',
    'V_Mar20'
])

df = CD.rangeAverage(df, 'employees_range', 'employees')
df = CD.rangeAverage(df, 'revenue_range', 'revenue')

#%%
#3.2.2. Замена с использованием сторонних данных
df = CD.fillSicUseNaics(df, 'data/DataSet/NAICS-to-SIC.csv')

#%%
#Тепловая карта

CD.HeatmapCreate(df, 20)
plt.show()

#%%
#4 Выбросы
"""
Немаловажным фактором являются выбросы в данных, такие ситуации обязательно нужно отслеживать
"""
column = [
    'S2_Apr19', 'S2_May19', 'S2_Jun19', 'S2_Jul19', 'S2_Aug19', 'S2_Sep19',
    'S2_Oct19', 'S2_Nov19', 'S2_Dec19', 'S2_Jan20', 'S2_Feb20'
]

df.boxplot(column)
plt.show()

#%%
#4.1 Сглаживание выбросов

for col in column:
    kvantl = df[col].describe()
    kvantl = df[col].describe()['75%']
    for row_index, row in df[df[col] > df[col].describe()['75%']].iterrows():
        d10 = df.loc[row_index, col] * 0.06
        df.loc[row_index, col] = int(int(kvantl) + d10)

df.boxplot(column)
plt.show()

#%%
#5. подготовка тестовых данных для проверки работы сети
testData, df = CD.getTestingData(df, 6)

#%%
#6. Сохраним очищенные данные
df.to_csv('data/DataSet/CleanData.csv', index=False, sep=";")
testData.to_csv('data/DataSet/TestData.csv', index=False, sep=";")

#%%
#7. Распределение бюджетов кампаний по naics и sic отраслям
#   импорт данных из очищенного датасета
adspends_s1 = np.genfromtxt("data/DataSet/CleanData.csv",
                            dtype='float64',
                            delimiter=";",
                            skip_header=1,
                            usecols=(range(24, 35)))
naics = np.genfromtxt("data/DataSet/CleanData.csv",
                      dtype='float64',
                      delimiter=";",
                      skip_header=1,
                      usecols=(5))
sic = np.genfromtxt("data/DataSet/CleanData.csv",
                    dtype='float64',
                    delimiter=";",
                    skip_header=1,
                    usecols=(7))

naics = np.asarray(naics).astype('int')
sic = np.asarray(sic).astype('int')
# вычислим суммарные траты кампании
Summs = np.asarray(DA.summByColumn(adspends_s1)).astype('int')
# сгруппируем траты по категориям naics и sic кампаний
X1 = DA.groupByColumn(np.column_stack((naics, Summs)))
X2 = DA.groupByColumn(np.column_stack((sic, Summs)))

#%%
#7.1 Распределение бюджетов кампаний по naics
w1 = []
for q in X1:
    w1.append(np.asarray(X1[q]).astype('int'))
w1 = np.asarray(w1)
EDA.groupedBar(w1)
plt.show()

#%%
#7.2 Распределение бюджетов кампаний по sic
w2 = []
for q in X2:
    w2.append(np.asarray(X2[q]).astype('int'))
w2 = np.asarray(w2)
EDA.groupedBar(w2)
plt.show()

#%%
#8 Модель регрессии на Dense-слоях
#8.1 Импорт данных
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

#%%
#8.2 Обработка и нормализация данных

#доп. стоблец баллы для компаний с разрешёнными в sic-кодами
ballSic = DM.covid19SicCode(SIC_codes, sucsSIC_codes)

X = np.column_stack((fiches, ballSic, adspends_s1[:, :-3]))
Y = adspends_s1[:, -3:]

X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('int')

X, mean, std = DM.normalization(X)

# перемешивание данных
indices = DM.mixedIndex(X)
X = X[indices]
Y = Y[indices]

#%%
#8.3 Инициализация модели

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

#%%
#8.4 Обучение модели
history = model.fit(X, Y, epochs=65, batch_size=32, validation_split=0.4)

#8.5 Итоговая точность
print(model.evaluate(X, Y))
#%%
#8.6 графики изменения качества модели

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

#%%
#8.7 Проверка модели на тестовых данных
#8.7.1 Импорт тестовых данных
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

#%%
#8.7.2 Обработка и нормализация данных
ballSic = DM.covid19SicCode(SIC_codes, sucsSIC_codes)

x_test = np.column_stack((fiches, ballSic, adspends_s1[:, :-3]))
y_test = adspends_s1[:, -3:]

x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype('int')

x_test = DM.normalization(x_test, mean, std)

#%%
#8.7.3 Predict
y_pred = np.array(model.predict(x_test))
y_pred = np.column_stack((adspends_s1[:, :-3], y_pred))

#%%
#8.7.4 Визуализация данных
for i in range(len(y_test)):
    plt.plot(adspends_s1[i, :], 'b', color='r', label='test')
    plt.plot(y_pred[i, :], 'b', label='pred')
    plt.title(i)
    plt.legend()
    plt.show()
    plt.figure()