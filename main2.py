# импорт пакетов
import pandas as pd
import numpy as np
import seaborn as sns
import re
import random

import src
from src import cleanData as CD

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

#----------------------------------------быстро взглянем на сами данные
"""Прежде чем переходить к процессу очистки, всегда нужно представлять исходный датасет. Давайте быстро взглянем на сами данные:
Этот код покажет нам, что набор данных состоит из 30471 строки и 292 столбцов. Мы увидим, являются ли эти столбцы числовыми или категориальными признаками.
Теперь мы можем пробежаться по чек-листу «грязных» типов данных и очистить их один за другим.
"""

plt.style.use('ggplot')
from matplotlib.pyplot import figure

#%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12, 8)

pd.options.mode.chained_assignment = None

# чтение данных
df = pd.read_csv('data/DataSet/data.csv', delimiter=';')

# shape and data types of the data
print(df.shape)
print(df.dtypes)

# отбор числовых колонок
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)

# отбор нечисловых колонок
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)

#------------------------------------------Отсутствующие данные
"""
Тепловая карта пропущенных значений
Когда признаков в наборе не очень много, визуализируйте пропущенные значения с помощью тепловой карты.
Приведенная ниже карта демонстрирует паттерн пропущенных значений для первых 30 признаков набора. По горизонтальной оси расположены признаки, по вертикальной – количество записей/строк. Желтый цвет соответствует пропускам данных.

Заметно, например, что признак life_sq имеет довольно много пустых строк, а признак floor – напротив, всего парочку – около 7000 строки.
"""
cols = df.columns[:21]  # первые 21 колонок
# определяем цвета
# желтый - пропущенные данные, синий - не пропущенные
colours = ['#000099', '#ffff00']
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
plt.show()
print()
"""
Процентный список пропущенных данных
Если в наборе много признаков и визуализация занимает много времени, можно составить список долей отсутствующих записей для каждого признака.
У признака life_sq отсутствует 21% значений, а у floor – только 1%.

Этот список является полезным резюме, которое может отлично дополнить визуализацию тепловой карты.
"""

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing * 100)))

#-------------------------Что делать с пропущенными значениями

#-------1.2.1. Отбрасывание записей
"""
Первая техника в статистике называется методом удаления по списку и заключается в простом отбрасывании записи, содержащей пропущенные значения. Это решение подходит только в том случае, если недостающие данные не являются информативными.

Для отбрасывания можно использовать и другие критерии. Например, из гистограммы, построенной в предыдущем разделе, мы узнали, что лишь небольшое количество строк содержат более 35 пропусков. Мы можем создать новый набор данных df_less_missing_rows, в котором отбросим эти строки.
"""
# сначала создаем индикатор для признаков с пропущенными данными
for col in df.columns:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        print('created missing indicator for: {}'.format(col))
        df['{}_ismissing'.format(col)] = missing

# отбрасываем строки с большим количеством пропусков
ismissing_cols = [col for col in df.columns if 'ismissing' in col]
df['num_missing'] = df[ismissing_cols].sum(axis=1)

ind_missing = df[df['num_missing'] > 10].index
df = df.drop(ind_missing, axis=0)

#----------------Внесение недостающих значений
"""
Для численных признаков можно воспользоваться методом принудительного заполнения пропусков. Например, на место пропуска можно записать среднее или медианное значение, полученное из остальных записей.

Для категориальных признаков можно использовать в качестве заполнителя наиболее часто встречающееся значение.

Возьмем для примера признак life_sq и заменим все недостающие значения медианой этого признака:
"""

#для значений года основания компании можно воспользоваться медианой по остальным значениям в этом столбце
med = df['year_founded'].median()
print(med)
df['year_founded'] = df['year_founded'].fillna(med)

df = CD.rangeAverage(df, 'employees_range', 'employees')
cols_to_drop = ['employees_range']

df = CD.rangeAverage(df, 'revenue_range', 'revenue')
cols_to_drop = ['employees_range', 'revenue_range']

df = df.drop(cols_to_drop, axis=1)

#NaicsToSic = pd.read_csv('data/DataSet/NAICS-to-SIC.csv', delimiter=';')

df = CD.fillSicUseNaics(df, 'data/DataSet/NAICS-to-SIC.csv')

cols = df.columns[:21]  # первые 21 колонок
colours = ['#000099', '#ffff00']
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
plt.show()

df.to_csv('data/DataSet/CleanData.csv', sep=";")

print()