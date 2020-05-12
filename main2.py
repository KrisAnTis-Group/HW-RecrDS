# импорт пакетов
import pandas as pd
import numpy as np
import seaborn as sns
import re
import random

import src
from src import cleanData as CD

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

#----------------------------------------быстро взглянем на сами данные
"""Прежде чем переходить к процессу очистки, всегда нужно представлять исходный датасет. Давайте быстро взглянем на сами данные:
Этот код покажет нам, что набор данных состоит из 30471 строки и 292 столбцов. Мы увидим, являются ли эти столбцы числовыми или категориальными признаками.
Теперь мы можем пробежаться по"company_id" чек-листу «грязных» типов данных и очистить их один за другим.
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

#------------------------------------------Отсутствующие данные
"""
Тепловая карта пропущенных значений
Когда признаков в наборе не очень много, визуализируйте пропущенные значения с помощью тепловой карты.
Приведенная ниже карта демонстрирует паттерн пропущенных значений для первых 30 признаков набора. По горизонтальной оси расположены признаки, по вертикальной – количество записей/строк. Желтый цвет соответствует пропускам данных.

Заметно, например, что признак life_sq имеет довольно много пустых строк, а признак floor – напротив, всего парочку – около 7000 строки.
"""

#CD.HeatmapCreate(df)
#plt.show()
"""
Процентный список пропущенных данных
Если в наборе много признаков и визуализация занимает много времени, можно составить список долей отсутствующих записей для каждого признака.
У признака life_sq отсутствует 21% значений, а у floor – только 1%.

Этот список является полезным резюме, которое может отлично дополнить визуализацию тепловой карты.
"""

CD.printPercentLostData(df)

#-------------------------Что делать с пропущенными значениями

#-------1.2.1. Отбрасывание записей
"""
Первая техника в статистике называется методом удаления по списку и заключается в простом отбрасывании записи, содержащей пропущенные значения. 
Это решение подходит только в том случае, если недостающие данные не являются информативными.

Для отбрасывания можно использовать и другие критерии. Например, из гистограммы, построенной в предыдущем разделе, мы узнали, 
что лишь небольшое количество строк содержат более 35 пропусков. 
Мы можем создать новый набор данных df_less_missing_rows, в котором отбросим эти строки.
"""
# сначала создаем индикатор для признаков с пропущенными данными
df = CD.dropRow(df, 10)

#----------------Внесение недостающих значений
"""
Для численных признаков можно воспользоваться методом принудительного заполнения пропусков. Например, на место пропуска можно записать среднее или медианное значение, полученное из остальных записей.

Для категориальных признаков можно использовать в качестве заполнителя наиболее часто встречающееся значение.

Возьмем для примера признак life_sq и заменим все недостающие значения медианой этого признака:
"""

#для значений года основания компании можно воспользоваться медианой по остальным значениям в этом столбце

df = CD.replacNaMedian(df, [
    'year_founded', 'V_Oct19', 'V_Nov19', 'V_Dec19', 'V_Jan20', 'V_Feb20',
    'V_Mar20'
])

df = CD.rangeAverage(df, 'employees_range', 'employees')
df = CD.rangeAverage(df, 'revenue_range', 'revenue')

df = CD.fillSicUseNaics(df, 'data/DataSet/NAICS-to-SIC.csv')

CD.HeatmapCreate(df, 20)
plt.show()
df.to_csv('data/DataSet/CleanData.csv', index=False, sep=";")

column = [
    'S2_Apr19', 'S2_May19', 'S2_Jun19', 'S2_Jul19', 'S2_Aug19', 'S2_Sep19',
    'S2_Oct19', 'S2_Nov19', 'S2_Dec19', 'S2_Jan20', 'S2_Feb20'
]
for col in column:
    kvantl = df[col].describe()['75%']
    for row_index, row in df[df[col] > df[col].describe()['75%']].iterrows():
        df.loc[row_index, col] = int(kvantl)

df.to_csv('data/DataSet/CleanData.csv', index=False, sep=";")

df['year_founded'].hist(bins=100)
plt.show()
df.boxplot(column=[
    'S2_Apr19', 'S2_May19', 'S2_Jun19', 'S2_Jul19', 'S2_Aug19', 'S2_Sep19',
    'S2_Oct19', 'S2_Nov19', 'S2_Dec19', 'S2_Jan20', 'S2_Feb20'
])
plt.show()
num_rows = len(df.index)
low_information_cols = []  #

for col in df.columns:
    cnts = df[col].value_counts(dropna=False)
    top_pct = (cnts / num_rows).iloc[0]

    if top_pct > 0.9:
        low_information_cols.append(col)
        print('{0}: {1:.5f}%'.format(col, top_pct * 100))
        print(cnts)
        print()
#df.to_csv('data/DataSet/CleanData.csv', index=False, sep=";")

print()