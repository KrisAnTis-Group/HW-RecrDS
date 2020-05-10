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

med = df['employees'].median()
TOKEN_RE = re.compile(r'[\d]+')

for row_index, row in df.iterrows():
    if not (row['employees_ismissing']):
        continue
    elif row['employees_range_ismissing']:
        df.loc[row_index, 'employees'] = med
    elif not row['employees_range_ismissing']:
        rangEemployees = TOKEN_RE.findall(row['employees_range'])
        if len(rangEemployees) == 2:
            df.loc[row_index,
                   'employees'] = random.randint(int(rangEemployees[0]),
                                                 int(rangEemployees[1]))
        else:
            df.loc[row_index,
                   'employees'] = random.randint(int(rangEemployees[0]),
                                                 int(rangEemployees[0]) * 2)

cols_to_drop = [
    'employees_range_ismissing', 'employees_ismissing', 'num_missing'
]
df = df.drop(cols_to_drop, axis=1)
# отбрасываем строки с большим количеством пропусков
for col in df.columns:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        print('created missing indicator for: {}'.format(col))
        df['{}_ismissing'.format(col)] = missing

# отбрасываем строки с большим количеством пропусков
ismissing_cols = [col for col in df.columns if 'ismissing' in col]
df['num_missing'] = df[ismissing_cols].sum(axis=1)
#если пропуски есть и в диапазоне и в точной оценке, то они обрабатываются отельно присвоением моды по точной оценке
ind_missing = df[df['num_missing'] > 0].index
df_less_missing_rows = df.drop(ind_missing, axis=0)

cols_to_drop = [
    'employees_range_ismissing', 'employees_ismissing', 'num_missing'
]
df_less_hos_beds_raion = df_less_missing_rows.drop(cols_to_drop, axis=1)
data = np.asarray(df_less_hos_beds_raion)

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