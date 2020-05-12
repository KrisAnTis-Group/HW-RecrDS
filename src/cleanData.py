# импорт пакетов
import pandas as pd
import numpy as np
import seaborn as sns

import re
import random

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib


def rangeAverage(df, rangeColName='', targetColName=''):

    med = df[targetColName].median()
    TOKEN_RE = re.compile(r'[\d]+')

    for row_index, row in df.iterrows():
        if not (row[targetColName + '_ismissing']):
            continue
        elif row[rangeColName + '_ismissing']:
            df.loc[row_index, targetColName] = med
        elif not row[rangeColName + '_ismissing']:
            rangEemployees = TOKEN_RE.findall(row[rangeColName])
            if len(rangEemployees) == 2:
                df.loc[row_index,
                       targetColName] = random.randint(int(rangEemployees[0]),
                                                       int(rangEemployees[1]))
            else:
                df.loc[row_index, targetColName] = random.randint(
                    int(rangEemployees[0]),
                    int(rangEemployees[0]) * 2)
    return df


def fillSicUseNaics(df, NaicsToSictable=''):

    NaicsToSic = pd.read_csv(NaicsToSictable, delimiter=';')

    TOKEN_RE = re.compile(r'[\d]+')

    for row_index, row in df[df['sic_code'].isnull()].iterrows():
        rangEemployees = TOKEN_RE.findall(row['naics_codes'])
        for code in rangEemployees:
            sample = NaicsToSic.loc[NaicsToSic['naics_code'] == int(code)]
            sample = ', '.join(sample['sic_code'])
            df.loc[row_index, 'sic_codes'] = sample

    return df


def printNumericCols(df):
    # отбор числовых колонок
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values
    print(numeric_cols)


def printNonNumericCols(df):
    # отбор нечисловых колонок
    df_non_numeric = df.select_dtypes(exclude=[np.number])
    non_numeric_cols = df_non_numeric.columns.values
    print(non_numeric_cols)


def initMatplotlib():
    plt.style.use('ggplot')
    matplotlib.rcParams['figure.figsize'] = (12, 8)
    pd.options.mode.chained_assignment = None


def HeatmapCreate(df, cols=None):
    #Тепловая карта пропущенных значений

    if cols is None:
        cols = df.shape[1]

    cols = df.columns[:cols]
    # определяем цвета
    # желтый - пропущенные данные, синий - не пропущенные
    colours = ['#000099', '#ffff00']
    sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))


def printPercentLostData(df):
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing * 100)))


def dropRow(df, CriticalMiss):

    for col in df.columns:
        missing = df[col].isnull()
        num_missing = np.sum(missing)

        if num_missing > 0:
            print('created missing indicator for: {}'.format(col))
            df['{}_ismissing'.format(col)] = missing

    # отбрасываем строки с большим количеством пропусков
    ismissing_cols = [col for col in df.columns if 'ismissing' in col]
    df['num_missing'] = df[ismissing_cols].sum(axis=1)

    ind_missing = df[df['num_missing'] > CriticalMiss].index
    df = df.drop(ind_missing, axis=0)

    return df


def replacNaMedian(df, cols):

    for col in cols:
        med = df[col].median()
        df[col] = df[col].fillna(med)
    return df


def getTestingData(df, count):
    #получить данные с положительным и отрицательными трендами по последним трём месяцам
    q = df[df['S2_Dec19'] > df['S2_Jan20']]
    q = q[q['S2_Jan20'] > q['S2_Feb20']]
    q = q[(q['S2_Dec19'] - q['S2_Jan20']) > q['S2_Jan20'].describe()['25%'] *
          0.1]
    q = q[(q['S2_Jan20'] - q['S2_Feb20']) > q['S2_Feb20'].describe()['25%'] *
          0.05]
    ind = q.index[:count // 2]
    df = df.drop(ind, axis=0)

    test_row = q

    q = df[df['S2_Dec19'] < df['S2_Jan20']]
    q = q[q['S2_Jan20'] < q['S2_Feb20']]
    q = q[(q['S2_Jan20'] - q['S2_Dec19']) > q['S2_Dec19'].describe()['25%'] *
          0.2]
    q = q[(q['S2_Feb20'] - q['S2_Jan20']) > q['S2_Jan20'].describe()['25%'] *
          0.08]
    ind = q.index[:count - count // 2]
    df = df.drop(ind, axis=0)

    return pd.concat([test_row, q]), df
