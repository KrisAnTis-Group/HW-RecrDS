# импорт пакетов
import pandas as pd
import numpy as np

import re
import random


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
