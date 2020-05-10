import numpy as np
import matplotlib.pyplot as plt
import re

import src
from src import dataModifier as DM
from src import edaPlotDataAdapt as DA

import eda as EDA

#   импорт данных из дата сета *.csv
#dataset = get_DataSet_on_numpy()
adspends_s1 = np.genfromtxt("data/DataSet/data.csv",
                            dtype=int,
                            delimiter=";",
                            skip_header=1,
                            usecols=(range(24, 35)))
naics = np.genfromtxt("data/DataSet/data.csv",
                      dtype=int,
                      delimiter=";",
                      skip_header=1,
                      usecols=(5))
sic = np.genfromtxt("data/DataSet/data.csv",
                    dtype=int,
                    delimiter=";",
                    skip_header=1,
                    usecols=(7))

naics = np.asarray(naics).astype('int')
sic = np.asarray(sic).astype('int')
Summs = np.asarray(DA.summByColumn(adspends_s1)).astype('int')
X1 = DA.groupByColumn(np.column_stack((naics, Summs)))
X2 = DA.groupByColumn(np.column_stack((sic, Summs)))

w1 = []
for q in X1:
    w1.append(np.asarray(X1[q]).astype('int'))
w1 = np.asarray(w1)

w2 = []
for q in X2:
    w2.append(np.asarray(X2[q]).astype('int'))
w2 = np.asarray(w2)


EDA.groupedBar(w1)
EDA.groupedBar(w2)

plt.show()