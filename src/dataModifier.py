import json
import re
import numpy as np


def json_load(path="data/train.json/train.json"):

    with open(path, "r") as read_file:
        dataJS = json.load(read_file)

    return dataJS


def get_categories(dataJS, types):

    Data = {}
    Rez = []
    for k in types:
        for q in dataJS[k]:
            if not q in Data:
                Data[q] = []
            Data[q].append(dataJS[k][q])
    for k in Data:
        Rez.append(Data[k])

    return Rez


def get_arr(dataJS, types):

    Data = []
    for k in types:
        for q in dataJS[k]:
            Data.append(dataJS[k][q])

    return Data


def modifier_fiches_type(dataJS, tupeConvert):

    for k in tupeConvert:
        for q in dataJS[k]:
            dataJS[k][q] = tupeConvert[k][dataJS[k][q]]

    return dataJS


def fullData_to_data(fullData):
    #'2016-06-16 05:55:27'
    Days = []
    for d in fullData:
        full2 = re.findall('\d{2}', d[0])
        t = full2[0] + full2[1] + '-' + full2[2] + '-' + full2[3]
        Days.append(t)

    return np.array(Days)


def fullData_to_time(fullData):
    #'2016-06-16 05:55:27'
    Time = []
    for d in fullData:
        full2 = re.findall('\d{2}', d[0])
        t = full2[4] + '-' + full2[5] + '-' + full2[6]
        Time.append(t)

    return np.array(Time)


def data_to_days(DataStr):

    Days = []
    for d in DataStr:
        full2 = re.findall('\d{2}', d)

        D = int(full2[3])
        D += int(full2[2]) * 30
        D += (int(full2[0] + full2[1])) * 365

        Days.append(D)

    return np.array(Days)


def data_to_MD(DataStr):

    MD = []

    for d in DataStr:
        full2 = re.findall('\d{2}', d)
        D = []
        #D.append(int(full2[0]+full2[1]))
        D.append(int(full2[2]))
        D.append(int(full2[3]))

        MD.append(D)

    return np.array(MD)


def time_to_sec(TimeStr):
    Time = []
    for d in TimeStr:
        full2 = re.findall('\d{2}', d)

        T = int(full2[2])
        T += int(full2[1]) * 60
        T += int(full2[0]) * 3600

        Time.append(T)

    return np.array(Time)


def time_to_HMS(TimeStr):

    HMS = []
    for d in TimeStr:
        full2 = re.findall('\d{2}', d)
        T = []
        T.append(full2[0])
        T.append(full2[1])
        T.append(full2[2])

        HMS.append(T)

    return np.array(HMS)


def to_one_hot(labels, demension=3):
    results = np.zeros((len(labels), demension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


def normalization(Data):
    mean = Data.mean(axis=0)
    Data -= mean
    std = Data.std(axis=0)
    Data /= std

    return Data


def mixedIndex(range):
    indices = np.arange(range.shape[0])
    np.random.shuffle(indices)

    return indices
