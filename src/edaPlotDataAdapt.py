import numpy as np
from random import randint


def getColors(countColors = 1):
    colors = []
    for i in range(countColors):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    return colors

def groupByColumn (data, col = 0):

    Z = {}
    for q in data:
        if not q[col] in Z:
            Z[q[col]]=[]
        Z[q[col]].append(q[1])

    return Z

def summByColumn (data):

    Summs=[]
    for q in data:
        Summs.append(np.sum(q))
        
    return Summs