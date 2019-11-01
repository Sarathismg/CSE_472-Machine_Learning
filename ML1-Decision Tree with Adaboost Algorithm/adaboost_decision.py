import csv
from sklearn import preprocessing
import pandas
import numpy as np


class Tree:
    def __init__(self, class_no):
        self.array =[]
        self.class_no = class_no

    def push_node(self, t):
        self.array.append(t)

def LabelEncode(a,no):
    enc = preprocessing.LabelEncoder()
    enc.fit(a[:,no])
    b = enc.transform(a[:,no])
    a[:,no] = b

def cleanUnLabelledData(a):
    for k in range(0,len(a)):
        if a[k][-1] == " ":
            np.delete(a,k,0);
            k = k - 1;


def fillMissingData(a, ind, isLabel):
    if isLabel == True:
        z = np.ndarray.tolist(a[:,ind])
        dat = max(z,key=z.count)
        for k in range(0, len(a)):
            if a[k][ind] == " ":
                a[k][ind] = dat

    else:


dataset = []

def preprocessed_ds1():
    dataframe = pandas.read_csv("Churn.csv")
    dataset = dataframe.values
    dataset = np.array(dataset)
    LabelEncode(dataset, 0)
    LabelEncode(dataset, 1)
    LabelEncode(dataset, 3)
    LabelEncode(dataset, 4)
    LabelEncode(dataset, 6)
    LabelEncode(dataset, 7)
    LabelEncode(dataset, 8)
    LabelEncode(dataset, 9)
    LabelEncode(dataset, 10)
    LabelEncode(dataset, 11)
    LabelEncode(dataset, 12)
    LabelEncode(dataset, 13)
    LabelEncode(dataset, 14)
    LabelEncode(dataset, 15)
    LabelEncode(dataset, 16)
    LabelEncode(dataset, 17)
    LabelEncode(dataset, 20)
    return dataset




dataset = preprocessed_ds1()
print(dataset[0])