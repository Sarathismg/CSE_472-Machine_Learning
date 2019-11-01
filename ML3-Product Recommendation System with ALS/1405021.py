import copy
import math
import time
import numpy as np
import pandas
from tqdm import tqdm

np.random.seed(17)

def readcsv(filename):
    '''
    :param filename:
    :return: array containing all elements directly
    '''

    dataframe = pandas.read_csv(filename)
    dataset = dataframe.values
    dataset = np.array(dataset)
    return dataset

def split_dataset(array, train_percent, val_percent, test_percent):
    '''

    :param array:
    :return: train, validation, test sets
    '''
    train = copy.deepcopy(array)
    valid = copy.deepcopy(array)
    test = copy.deepcopy(array)
    non_zero_pos = np.array(np.where(array[:, :] != 99)).T
    np.random.shuffle(non_zero_pos)
    train_pos = (non_zero_pos[0:int(train_percent * len(non_zero_pos))]).T
    val_pos = (non_zero_pos[int(train_percent * len(non_zero_pos)): int((train_percent + val_percent) * len(non_zero_pos))]).T
    test_pos = (non_zero_pos[int((train_percent + val_percent) * len(non_zero_pos)): ]).T
    train_pos = train_pos[0], train_pos[1]
    val_pos = val_pos[0], val_pos[1]
    test_pos = test_pos[0], test_pos[1]
    train[val_pos]=99
    train[test_pos]=99
    valid[train_pos] = 99
    valid[test_pos] = 99
    test[train_pos]= 99
    test[val_pos] = 99

    return train, valid, test

def columnOp(data, lambda_v, k, u):
    '''

    :param data:
    :param lambda_v:
    :param k:
    :param u:
    :return: v
    '''
    v = np.zeros((k, len(data[0]))) #k*m
    for col in tqdm(np.arange((len(v[0])))):
        acc = np.where(data[:, col]!= 99)
        u_temp = u[acc] #n' * k
        u_temp = u_temp.T #k*n'
        t1 = np.linalg.inv(np.dot(u_temp, u_temp.T) + lambda_v * np.identity(k)) #k*k
        t2 = np.sum(data[acc, col] * u_temp, axis = 1) # k
        t2 = np.reshape(t2, (len(t2),1))
        v[:, col] = np.dot(t1,t2).flatten()

    return v

def rowOp(data, lambda_u, k, v):
    '''

    :param data:
    :param lambda_u:
    :param k:
    :param v:
    :return: u
    '''
    u = np.zeros((len(data), k))
    for row in tqdm(np.arange(len(u))):
        acc = np.where(data[row,:] != 99)
        v_temp = v[:, acc[0]] #k*m'
        t1 = np.linalg.inv(np.dot(v_temp, v_temp.T) + lambda_u * np.identity(k)) #k*k
        t2 = np.sum(data[row, acc] * v_temp, axis = 1)
        t2 = np.reshape(t2, (len(t2),1))
        u[row] = np.dot(t1,t2).flatten()
    return u


def getRMSEloss(u, v, train_data):
    '''

    :param u:
    :param v:
    :param train_data:
    :return: RMSE error of valid terms
    '''
    places = np.where(train_data[:, :] != 99)
    derived = np.dot(u, v)
    sub = train_data[places] - derived[places]
    return math.sqrt(np.mean(np.square(sub)))


def train_model(train_data, validation_data, dim, lambda_u, lambda_v):
    '''

    :param train_data:
    :param dim:
    :param lambda_u:
    :param lambda_v:
    :return: trained model
    '''
    u = np.random.randn(len(train_data), dim)
    v = np.array([])
    converge = False
    epochs = 0
    prev_loss = 10000
    while(not converge):
        epochs += 1
        v = columnOp(train_data, lambda_v, dim, u)
        u = rowOp(train_data, lambda_u, dim, v)
        loss = getRMSEloss(u, v, train_data)
        val_loss = getRMSEloss(u, v, validation_data)
        print("Epoch: "+str(epochs)+"  , train_loss: "+str(loss)+"  ,val_loss: "+str(val_loss))
        if prev_loss - loss < 0.01:
            converge = True
            return (u, v), val_loss
        prev_loss = loss


all_data = readcsv('data.csv')
lambda_p = [0.01, 0.1, 1.0, 10.0]
k_arr = [5, 10, 20, 40]
train, validation, test = split_dataset(all_data, 0.6, 0.2, 0.2)
train, validation, test = train[:,1:], validation[:,1:], test[:, 1:]
lowest = 10000
best_model=(None,None)
t1 = time.time()

for l in lambda_p:
    for k in k_arr:
        print("==Train with k="+str(k)+"  l="+str(l))
        model, val_loss = train_model(train, validation, k, l, l)
        if val_loss < lowest:
            lowest = val_loss
            print("< BEST TILL NOW: k= "+str(k)+" l = "+str(l)+" >")
            best_model = model

t2 = time.time()

test_loss = getRMSEloss(best_model[0], best_model[1], test)
print("TEST LOSS: "+str(test_loss))
print("Total Training Time: "+ str(t2 - t1))
np.save('model_u',best_model[0])
np.save('model_v', best_model[1])