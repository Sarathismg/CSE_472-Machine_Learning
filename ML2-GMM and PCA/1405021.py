import math
from sklearn import datasets

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

fname = 'data.txt'
full_array = []
labels = []

def calcsigma(x_arr, miu = None):
    if miu == None:
        miu = np.mean(x_arr, axis=0)
    x_arr_temp = x_arr - miu
    sum = np.zeros((len(x_arr_temp[0]),len(x_arr_temp[0])))
    for k in range(0, len(x_arr_temp)):
        sum = sum + np.outer(x_arr_temp[k], x_arr_temp[k])

    sum = sum / len(x_arr_temp)
    return sum

def get_gaussian(x, miu_k, sigma_k, dimension = 2):
    insideexp = np.matmul((x - miu_k), np.linalg.inv(sigma_k))
    insideexp = np.matmul(insideexp, np.array((x-miu_k)).transpose())
    insideexp = (-1/2)*insideexp

    return (1 / math.sqrt(((2 * math.pi) ** dimension) * np.linalg.det(sigma_k))) * (math.exp(insideexp))

def evaluate(x, params):
    total = 0
    for m in range(0, len(x)):
        subtotal = 0
        for k in range(0, len(params)):
            miu1, sig1, w1 = params[k]
            subtotal += w1 * get_gaussian(x[m], miu1, sig1)
        total += math.log(subtotal)
    return total


def dyna_plot(x, prob):
    green = []
    blue = []
    red = []
    for m in range(0, len(x)):
        if prob[m][0] > prob[m][1] and prob[m][0] > prob[m][2]:
            green.append(x[m])
        elif prob[m][1] > prob[m][0] and prob[m][1] > prob[m][2]:
            blue.append(x[m])
        else:
            red.append(x[m])

    green = np.array(green)
    red = np.array(red)
    blue = np.array(blue)
    plt.plot(green[:, 0], green[:, 1], 'go')
    plt.plot(red[:, 0], red[:, 1], 'ro')
    plt.plot(blue[:, 0], blue[:, 1], 'bo')
    plt.pause(0.1)



with open(fname) as f1:
    content = f1.readlines()

for j in range(0, len(content)):
    temp = content[j].split()
    full_array.append(temp)


full_array = np.array(full_array).astype(float)
#print(full_array.shape)
temp2 = full_array - np.mean(full_array, axis=0)
temp3 = []
for i in range(0, len(full_array)):
    z = np.outer(temp2[i], temp2[i])
    temp3.append(z)

temp3 = np.array(temp3)
temp3 = sum(temp3)

w, v = LA.eig(temp3)
v = v[:, :2]
p = []
for k in range(len(full_array)):
    p.append(np.matmul(full_array[k], v))
p = np.array(p)
plt.plot(p[:,0],p[:,1], 'bo')
plt.show()
print(p[0])

no_of_clusters = int(input())
params = []
min_p = np.min(p, axis = 0)
max_p = np.max(p, axis= 0)
rnd = random.Random()

for i in range(0, no_of_clusters):
    miu = (random.uniform(float(min_p[0]), float(max_p[0])), random.uniform(float(min_p[1]), float(max_p[1])))
    miu = (random.uniform(0,1), random.uniform(0,1))
    sigma = calcsigma(p, miu)
    w = 1 / no_of_clusters
    params.append((miu, sigma, w))

epochs = 100
converge = False
last = 0

while( converge==False and epochs > 0):
    epochs -= 1
    prob = []
    for m in range(0, len(p)):
        prob_i = []
        temp = []
        for k in range(0, no_of_clusters):
            miu1, sig1, w1 = params[k]
            #print("gauss: "+str(get_gaussian(p[m], miu1, sig1)))
            t = w1 * get_gaussian(p[m], miu1, sig1)
            temp.append(t)
        temp=np.array(temp)
        prob_i = temp / sum(temp)
        prob.append(prob_i)

    '''M step'''
    for k in range(0, no_of_clusters):
        miu_t = np.zeros(2)
        '''dim = 2'''
        sig_t = np.zeros((len(p[0]), len(p[0])))
        w_t = 0
        for m in range(0 , len(p)):
            miu_t += prob[m][k]*p[m]
        miu_t = miu_t / sum(prob)[k]

        arr_temp = p - miu_t
        for m in range(0, len(p)):
            sig_t = sig_t + prob[m][k] * np.outer(arr_temp[m], arr_temp[m])

        sig_t = sig_t / sum(prob)[k]
        w_t = sum(prob)[k] / len(arr_temp)
        params[k] = (miu_t, sig_t, w_t)
    res = evaluate(p, params)
    print("epoch: "+str(100-epochs)+" ,  log-likelihood: "+str(res))
    if abs(last - res) < 0.0001:
        print("convergence criterion")
        plt.show()
        break
    else:
        last = res
    dyna_plot(p, prob)

a1,a2,a3 = params[0]
a4,a5,a6 = params[1]
a7,a8,a9 = params[2]

print(a1)
print(a4)
print(a7)
print("--")
print(a3)
print("--")
print(a6)
print("--")
print(a9)
