import numpy as np
import matplotlib.pyplot as plt
import math
import random

def get_gaussian(x, miu_k, sigma_k, dimension = 2):
    insideexp = np.matmul((x - miu_k), np.linalg.inv(sigma_k))
    insideexp = np.matmul(insideexp, np.array((x-miu_k)).transpose())
    insideexp = (-1/2)*insideexp

    #return (1/((2*math.pi)**(dimension/2)))*(np.linalg.det(sigma_k)**(-0.5))*(math.exp(insideexp))
    return (1 / math.sqrt(((2 * math.pi) ** dimension) * np.linalg.det(sigma_k))) * (math.exp(insideexp))

f = np.random.uniform(1,8,[20000, 2])
print(f)
a = []
for m in f:
    s = random.uniform(0, 1)
    if get_gaussian(m, [4, 4], [[.01, 0],[0,.01]]) > s:
        a.append(m)
a = np.array(a)
print(a.shape)

plt.plot(a[:,0],a[:,1], 'ro')
plt.show()
