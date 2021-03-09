import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio

#Gaussian kernel function
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.power(x1-x2, 2).sum() /(2 * (sigma ** 2)))

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
ans = gaussian_kernel(x1, x2, sigma)
print(ans)

#load data
mat = sio.loadmat('./data/ex6data2.mat')
print(mat.keys())
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')
data.head()
