import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
print("________Ritik kashyap ________")

def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k ** 2))
    return weights

def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    W = (xmat.T * (wei * xmat)).I * (xmat.T * (wei * ymat.T))
    return W

def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    return ypred

# Load data points
data = pd.read_csv('data10.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)

# Preparing and add 1 in bill
mbill = np.mat(bill)
mtip = np.mat(tip)
m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mbill.T))

# Set k here
ypred = localWeightRegression(X, mtip, 2)

# Sort data points based on total_bill
SortIndex = X[:, 1].argsort(0)
xsort = X[SortIndex][:, 0]

# Plot the data points and predicted values
plt.scatter(bill, tip, color='green', label='Data')
plt.plot(xsort, ypred[SortIndex], color='red', linewidth=2, label='Predicted')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.legend()
plt.show()
