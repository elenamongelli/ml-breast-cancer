import numpy as np
import xlrd

import matplotlib.pyplot as plt
from scipy.linalg import svd

# Load xls sheet with data
doc = xlrd.open_workbook('C:/Users/Elena/Desktop/data_test_pca.xls').sheet_by_index(0)

# Extract attribute names (1st row, column 1 to 30)
attributeNames = doc.row_values(0, 2, 32)

# Extract class names to python list,
# then encode with integers (dict)

#column 1, from 1st row to 569th row
classLabels = doc.col_values(1, 1, 569)
classNames = sorted(set(classLabels))

#number of attributes
classDict = dict(zip(classNames, range(30)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Preallocate memory, then extract excel data to matrix X
X = np.empty((568, 30))
for i, col_id in enumerate(range(2, 31)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, 569))

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

print('Ran Exercise 2.1.1')

## PCA analysis (ex 2.1.3)

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

print('Ran Exercise 2.1.3')

Y = X - np.ones((N,1))*X.mean(0)

U,S,Vh = svd(Y,full_matrices=False)
V=Vh.T
N,M = X.shape

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0, 1, 2, 3, 4, 5, 6]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b', 'c', 'm', 'y', 'k']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames, rotation=90)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
# plt.legend(legendStrs, loc="best")
plt.grid()
plt.title('Cancer: PCA Component Coefficients')
plt.show()

print('PC1:')
print(V[:,0].T)

# How does this translate to the actual data and its projections?
# Looking at the data for PC1:

# Projection of cancer class onto the 2nd principal component.
all_cancer_data1 = Y[y==1,:]
all_cancer_data0 = Y[y==0,:]

print('First cancer observation')
print(all_cancer_data1[0,:])

print('...and its projection onto PC2')
print(all_cancer_data1[0,:]@V[:,1])
