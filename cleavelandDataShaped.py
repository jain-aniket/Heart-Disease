from numpy import genfromtxt
import numpy as np
cleavelandData = genfromtxt("processedclevelanddata.csv", delimiter=',')

filteredData = cleavelandData[~np.isnan(cleavelandData).any(axis=1)]

data,labels = np.split(filteredData, [13], axis = 1)

print(filteredData)
