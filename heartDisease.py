import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
clevelandData = genfromtxt("processedclevelanddata.csv", delimiter=',')
filteredData = clevelandData[~np.isnan(clevelandData).any(axis=1)]
data,labels = np.split(filteredData, [13], axis = 1)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.5, shuffle=False)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)



#X_test = [[0, 0, 16, 16, 16, 16, 0, 0, 0, 16, 0, 0, 0, 0, 16, 0, 0, 16, 0, 0, 0, 0, 16, 0, 0, 16, 0, 0, 0, 0, 16, 0, 0, 16, 0, 0, 0, 0, 16, 0, 0, 16, 0, 0, 0, 0, 16, 0, 0, 16, 0, 0, 0, 0, 16, 0, 0, 0, 16, 16, 16, 16, 0, 0]]

# Now predict the value of the digit on the second half:
predicted = classifier.predict(X_test)



print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)
