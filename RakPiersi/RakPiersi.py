
import numpy as np
import sklearn as skl
import pandas as pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


#wczytsanie naszych danych
data = pandas.read_table("wdbc.data", sep=",", header=None)


#podział danych według naszej funkcji
train, test = train_test_split(data, test_size = 0.2, random_state = 1)
#train
#x = train[:, (2,3)]
#y = (train[1] == 'M').astype(np.float64)

print(train)

x = train.iloc[:, 2:4].values
y = train.iloc[:, 1].values

X_test = test.iloc[:, 2:4].values
Y_test = test.iloc[:, 1].values

classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(x,y)

Y_pred = classifier.predict(X_test)
test["Predictions"] = Y_pred

print(test)

#cm = confusion_matrix(Y_test,Y_pred)
#accuracy = float(cm.diagonal().sum())/len(Y_test)
#print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)


plt.figure(figsize = (7,7))
X_set, y_set = x, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('black', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'orange'))(i), label = j)

plt.title('rak')

plt.show()

#print("data", len(train), "dataa", len(test))
