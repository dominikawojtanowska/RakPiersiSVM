
import numpy as np
import sklearn as skl
import pandas as pandas
import matplotlib as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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


cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)

#print("data", len(train), "dataa", len(test))
