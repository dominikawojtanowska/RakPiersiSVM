import numpy as np
import sklearn as skl
import pandas as pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

#wczytsanie naszych danych
data = pandas.read_table("wdbc.data", sep=",", header=None)

#podział danych według naszej funkcji
y = data.iloc[:, 1].values

train, test = train_test_split(data, test_size = 0.25, random_state = 1, stratify=y)

x = train.iloc[:, 2:].values
y = train.iloc[:, 1].values

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
print(x_scaled)

X_test = test.iloc[:, 2:].values
Y_test = test.iloc[:, 1].values

x_test_scaled = min_max_scaler.fit_transform(X_test)
# print(X_test)

classifier = SVC(kernel='linear', random_state = 1, class_weight='balanced')
classifier.fit(x_scaled,y)

Y_pred = classifier.predict(x_test_scaled)

# print(test)

cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)