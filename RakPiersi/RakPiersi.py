
import numpy as np
import sklearn as skl
import pandas as pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer   
from sklearn.covariance import EllipticEnvelope   #pomogła przy wyznaczaniu wartości odstających - aczkolwiek tu okazała się zbędna
from sklearn.feature_selection import SelectPercentile, f_classif   #pomogła przy wyzanczeniu kolumn nieznaczących - przy tej liczbie danychrównież nie wpłyneła na wynik
import matplotlib.pyplot as plt


#wczytyanie naszych danych
data = pandas.read_table("wdbc.data", sep=",", header=None)
data.drop([data.columns[13], data.columns[10], data.columns[11], data.columns[16], data.columns[31]], axis=1).head(2)



#podział zbiorów
y = data.iloc[:, 1].values
train, test = train_test_split(data, test_size = 0.25, random_state = 1, stratify=y)


x = train.iloc[:, 2:].values 
y = train.iloc[:, 1].values

X_test = test.iloc[:, 2:].values
Y_test = test.iloc[:, 1].values


#normalizacja
normalizer = Normalizer()
normalizer.transform(x)


classifier = SVC(kernel='linear', random_state = 1, C=10)
classifier.fit(x, y)

normalizer = Normalizer()
normalizer.transform(X_test)

Y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nDokładność dla liniowej klasyfikacji SVM: ", accuracy)



classifierPoly = Pipeline([
    ("scaler", StandardScaler()), ("svm,clf",SVC(kernel="poly", C=13))])
classifierPoly.fit(x, y)

Y_pred = classifierPoly.predict(X_test)


cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nDokładność dla nieliniowej klasyfikacji SVM: ", accuracy)



classifier2 = Pipeline([
    ("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=5, loss="hinge"))])
classifier2.fit(x, y)

Y_pred = classifier2.predict(X_test)

cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nDokładność dla liniowej klasyfikacji SVM z użyciem zawiasowej funkcji straty: ", accuracy)



#dla 2 cech atrybutu 9, 26 - rysowanie wykresów - wizualizacja metody SVM


x = train.iloc[:, [9,25]].values 
y = train.iloc[:, 1].values

X_test = test.iloc[:, [9,25]].values
Y_test = test.iloc[:, 1].values


svc = SVC(kernel='linear', random_state = 1, C=10)
svc.fit(x, y)

Y_pred = svc.predict(X_test)

cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nDokładność dla liniowej klasyfikacji SVM według jedynie 2 parametrów równa jest: ", accuracy)


color = ["pink" if c=='B' else "yellow" for c in Y_test]
plt.scatter(X_test[:, 0], X_test[:, 1], c=color)
w = svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5 , 2.5) 
yy = a * xx - (svc.intercept_[0]) / w[1]
plt.plot(xx,yy)
plt.show()