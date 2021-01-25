
import numpy as np
import sklearn as skl
import pandas as pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt


def plot_svc_decision_function(clf, ax=None):
    """Plot the decision function for a 2D SVC"""

    if ax is None:
        ax = plt.gca()

    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)

    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[i, j] = clf.decision_function([xi, yj])

    # plot the margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


#wczytsanie naszych danych
data = pandas.read_table("wdbc.data", sep=",", header=None)
data.drop([data.columns[13], data.columns[10], data.columns[11], data.columns[16], data.columns[31]], axis=1).head(2)


y = data.iloc[:, 1].values
train, test = train_test_split(data, test_size = 0.25, random_state = 1, stratify=y)


x = train.iloc[:, 2:].values 
y = train.iloc[:, 1].values

X_test = test.iloc[:, 2:].values
Y_test = test.iloc[:, 1].values

normalizer = Normalizer()
normalizer.transform(x)

#outliner_detector = EllipticEnvelope(contamination=.1)
#outliner_detector.fit_predict(x)


classifier = SVC(kernel='linear', random_state = 1, class_weight='balanced')
classifier.fit(x, y)


Y_pred = classifier.predict(X_test)
test["Predictions"] = Y_pred

print(test)

cm = confusion_matrix(Y_test,Y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)






plt.scatter(x[:, 17], x[:, 24], c=y, s=50, cmap='spring')
plot_svc_decision_function(classifier);