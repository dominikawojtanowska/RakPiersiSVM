
import numpy as np
import sklearn as skl
import pandas as pandas
import matplotlib as plt
from sklearn.svm import SVC


#splitowanie danych 
def split_data(data, scale):
    shuff = np.random.permutation(len(data))
    size = int(len(data) * scale)
    test = shuff[:size]
    train = shuff[size:]
    return data.iloc[train], data.iloc[test]

#wczytsanie naszych danych
data = pandas.read_table("wdbc.data", sep=",", header=None)


#podział danych według naszej funkcji
train, test = split_data(data, 0.25)
#train
#x = train[:, (2,3)]
#y = (train[1] == 'M').astype(np.float64)

print(train)

x = train.iloc[:, 2:4].values
y = train.iloc[:, 1].values

classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(x,y)

#print("data", len(train), "dataa", len(test))
