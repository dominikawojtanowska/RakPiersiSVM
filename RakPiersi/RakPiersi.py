
import numpy as np
import sklearn as skl
import pandas as pandas


#splitowanie danych 
def split_data(data, scale):
    shuff = np.random.permutation(len(data))
    size = int(len(data) * scale)
    test = shuff[:size]
    train = shuff[size:]
    return data.iloc[train], data.iloc[test]

#wczytsanie naszych danych
data = pandas.read_table("wdbc.data", sep=",", header=None)
#data.info();

#podział danych według naszej funkcji
train, test = split_data(data, 0.25)

#print("data", len(train), "dataa", len(test))
