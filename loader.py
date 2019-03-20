from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class load: #in future verions add scaling options, test_size, dataset model etc..
    def __init__(self, dataset = 'iris', poc=0.5, test_size=0.2):
        if dataset is 'iris':
            ds = datasets.load_iris()
        elif dataset is 'mnist':
            ds = datasets.load_digits()
        elif dataset is 'boston':
            ds = datasets.load_boston()
        elif dataset is 'wine':
            ds = datasets.load_wine()
        n_samples = len(ds.target)
        DSL = int(poc*n_samples) #small dataset limit - usually 100

        X = ds.data[:DSL]
        y = ds.target[:DSL]

        # print(y.max())
        #scaling:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size = test_size, random_state=137)