import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('C:\\Users\\vasu\\Desktop\\DATASET\\housing.csv')
dataset.isnull().sum()

X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]].values
y = dataset.iloc[:, 8].values

from sklearn.impute import SimpleImputer
sim = SimpleImputer(missing_values = np.nan, strategy = 'median')
sim.fit(X[:, [4]])
sim.statistics_
X[:, [4]] = sim.transform(X[:, [4]])
# sim.fit_transform()

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, 8] = lab.fit_transform(X[:, 8])
lab.classes_

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.score(X, y)

y_pred = lin_reg.predict(X)




























































