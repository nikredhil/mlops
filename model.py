import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('/Users/nikhilreddy/Downloads/aa/Iris.csv')
X = df.iloc[:, :4].values
y = df.iloc[:, -1].values
model = RandomForestClassifier().fit(X, y)
pickle.dump(model, open('model.pkl', 'wb'))
