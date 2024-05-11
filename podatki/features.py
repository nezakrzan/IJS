import numpy as np
import pandas as pd
import seaborn as sns 
import tsfresh as ts

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import impute

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV

import matplotlib.pyplot as plt

df = pd.read_csv('data_df.csv')

df1 = df[(df['date'] > "2021-12-31") & (df['date'] <= "2022-02-01")]
df2 = df[df['date'] <= "2021-04-01"]
data = pd.concat([df2, df1], ignore_index=True)

# postaja CELJE - nima difuznega sevanja
data = data[data[' station name'] == "ROGLA"]

data = data.dropna(axis=1)
data.reset_index(inplace = True)
data["id"] = (data.index // 48)

# set target
target = data.loc[:,["date", "DF"]]

# stolpci, ki ne ustrezajo
del data["DF"], data[" station name"], data["index"]

data_rolled = roll_time_series(data, column_id="id", column_sort="date", max_timeshift=48)
X = extract_features(data_rolled, column_id="id", column_sort="date")

# INDEXING
# za povezavo s targetom
X = X.set_index(X.index.map(lambda x: x[1]), drop=True)
X.index.name = "timestamp"

# CREATE TARGET
y_i = target.set_index("date").sort_index()
# Equal data and target rows
y_i = y_i[y_i.index.isin(X.index)]
X = X[X.index.isin(y_i.index)]

y = pd.Series(y_i['DF'])

X_utilities = ts.utilities.dataframe_functions.impute(X)
#features_filtered = ts.select_features(X, target["DF"])
features_filtered = ts.select_features(X_utilities, y)

features_filtered.to_csv('features_filtered.csv')
y.to_csv('y.csv')

# random forest za featurse koliko se jih splača
# wrapanje seprv izbiras featurse, dodajs notr ali ven
# accourency in tisti ki so 0 lahko daš ven

# In all feature selection procedures, it is a good practice to select the features by 
# examining only the training set. This is to avoid overfitting.
X = pd.read_csv('features_filtered.csv')
y = pd.read_csv('y.csv')

X_train = X[X['timestamp'] <= "2021-04-01"]
X_test = X[X['timestamp'] > "2022-01-01"]
y_train = y[y['date'] <= "2021-04-01"]# jan-mar
y_test = y[y['date'] > "2022-01-01"] # januar

del y_train["date"], y_test["date"], X_train["timestamp"], X_test["timestamp"]

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# y_train = scaler.transform(y_train)
# y_test = scaler.transform(y_test)

# model fitting and feature selection altogether 
# Izbrali bomo tiste značilnosti, katerih pomembnost je večja od 
# povprečne pomembnosti vseh značilnosti privzeto, vendar lahko to prag spremenimo, če želimo
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train.values.ravel())

y_pred = model.predict(X_test)

print('Model mean squared error score: {0:0.4f}'. format(mean_squared_error(y_test, y_pred)))

feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_scores

feature_scores.to_csv('feature_scores.csv')
feature_scores = pd.read_csv('feature_scores.csv')
feature_scores = feature_scores.rename(columns={'Unnamed: 0': 'feature', '0': 'value'})

selected_features = feature_scores[feature_scores['value'] != 0]
names_features = selected_features.loc[:,  'feature']

selected_X = X.loc[:, names_features]
selected_X['timestamp'] = X['timestamp']

X_train = selected_X[selected_X['timestamp'] <= "2021-04-01"]
X_test = selected_X[selected_X['timestamp'] > "2022-01-01"]
del X_train["timestamp"], X_test["timestamp"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train.values.ravel())


