import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
import seaborn as sns 
import plotly.express as px
import csv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

#################################################################################################
#                                         PODATKI
#################################################################################################
X = pd.read_csv('features_filtered.csv')
y = pd.read_csv('y.csv')
feature_scores = pd.read_csv('feature_scores.csv')

feature_scores = feature_scores.rename(columns={'Unnamed: 0': 'feature', '0': 'value'})
selected_features = feature_scores[feature_scores['value'] != 0]
names_features = selected_features.loc[:,  'feature']
selected_X = X.loc[:, names_features]
selected_X['timestamp'] = X['timestamp']

#################################################################################################
#                                         Train, Test
#################################################################################################
# train, test
X_train = selected_X[selected_X['timestamp'] <= "2021-04-01"]
X_test = selected_X[selected_X['timestamp'] > "2022-01-01"]
y_train = y[y['date'] <= "2021-04-01"]# jan-mar
y_test = y[y['date'] > "2022-01-01"] # januar
del X_train["timestamp"], X_test["timestamp"], y_train["date"], y_test["date"]

#################################################################################################
#                                         MODELI
#################################################################################################
# transform data
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

transform = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category"))
    ]
)

# modeli
# DecisionTreeRegressor
dtr = Pipeline(
    steps = [
        ('dtr', DecisionTreeRegressor())
    ]
)
param_grid_dtr = [
    {
        'dtr__max_depth' : [3, 4, 5, 6] , 
        'dtr__min_samples_split' : [30, 40], 
        'dtr__min_samples_leaf' : [10, 12]
        #'dtr__' : [DecisionTreeRegressor()]
    } ]

# HistGradientBoostingRegressor
hgbr = Pipeline(
    steps = [
        ('hgbr', HistGradientBoostingRegressor())
    ]
)
param_grid_hgbr = [
    {
        'hgbr__loss': ['squared_error', 'absolute_error'],
        'hgbr__max_iter': [100, 150],
        'hgbr__max_bins': [10, 100, 255],
        'hgbr__min_samples_leaf': [10, 12]
        #'hgbr__' : [HistGradientBoostingRegressor()]
    }]

# KNeighborsRegressor
knr = Pipeline(
    steps = [
        ('transform', transform),
        ('knr', KNeighborsRegressor())
    ]
)
param_grid_knr = [
    {
        'knr__n_neighbors': [3, 5, 10]
        #'knr__' : [KNeighborsRegressor()]
     }]

# LinearRegression
lr = Pipeline(
    steps = [
        ('lr', LinearRegression())
    ]
)
param_grid_lr = [
     {
        #'lr__' : [LinearRegression()]
     }]

# RandomForestRegressor
rfr = Pipeline(
    steps = [
        ('rfr', RandomForestRegressor())
    ]
)
param_grid_rfr = [
     {
         'rfr__n_estimators': [100, 150], 
         'rfr__min_samples_split': [30, 40], 
         'rfr__min_samples_leaf': [10, 12],
         'rfr__max_features': ['sqrt', 'log2'], 
         'rfr__max_depth': [3, 4, 5, 6]
         #'rfr__' : [RandomForestRegressor()]
    }
]

# MLPRegressor
mlp = Pipeline(
    steps = [
        ('transform', transform),
        ('mlp', MLPRegressor())
    ]
)
param_grid_mpl = [
    {
       'mlp__hidden_layer_sizes': [(10,10), (5,5)],
       'mlp__alpha' :  [0.01, 0.001],
       'mlp__max_iter' : [100, 200] # popravi, ne skonvergira?!?
       #'mlp__' : [MLPRegressor()]
    } 
]

models = [dtr, hgbr, knr, lr, rfr, mlp]
param_grid = [param_grid_dtr, param_grid_hgbr, param_grid_knr, param_grid_lr, param_grid_rfr, param_grid_mpl]
model_name = ["DecisionTreeRegressor", "HistGradientBoostingRegressor", "KNeighborsRegressor",
              "LinearRegression", "RandomForestRegressor", "MLPRegressor"]

#################################################################################################
#                                         FITTING MODELS
#################################################################################################
# rezultati
model_all = {}
res = {}

for i in range(len(models)):
    # training
    print(f"Tuning model '{model_name[i]}'")
    model = GridSearchCV(models[i], param_grid[i])
    model.fit(X_train, y_train.values.ravel())
    #model_all.append(model)
    print(f"Best parameters for model '{model_name[i]}': {model.best_params_}\n")
    col_name = f"{model_name[i]}"
    if col_name not in model_all:
        model_all[col_name] = []
    model_all[col_name].append(model)
    model_all[col_name].append(model.best_params_)
        
    # prediction
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # scoring in shranjevanje rezultatov
    score_mse = mean_squared_error(test_pred, y_test)
    col_name = f"{model_name[i]}"
    if col_name not in res:
        res[col_name] = []
    res[col_name].append(score_mse)

    score_mape = mean_absolute_percentage_error(test_pred, y_test)
    col_name = f"{model_name[i]}"
    if col_name not in res:
        res[col_name] = []
    res[col_name].append(score_mape) 

#################################################################################################
#                                         SHRANJEVANJE
#################################################################################################
rezultati = pd.DataFrame(res)
rezultati.to_csv('rezultati_features.csv', index=False)

res = rezultati.T
res.index = res.index.set_names('name')
res.columns = ['MeanSquaredError', 'MeanAbsolutePercentageError']
res = res.reset_index()
res = res.sort_values(by=['MeanSquaredError', 'MeanAbsolutePercentageError'], ascending=True)

modeli = pd.DataFrame(model_all)
modeli.to_csv('modeli_features.csv', index=False)
mod = modeli.T
mod.index = mod.index.set_names('name')
mod.columns = ['GridSearch', 'opt_sprem']
mod = mod.reset_index()

#################################################################################################
#                                         FORCASTING
#################################################################################################
opt_num = res.index.values[0]
print(f"Optimal model is {mod.loc[opt_num, 'name']} with number {opt_num}")

model_opt = mod.loc[opt_num]
model_opt['opt_sprem']

opt_mod = RandomForestRegressor(max_depth = 5, max_features='sqrt', min_samples_leaf=10, min_samples_split=40, n_estimators=100)
opt_mod.fit(X_train, y_train.values.ravel())

ypred_train = opt_mod.predict(X_train)
ypred_test = opt_mod.predict(X_test)

ypred = pd.DataFrame(ypred_test)
ypred.to_csv('ypred_fetures.csv', index=False)