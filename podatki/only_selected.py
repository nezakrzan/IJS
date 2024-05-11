import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
import seaborn as sns 
import plotly.express as px
import csv

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data_df.csv')

data = df[["date", "globalno sev. [W/m2]", "DF", "energija uvb [mW/m2]", "week_number"]]
data = df.dropna()
data.reset_index(inplace = True)

# set target
target = data.loc[:,["DF"]]

del data["index"], data[" station name"], data["date"], data["DF"]

# set data
df = data.loc[:30000]
target = target.loc[:30000]

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
       'mlp__max_iter' : [500, 800]
       #'mlp__' : [MLPRegressor()]
    } 
]

models = [dtr, hgbr, knr, lr, rfr, mlp]
param_grid = [param_grid_dtr, param_grid_hgbr, param_grid_knr, param_grid_lr, param_grid_rfr, param_grid_mpl]
model_name = ["DecisionTreeRegressor", "HistGradientBoostingRegressor", "KNeighborsRegressor",
              "LinearRegression", "RandomForestRegressor", "MLPRegressor"]


from sklearn.model_selection import train_test_split
n_repeats = 1
res = {}
model_all = {}

for i in range(n_repeats):
    # train test data
    X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=0.7)

    for l in range(len(models)):
        # training
        print(f"Tuning model '{model_name[l]}'")
        model = GridSearchCV(models[l], param_grid[l])
        model.fit(X_train, y_train.values.ravel())
        #model_all.append(model)
        #print(f"Best parameters for model '{model_name[l]}': {model.best_params_}\n")
        col_name = f"{model_name[l]}"
        if col_name not in model_all:
            model_all[col_name] = []
        model_all[col_name].append(model)
        model_all[col_name].append(model.best_params_)
        
        # prediction
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # scoring in shranjevanje rezultatov
        score_mse = mean_squared_error(test_pred, y_test)
        col_name = f"{model_name[l]}_mse"
        if col_name not in res:
            res[col_name] = []
        res[col_name].append(score_mse)

        score_mape = mean_absolute_percentage_error(test_pred, y_test)
        col_name = f"{model_name[l]}_mape"
        if col_name not in res:
            res[col_name] = []
        res[col_name].append(score_mape)
