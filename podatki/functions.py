import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from sklearn.utils import all_estimators
from sklearn import base
import sklearn.metrics as metrics 
from sklearn.exceptions import ConvergenceWarning

############################################################################################################
############################################################################################################

# UREJANJE TABEL, STATISTIKE

def read_txt(ime):
    df = pd.read_table(ime, sep=',')
    df[' valid'] = pd.to_datetime(df[' valid'], format='%Y/%m/%d')
    df.sort_values(by=' valid', inplace = True) #urejanje po datumu
    return df

# Zdruzitev tabel iz dveh let
def tabela(ime1, ime2):
   tabela1 = pd.read_table(ime1, sep=',')
   tabela2 = pd.read_table(ime1, sep=',')
   df = pd.concat([tabela1, tabela2], axis=0)
   df[' valid'] = pd.to_datetime(df[' valid'], format='%Y/%m/%d')
   df.sort_values(by=' valid', inplace = True) #urejanje po datumu
   return df

# stolpci tabele
def stolpci(data):
    d = {'column type': data.dtypes}
    return pd.DataFrame(data=d)

# tipi(count)
def count_type(data):
    d = {'count type' : data.dtypes.value_counts()}
    return pd.DataFrame(data = d)

# Glavne statistike tabele
def describe_tabela(data):
    df = pd.DataFrame(data.describe(include='all')).iloc[[0, 1, 4, 6, 8, 12]]
    df = df.append(data.dtypes, ignore_index = True)
    df = df.append(data.isna().sum(), ignore_index=True)
    df.index = ['count', 'unique', 'mean', 'min', 'max', 'column type', 'count NaN']
    print(f"Data shape: {data.shape}\nNumber of rows {data.shape[0]}\nNumber of columns {data.shape[1]}")
    return pd.DataFrame(df)

def statistike(data, type=" "):
    if type == "object":
        df = pd.DataFrame(data.describe(include=type)).iloc[[0, 1, 3]]
    else:
        df = pd.DataFrame(data.describe())
    return df

# postaje, id postaje, podatki za postajo
def postaje(data):
    d = {'station name': pd.unique(data[' station name'].values), 
         'station id': pd.unique(data['station id'].values)}
    df = pd.DataFrame(data=d)

    stolpci_id = {}
    for i in df['station name'].values:
        stolpci = []
        df2 = data[(data[' station name'] == i)]
        for col in df2.columns:
            if df2[col].isna().sum() == df2.shape[0] :
                col =+1
            else:
                stolpci.append(col)
        stolpci_id[i] = stolpci
    
    df['stolpci'] = df['station name'].map(stolpci_id)
    return df

# stetje object v stolpcu
def object_count(data, station_name, column):
    if station_name == " ":
        df = pd.DataFrame(data[column].value_counts())
    else:
        d = data[(data[' station name'] == station_name)]
        df = pd.DataFrame(d[column].value_counts())
    return station_name and df

# Visualize the Time Series
def plot_df(df, x, y, title='', ylabel= ''):
    '''
    df = tabela
    x = x os na grafu
    y = y os na grafu
    '''
    plt.figure(figsize=(10,4))
    plt.plot(x, y)
    plt.gca().set(title=title, xlabel='Date', ylabel=ylabel)
    plt.show()

def multilines_plot(data, x, y, ylabel='', title=''):
    '''
    y as ['', '']
    '''
    data.plot(x=x, y=y, figsize=(10,5))
    plt.gca().set(title=title, xlabel='Date', ylabel=ylabel)
    plt.show()


############################################################################################################
############################################################################################################
    
#  REGRESSION
def init_regressors():
    
    regressor_names=[]
    estimators = all_estimators()
    print ('Number of available regression models: ',len(estimators))

    # Check if the estimator is of subclass Regressor, if so append them to our list
    for name, estimator in estimators:
        if issubclass(estimator, base.RegressorMixin):        
            regressor_names.append(name)

    # Make a dataframe from the estimators
    regressors=pd.DataFrame(regressor_names,columns=['name'])
    
    # These estimators cause errors, bad results or long processing times, so we can remove them
    regressors=regressors[regressors['name'] != 'QuantileRegressor']
    regressors=regressors[regressors['name'] != 'StackingRegressor']
    regressors=regressors[regressors['name'] != 'TheilSenRegressor']

    # We save the regressors that we can use to select the best one later 
    regressors.to_csv(f'regressors.csv',index=False)
    
    # When correctly saved, we can load them and turn them into a list of all the names
    regressors=pd.read_csv(f'regressors.csv')
    regressors=regressors['name'].values.tolist()


# evaluating the model
def get_metrics(y_orig,y_pred,name,MAE,MSE,RMSE,R2, new_removed_regressors,verbose):
 
    # Calculate the metrics for the current estimator
    mae = metrics.mean_absolute_error(y_orig,y_pred)
    mse = metrics.mean_squared_error(y_orig,y_pred)
    rmse = np.sqrt(mse) # or mse**(0.5)  
    r2 = metrics.r2_score(y_orig,y_pred)

    # Show the results for each estimator while testing
    if verbose==True:
        print(f"Sklearn.metrics for {name}:")
        print("MAE:",mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R-Squared:", r2) 

    # Add the metric values to their corresponding dictionary
    MAE[name]=mae
    MSE[name]=mse
    RMSE[name]=rmse
    R2[name]=r2 

    # Remove this estimators if rmse is greater than 25% and add it to new_removed_regressors list
    if abs(rmse)>(y_orig.values.mean()/4):
        print(' >>> REMOVING ',name)
        new_removed_regressors.append(name)

    # If the new_removed_regressors list is empty, add the 'None' value to it, so we can return it
    try:
        new_removed_regressors
    except NameError:
        new_removed_regressors = 'None' 

    # Return the metric dictionaries and the list with removed_regressors
    return MAE,MSE,RMSE,R2,new_removed_regressors


# best regressor
def get_best_regressor(X_train,X_test,y_train,y_test,test_start,verbose):   

    # Create a metrics dictionary
    MAE={}
    MSE={}
    RMSE={}
    R2={} 

    # Create a list with regressors with too big error rates
    removed_regressors=[]      
    new_removed_regressors=[]
    
    # Get all estimators from sklearn
    estimators = all_estimators()
    print ('Number of available regression models: ',len(estimators))

    # Load all suitable estimators from .csv
    regressors=pd.read_csv(f'regressors.csv')

    # Remove the unsuitable estimators
    try: 
        removed_regressors=pd.read_csv(f'removed_regressors.csv')
        for regr_name in removed_regressors['name']:
            regressors=regressors[regressors['name'] != regr_name]
    except:
        removed_regressors=pd.DataFrame()

    # Make a list of all suitable estimators
    regressors=regressors['name'].values.tolist()

    """
     1. Loop through all estimators from the library, 
     2. Check if it is a regression model, 
     3. Check if it is in the suitable estimators list
     4. Fit on training data and predict values on testing data,
     5. Calculate the metrics
     5. Return the metrics and the non suitable estimators
    """
    for name, estimator in estimators:
        if issubclass(estimator, base.RegressorMixin): 
            if name in regressors:
                try:  
                    #print ('____________ ',estimator(),' _______________')

                    # Fit the estimator on the training data 
                    regression_model=estimator().fit(X_train, y_train)

                    # Prepare the test data
                    X_test_selected = X_test[X_train.columns]

                    # Predict values for test data
                    y_pred = pd.Series(regression_model.predict(X_test_selected), index=X_test_selected.index)

                    #this is for getting the metrics
                    y_orig=y_test.dropna().to_frame()           
                    y_pred=y_pred[test_start:].dropna()
                    y_pred = y_pred[y_pred.index.isin(y_orig.index)]
  
                    MAE,MSE,RMSE,R2,new_removed_regressors=get_metrics(y_orig,y_pred,name,MAE,MSE,RMSE,R2, new_removed_regressors,verbose) 
  
                except:
                    continue
                
    return MAE,MSE,RMSE,R2,removed_regressors,new_removed_regressors


# convert series to supervised learning
def data_preparation(data, window=1, look_forward=1, dropnan=True):
    # window = okno(pogledas nazaj)
    # look_forward = pogled naprej

    df = pd.DataFrame(data)
    cols, names = list(), list()

    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(data.shape[1])]
        # forecast sequence (t, t+1, ... t+n)
    
    # current timestep (t=0)
    for i in range(0, look_forward):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(data.shape[1])]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(data.shape[1])]
    
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.iloc[:,:data.shape[1] * window], agg.iloc[:,[agg.shape[1] - 1]]
