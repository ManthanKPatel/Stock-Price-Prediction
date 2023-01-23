#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Initial Part


import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
#import fAux
import matplotlib.pyplot as plt
import seaborn as sns
import sys

np.random.seed() #to fix the results
 


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from datetime import datetime
import alphalens

from time import time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import talib as ta
from alphalens.tears import create_summary_tear_sheet
from alphalens.utils import get_clean_factor_and_forward_returns
import re

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit


# In[3]:


file_path = 'outputfile.txt'
sys.stdout = open(file_path, "w")


# In[4]:


df_Gold = pd.read_csv('Gold Futures Historical Data.csv')


# In[5]:


df_Crude = pd.read_csv('Crude Oil WTI Futures Historical Data.csv')


# In[6]:


df = pd.read_csv('Try-3.csv')


# In[7]:


df['DATE'] = pd.to_datetime(df['date'] )
df_Crude['DATE'] = pd.to_datetime(df_Crude['Date'] )
df_Gold['DATE'] = pd.to_datetime(df_Gold['Date'] )


# In[8]:


df.set_index(["DATE"],inplace=True)
df.drop(['date'], axis=1, inplace=True)

df_Gold.set_index(["DATE"],inplace=True)
df_Gold.drop(['Date'], axis=1, inplace=True)

df_Crude.set_index(["DATE"],inplace=True)
df_Crude.drop(['Date'], axis=1, inplace=True)


# In[9]:


df_Gold = df_Gold.iloc[::-1]
df_Crude = df_Crude.iloc[::-1]


# In[10]:


Stock_name = df['symbol'].unique()


# In[11]:


df_SandP = pd.read_csv('S&P 500 Historical Data.csv')
df_Nasdaq = pd.read_csv('Nasdaq Historical Data.csv')
df_DJIA = pd.read_csv('Dow Jones Industrial Average Historical Data.csv')


# In[12]:


df_SandP['DATE'] = pd.to_datetime(df_SandP['Date'])
df_Nasdaq['DATE'] = pd.to_datetime(df_Nasdaq['Date'])
df_DJIA['DATE'] = pd.to_datetime(df_DJIA['Date'])
df_Nasdaq.rename(columns = {'Close':'Price'},inplace = True)


# In[13]:


df_SandP.set_index(["DATE"],inplace=True)
df_Nasdaq.set_index(["DATE"],inplace=True)
df_DJIA.set_index(["DATE"],inplace=True)


# In[14]:


df_SandP.drop(['Date'], axis=1, inplace=True)
df_Nasdaq.drop(['Date'], axis=1, inplace=True)
df_DJIA.drop(['Date'], axis=1, inplace=True)


# In[15]:


df_SandP = df_SandP.iloc[::-1]
df_DJIA = df_DJIA.iloc[::-1]


# In[16]:


df_SandP = df_SandP.astype(float, errors = 'ignore')
df_Nasdaq = df_Nasdaq.astype(float, errors = 'ignore')
df_DJIA = df_DJIA.astype(float, errors = 'ignore')


# In[17]:


Aa = df['OPrice'].loc[df['symbol'] == Stock_name[0]]
Ag = df['OPrice'].loc[df['symbol'] == Stock_name[1]]
In = df['OPrice'].loc[df['symbol'] == Stock_name[2]]
Crude = df_Crude['Price']

df_Gold['Price'] = df_Gold['Price'].apply(lambda x: str(x.replace(',','')))
df_Gold['Price'] = df_Gold['Price'].astype(float, errors = 'ignore')
Gold = df_Gold['Price']


# In[18]:


df_SandP = df_SandP.applymap(lambda x: str(x. replace(',','')))
df_DJIA = df_DJIA.applymap(lambda x: str(x. replace(',','')))


# In[19]:


df_List = [df_SandP,df_Nasdaq,df_DJIA]


# In[20]:


for column in Stock_name:
    df_Target = df['OPrice'].loc[df['symbol'] == column]
    for df_Feat in df_List:
        df_Feat['Price'] = df_Feat['Price'].astype(float, errors = 'ignore')
        df_Feat['Open'] = df_Feat['Open'].astype(float, errors = 'raise')
        df_Feat['Low'] = df_Feat['Low'].astype(float, errors = 'raise')
        df_Feat['High'] = df_Feat['High'].astype(float, errors = 'raise')


# In[21]:


df_SandP = df_SandP.interpolate()
df_Nasdaq = df_Nasdaq.interpolate()
df_DJIA = df_DJIA.interpolate()


# In[22]:


### Indicators ###
def simple_moving_average(df):
    df['SMA_5'] = df['Open'].transform(lambda x:x.rolling(window = 5).mean())
    df['SMA_15'] = df['Open'].transform(lambda x:x.rolling(window = 15).mean())
    df['SMA_30'] = df['Open'].transform(lambda x:x.rolling(window = 30).mean())
    df['SMA_60'] = df['Open'].transform(lambda x:x.rolling(window = 60).mean())

    df['SMA_ratio'] = df['SMA_60'] / df['SMA_5']
    
    return df

def MACD(df):
    df['ewm_15'] = df['Open'].transform(lambda x:x.ewm(span=15, adjust = False).mean())
    df['ewm_30'] = df['Open'].transform(lambda x:x.ewm(span=30, adjust = False).mean())
    df['MACD'] = df['ewm_30'] - df['ewm_15']
    return df

def stochastic_oscillators(df):
    df['LOWEST15D'] = df['Low'].transform(lambda x:x.rolling(window = 15).min())
    df['HIGHEST15D'] = df['High'].transform(lambda x:x.rolling(window = 15).max())
    
    df['LOWEST30D'] = df['Low'].transform(lambda x:x.rolling(window = 30).min())
    df['HIGHEST30D'] = df['High'].transform(lambda x:x.rolling(window = 30).max())

    df['Stochastic_15'] = ((df['Open'] - df['LOWEST15D'])/(df['HIGHEST15D'] - df['LOWEST15D']))*100
    df['Stochastic_30'] = ((df['Open'] - df['LOWEST30D'])/(df['HIGHEST30D'] - df['LOWEST30D']))*100
    
    df['Stochastic_%D_15'] = df['Stochastic_15'].rolling(window = 15).mean()
    df['Stochastic_%D_30'] = df['Stochastic_30'].rolling(window = 30).mean()
    
    df['Stochastic_Ratio'] = df['Stochastic_%D_15']/df['Stochastic_%D_30']
    return df

def ROC(df):
    df['RC_5'] = df['Open'].transform(lambda x: x.pct_change(periods = 5)) 
    df['RC_15'] = df['Open'].transform(lambda x: x.pct_change(periods = 15)) 
    df['RC_30'] = df['Open'].transform(lambda x: x.pct_change(periods = 30)) 
    df['RC_60'] = df['Open'].transform(lambda x: x.pct_change(periods = 60)) 
    return df

    
import talib as ta

def RSI(df):
    df['RSI_5']=ta.RSI(np.array(df['Open']), timeperiod=5)
    df['RSI_15']=ta.RSI(np.array(df['Open']), timeperiod=15)
    df['RSI_30']=ta.RSI(np.array(df['Open']), timeperiod=30)
    df['RSI_60']=ta.RSI(np.array(df['Open']), timeperiod=60)
    return df
    
def ADX(df):
    df['ADX_5']=ta.ADX(np.array(df['High']),np.array(df['Low']), np.array(df['Open']), timeperiod =5)
    df['ADX_15']=ta.ADX(np.array(df['High']),np.array(df['Low']), np.array(df['Open']), timeperiod =15)
    df['ADX_30']=ta.ADX(np.array(df['High']),np.array(df['Low']), np.array(df['Open']), timeperiod =30)
    df['ADX_60']=ta.ADX(np.array(df['High']),np.array(df['Low']), np.array(df['Open']), timeperiod =60)
    return df

    


# In[23]:


for df_temp in [df_SandP,df_DJIA,df_Nasdaq]:
    df_temp = simple_moving_average(df_temp)
    df_temp = MACD(df_temp)
    df_temp = stochastic_oscillators(df_temp)
    df_temp = ADX(df_temp)
    df_temp = ROC(df_temp)
    df_temp = RSI(df_temp)
    


# In[24]:


df_SandP = df_SandP.fillna(0)
df_Nasdaq = df_Nasdaq.fillna(0)
df_DJIA = df_DJIA.fillna(0)


# In[25]:


for df_Feat in [df_SandP,df_DJIA,df_Nasdaq]:
    for i in Crude.index:
        if i in df_Feat.index:
            df_Feat.loc[i,'Crude'] = Crude.loc[i]

for df_Feat in [df_SandP,df_DJIA,df_Nasdaq]:
    for i in Gold.index:
        if i in df_Feat.index:
            df_Feat.loc[i,'Gold'] = Gold.loc[i]


# In[26]:


for column in Stock_name:
    df_Target = df['OPrice'].loc[df['symbol'] == column]
    for df_Feat in [df_SandP,df_DJIA,df_Nasdaq]:
        for i in df_Target.index:
            df_Feat.loc[i,column] = df_Target.loc[i]


# In[27]:


df_SandP = df_SandP.drop(['Change %','Vol.','AAPL'],axis = 1)
df_SandP = df_SandP[df_SandP.index < '2012-01-01'] 
df_SandP = df_SandP[df_SandP.index > '2001-01-01']


# In[28]:


df_DJIA = df_DJIA.drop(['Change %','AAPL'],axis = 1)
df_DJIA = df_DJIA[df_DJIA.index < '2012-01-01']
df_DJIA = df_DJIA[df_DJIA.index > '2001-01-01']


# In[29]:


df_Nasdaq = df_Nasdaq.drop(['AAPL'],axis = 1)
df_Nasdaq = df_Nasdaq[df_Nasdaq.index < '2012-01-01']
df_Nasdaq = df_Nasdaq[df_Nasdaq.index > '2001-01-01']


# In[30]:


df_SandP = df_SandP.interpolate()
df_Nasdaq = df_Nasdaq.interpolate()
df_DJIA = df_DJIA.interpolate()

df_SandP = df_SandP.iloc[1:,:]
df_Nasdaq = df_Nasdaq.iloc[1:,:]

df_DJIA = df_DJIA.iloc[1:,:]


# In[31]:


df_Nasdaq = df_Nasdaq.drop(["Volume","Adj Close"],axis=1)
df_DJIA = df_DJIA.drop(["Vol."],axis=1)


# In[32]:


df_DJIA.shape


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer 
from sklearn import preprocessing
#import phik
#from phik.report import plot_correlation_matrix
from scipy.special import ndtr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, f1_score, mean_squared_error as mse, mean_absolute_error as mae
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
#from mixed_naive_bayes import MixedNB
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


# In[34]:


X = df_SandP.drop(['AMGN','INTC'], axis = 1)
y = df_SandP[['AMGN','INTC']]

X_train_SandP = X.iloc[0:round(0.7*(len(X.index)))]
X_test_SandP = X.iloc[round(0.7*(len(X.index))):]
y_train_SandP = y.iloc[0:round(0.7*(len(y.index)))]
y_test_SandP = y.iloc[round(0.7*(len(y.index))):]

training_error_SandP = []
test_error_SandP = []
# Creating Pipeline

#making numeric features
numeric_sub_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_sub_pipeline, X_train_SandP.columns)],
         remainder='passthrough')

# Creating Regressor Objects
Regressor_lr =  LinearRegression()
Regressor_knn = KNeighborsRegressor()
Regressor_SVR = SVR()
Regressor_RFR = RandomForestRegressor()
Regressor_ABR = AdaBoostRegressor()
Regressor_GBR = GradientBoostingRegressor()

wrapper_lr = Regressor_lr
wrapper_knn = Regressor_knn
wrapper_SVR = MultiOutputRegressor(Regressor_SVR)
wrapper_RFR = MultiOutputRegressor(Regressor_RFR)
wrapper_ABR = MultiOutputRegressor(Regressor_ABR)
wrapper_GBR = MultiOutputRegressor(Regressor_GBR)

# Creating parameter grids for Models

c_rs = np.logspace(3,-4,num = 20, endpoint = True)
p_rs= ["l1", "l2"]

param_grid_lr =  {'regressor__fit_intercept': [True,False]}
param_grid_knn = {'regressor__n_neighbors':[i for i in range(2,12)]}
param_grid_svr = {'regressor__estimator__C':[1,2,3,5,10,15],'regressor__estimator__kernel': ['poly','rbf','linear','sigmoid'],'regressor__estimator__gamma':['scale','auto']}
param_grid_rfr = {'regressor__estimator__n_estimators':[70,100,150,200,300,500,800,1000]}
param_grid_abr = {'regressor__estimator__n_estimators':[50,100,150,300,500,600,800,1000],'regressor__estimator__learning_rate':[0.2,0.1,0.01,0.05,0.001,0.0001],'regressor__estimator__loss': ['linear','square','exponential']}
param_grid_gbr = {'regressor__estimator__n_estimators':[50,100,150,300,500,600,800,1000],'regressor__estimator__learning_rate':[0.2,0.1,0.01,0.05,0.001,0.0001]}

#Creating Model List
models_list = {'Linear Regression': (wrapper_lr, param_grid_lr),
               'K Nearest Neighbours': (wrapper_knn, param_grid_knn),
               'Support Vector Machine': (wrapper_SVR, param_grid_svr),
               'Random Forest Regressor': (wrapper_RFR, param_grid_rfr),
               'AdaBoost Regressor': (wrapper_ABR, param_grid_abr),
               'Gradient Boosting Regressor': (wrapper_GBR, param_grid_gbr)}

#Creating a model comparison function
def train_and_score_model(model_name, pipeline, model_info):
    
    grid_search = RandomizedSearchCV(pipeline, model_info, cv=5, return_train_score=True)
    grid_search.fit(X_train_SandP, y_train_SandP)
    best_parameters = grid_search.best_params_

    pred_y_train = grid_search.predict(X_train_SandP)
    pred_y_test = grid_search.predict(X_test_SandP)
    
    score_test = r2_score(y_test_SandP,pred_y_test)
    score_train = r2_score(y_train_SandP,pred_y_train)
    MSE_Score_test = mse(y_test_SandP,pred_y_test)
    MSE_Score_train = mse(y_train_SandP,pred_y_train)
    
    training_error_SandP.append(MSE_Score_train)
    test_error_SandP.append(MSE_Score_test)
    

    return pred_y_test,pred_y_train,score_test,score_train,MSE_Score_test,MSE_Score_train,best_parameters,model_name

cols= ['Test MSE','Train MSE','test_score','train_score','Best Parameters','Model Name']
lst = []
j = 0
for model_name, model_info in models_list.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor',model_info[0])])
    pred_value_test,pred_value_train,s_test,s_train,MSE_test,MSE_train,best_parameters,model_name = train_and_score_model(model_name, pipeline, model_info[1])
    lst.append([MSE_test,MSE_train,s_test,s_train,best_parameters,model_name])
    plt.figure(figsize=[20,10])
    plt.subplot(2,3,(j+1))
    plt.title(model_name)
    plt.plot(pred_value_test, label = "predictions")
    plt.plot(y_test_SandP.to_numpy(), label = "actual")
    plt.legend()
    plt.tight_layout()
    j = j+1
plt.show()
data = pd.DataFrame(lst,columns = cols)
data = data.set_index('Model Name')


# In[35]:


plt.figure(figsize=[20,10])
x=[0,1,2,3,4,5]
values=['Linear_Regression', 'K_Nearest_Neighbours','Support_Vector_Machine','Random_Forest_Regressor','AdaBoost_Regressor','Gradient_Boosting_Regressor']
plt.title('S&P 500 Data')
plt.plot(training_error_SandP,  color='blue', label='train error')
plt.plot(test_error_SandP, color='green', label='test error')
plt.ylabel("Error")
plt.xlabel("Models")
plt.xticks(x,values)
plt.legend()
plt.show()


# In[36]:


data.to_csv('SandP_best_models.csv')


# In[37]:


data


# In[38]:


df_1 = pd.DataFrame(pred_value_test, columns=["AMGN_SandP","INTC_SandP"],index = X_test_SandP.index)


# In[39]:


MAE_Score_test = mae(y_test_SandP.iloc[:,1],pred_value_test[:,1])
MAE_Score_train = mae(y_train_SandP.iloc[:,1],pred_value_train[:,1])


# In[40]:


MAE_Score_test


# In[41]:


MAE_Score_train


# In[42]:


plt.figure(figsize=[20,10])
plt.plot(pred_value_test, label = "predictions")
plt.plot(y_test_SandP.to_numpy(), label = "actual")
plt.legend()
plt.show()


# In[43]:


X = df_Nasdaq.drop(['AMGN','INTC'], axis = 1)
y = df_Nasdaq[['AMGN','INTC']]

X_train_Nasdaq = X.iloc[0:round(0.7*(len(X.index)))]
X_test_Nasdaq = X.iloc[round(0.7*(len(X.index))):]
y_train_Nasdaq = y.iloc[0:round(0.7*(len(y.index)))]
y_test_Nasdaq = y.iloc[round(0.7*(len(y.index))):]

training_error_Nasdaq = []
test_error_Nasdaq = []

# Creating Pipeline

#making numeric features
numeric_sub_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_sub_pipeline, X_train_Nasdaq.columns)],
         remainder='passthrough')

# Creating Regressor Objects
Regressor_lr =  LinearRegression()
Regressor_knn = KNeighborsRegressor()
Regressor_SVR = SVR()
Regressor_RFR = RandomForestRegressor()
Regressor_ABR = AdaBoostRegressor()
Regressor_GBR = GradientBoostingRegressor()

wrapper_lr = Regressor_lr
wrapper_knn = Regressor_knn
wrapper_SVR = MultiOutputRegressor(Regressor_SVR)
wrapper_RFR = MultiOutputRegressor(Regressor_RFR)
wrapper_ABR = MultiOutputRegressor(Regressor_ABR)
wrapper_GBR = MultiOutputRegressor(Regressor_GBR)

# Creating parameter grids for Models

c_rs = np.logspace(3,-4,num = 20, endpoint = True)
p_rs= ["l1", "l2"]

param_grid_lr =  {'regressor__fit_intercept': [True,False]}
param_grid_knn = {'regressor__n_neighbors':[i for i in range(2,12)]}
param_grid_svr = {'regressor__estimator__C':[1,2,3,5,10,15],'regressor__estimator__kernel': ['poly','rbf','linear','sigmoid'],'regressor__estimator__gamma':['scale','auto']}
param_grid_rfr = {'regressor__estimator__n_estimators':[70,100,150,200,300,500,800,1000]}
param_grid_abr = {'regressor__estimator__n_estimators':[50,100,150,300,500,600,800,1000],'regressor__estimator__learning_rate':[0.2,0.1,0.01,0.05,0.001,0.0001],'regressor__estimator__loss': ['linear','square','exponential']}
param_grid_gbr = {'regressor__estimator__n_estimators':[50,100,150,300,500,600,800,1000],'regressor__estimator__learning_rate':[0.2,0.1,0.01,0.05,0.001,0.0001]}

#Creating Model List
models_list = {'Linear Regression': (wrapper_lr, param_grid_lr),
               'K Nearest Neighbours': (wrapper_knn, param_grid_knn),
               'Support Vector Machine': (wrapper_SVR, param_grid_svr),
               'Random Forest Regressor': (wrapper_RFR, param_grid_rfr),
               'AdaBoost Regressor': (wrapper_ABR, param_grid_abr),
               'Gradient Boosting Regressor': (wrapper_GBR, param_grid_gbr)}

#Creating a model comparison function
def train_and_score_model(model_name, pipeline, model_info):
    
    grid_search = RandomizedSearchCV(pipeline, model_info, cv=5, return_train_score=True)
    grid_search.fit(X_train_Nasdaq, y_train_Nasdaq)
    best_parameters = grid_search.best_params_

    pred_y_train = grid_search.predict(X_train_Nasdaq)
    pred_y_test = grid_search.predict(X_test_Nasdaq)
    
    score_test = r2_score(y_test_Nasdaq,pred_y_test)
    score_train = r2_score(y_train_Nasdaq,pred_y_train)
    MSE_Score_test = mse(y_test_Nasdaq,pred_y_test)
    MSE_Score_train = mse(y_train_Nasdaq,pred_y_train)
    
    training_error_Nasdaq.append(MSE_Score_train)
    test_error_Nasdaq.append(MSE_Score_test)

    return pred_y_test,pred_y_train,score_test,score_train,MSE_Score_test,MSE_Score_train,best_parameters,model_name

cols= ['Test MSE','Train MSE','test_score','train_score','Best Parameters','Model Name']
lst = []
j = 0
for model_name, model_info in models_list.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor',model_info[0])])
    pred_value_test,pred_value_train,s_test,s_train,MSE_test,MSE_train,best_parameters,model_name = train_and_score_model(model_name, pipeline, model_info[1])
    lst.append([MSE_test,MSE_train,s_test,s_train,best_parameters,model_name])
    plt.figure(figsize=[20,10])
    plt.subplot(2,3,(j+1))
    plt.title(model_name)
    plt.plot(pred_value_test, label = "predictions")
    plt.plot(y_test_Nasdaq.to_numpy(), label = "actual")
    plt.legend()
    plt.tight_layout()
    j = j+1
plt.show()
data = pd.DataFrame(lst,columns = cols)
data = data.set_index('Model Name')


# In[44]:


plt.figure(figsize=[20,10])
x=[0,1,2,3,4,5]
values=['Linear_Regression', 'K_Nearest_Neighbours','Support_Vector_Machine','Random_Forest_Regressor','AdaBoost_Regressor','Gradient_Boosting_Regressor']
plt.title('NASDAQ')
plt.plot(training_error_Nasdaq,  color='blue', label='train error')
plt.plot(test_error_Nasdaq, color='green', label='test error')
plt.ylabel("Error")
plt.xlabel("Models")
plt.xticks(x,values)
plt.legend()
plt.show()


# In[45]:


data.to_csv('Nasdaq_best_models.csv')


# In[46]:


data


# In[47]:


df_2 = pd.DataFrame(pred_value_test, columns=["AMGN_Nasdaq","INTC_Nasdaq"],index = X_test_Nasdaq.index)


# In[48]:


X = df_DJIA.drop(['AMGN','INTC'], axis = 1)
y = df_DJIA[['AMGN','INTC']]

X_train_DJIA = X.iloc[0:round(0.7*(len(X.index)))]
X_test_DJIA = X.iloc[round(0.7*(len(X.index))):]
y_train_DJIA = y.iloc[0:round(0.7*(len(y.index)))]
y_test_DJIA = y.iloc[round(0.7*(len(y.index))):]

training_error_DJIA = []
test_error_DJIA = []

# Creating Pipeline

#making numeric features
numeric_sub_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_sub_pipeline, X_train_DJIA.columns)],
         remainder='passthrough')

# Creating Regressor Objects
Regressor_lr =  LinearRegression()
Regressor_knn = KNeighborsRegressor()
Regressor_SVR = SVR()
Regressor_RFR = RandomForestRegressor()
Regressor_ABR = AdaBoostRegressor()
Regressor_GBR = GradientBoostingRegressor()

wrapper_lr = Regressor_lr
wrapper_knn = Regressor_knn
wrapper_SVR = MultiOutputRegressor(Regressor_SVR)
wrapper_RFR = MultiOutputRegressor(Regressor_RFR)
wrapper_ABR = MultiOutputRegressor(Regressor_ABR)
wrapper_GBR = MultiOutputRegressor(Regressor_GBR)

# Creating parameter grids for Models

c_rs = np.logspace(3,-4,num = 20, endpoint = True)
p_rs= ["l1", "l2"]

param_grid_lr =  {'regressor__fit_intercept': [True,False]}
param_grid_knn = {'regressor__n_neighbors':[i for i in range(2,12)]}
param_grid_svr = {'regressor__estimator__C':[1,2,3,5,10,15],'regressor__estimator__kernel': ['poly','rbf','linear','sigmoid'],'regressor__estimator__gamma':['scale','auto']}
param_grid_rfr = {'regressor__estimator__n_estimators':[70,100,150,200,300,500,800,1000]}
param_grid_abr = {'regressor__estimator__n_estimators':[50,100,150,300,500,600,800,1000],'regressor__estimator__learning_rate':[0.2,0.1,0.01,0.05,0.001,0.0001],'regressor__estimator__loss': ['linear','square','exponential']}
param_grid_gbr = {'regressor__estimator__n_estimators':[50,100,150,300,500,600,800,1000],'regressor__estimator__learning_rate':[0.2,0.1,0.01,0.05,0.001,0.0001]}

#Creating Model List
models_list = {'Linear Regression': (wrapper_lr, param_grid_lr),
               'K Nearest Neighbours': (wrapper_knn, param_grid_knn),
               'Support Vector Machine': (wrapper_SVR, param_grid_svr),
               'Random Forest Regressor': (wrapper_RFR, param_grid_rfr),
               'AdaBoost Regressor': (wrapper_ABR, param_grid_abr),
               'Gradient Boosting Regressor': (wrapper_GBR, param_grid_gbr)}

#Creating a model comparison function
def train_and_score_model(model_name, pipeline, model_info):
    
    grid_search = RandomizedSearchCV(pipeline, model_info, cv=5, return_train_score=True)
    grid_search.fit(X_train_DJIA , y_train_DJIA)
    best_parameters = grid_search.best_params_

    pred_y_train = grid_search.predict(X_train_DJIA)
    pred_y_test = grid_search.predict(X_test_DJIA)
    
    score_test = r2_score(y_test_DJIA,pred_y_test)
    score_train = r2_score(y_train_DJIA,pred_y_train)
    MSE_Score_test = mse(y_test_DJIA,pred_y_test)
    MSE_Score_train = mse(y_train_DJIA,pred_y_train)

    training_error_DJIA.append(MSE_Score_train)
    test_error_DJIA.append(MSE_Score_test)
    
    return pred_y_test,pred_y_train,score_test,score_train,MSE_Score_test,MSE_Score_train,best_parameters,model_name

cols= ['Test MSE','Train MSE','test_score','train_score','Best Parameters','Model Name']
lst = []
j = 0
for model_name, model_info in models_list.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor',model_info[0])])
    pred_value_test,pred_value_train,s_test,s_train,MSE_test,MSE_train,best_parameters,model_name = train_and_score_model(model_name, pipeline, model_info[1])
    lst.append([MSE_test,MSE_train,s_test,s_train,best_parameters,model_name])
    plt.figure(figsize=[20,10])
    plt.subplot(2,3,(j+1))
    plt.title(model_name)
    plt.plot(pred_value_test, label = "predictions")
    plt.plot(y_test_DJIA.to_numpy(), label = "actual")
    plt.legend()
    plt.tight_layout()
    j = j+1
plt.show()
data = pd.DataFrame(lst,columns = cols)
data = data.set_index('Model Name')


# In[49]:


plt.figure(figsize=[20,10])
x=[0,1,2,3,4,5]
values=['Linear_Regression', 'K_Nearest_Neighbours','Support_Vector_Machine','Random_Forest_Regressor','AdaBoost_Regressor','Gradient_Boosting_Regressor']
plt.title('DJIA')
plt.plot(training_error_DJIA,  color='blue', label='train error')
plt.plot(test_error_DJIA, color='green', label='test error')
plt.ylabel("Error")
plt.xlabel("Models")
plt.xticks(x,values)
plt.legend()
plt.show()


# In[50]:


MAE_Score_test = mae(y_test_DJIA.iloc[:,1],pred_value_test[:,1])
MAE_Score_train = mae(y_train_DJIA.iloc[:,1],pred_value_train[:,1])


# In[51]:


MAE_Score_train


# In[52]:


MAE_Score_test


# In[53]:


data.to_csv('DJIA_best_models.csv')


# In[54]:


data


# In[55]:


df_3 = pd.DataFrame(pred_value_test, columns=["AMGN_DJIA","INTC_DJIA"],index = X_test_DJIA.index)


# In[57]:


price = pd.concat([df_1,df_2,df_3],axis = 1)


# In[58]:


factor = price.stack()


# In[60]:


alpha_data = get_clean_factor_and_forward_returns(factor = factor,
                                                    prices = price,
                                                    quantiles=5,
                                                    periods=(1, 5, 15, 30, 60),
                                                    max_loss = 60)


# In[61]:


alpha_data.info()


# In[62]:


create_summary_tear_sheet(alpha_data)


# In[63]:


alpha_data.head(15)


# In[64]:


from alphalens.tears import create_full_tear_sheet
alphalens.tears.create_full_tear_sheet(alpha_data, long_short=True, group_neutral=False, by_group=False)


# In[ ]:




