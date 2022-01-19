#!/usr/bin/env python
# coding: utf-8

# In[71]:


import csv
import pandas as pd
import numpy as np
import os
import datetime
from datetime import date
import dateutil
from time import gmtime, strftime
import time
import math
pd.set_option('display.max_columns', 100)
pd.options.display.max_colwidth = 100
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pandas import read_csv, datetime, DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
pd.options.display.float_format = '{:.2f}'.format
import itertools


# In[2]:


f = pd.read_csv("movie_metadata.txt")


# In[3]:


# Deselecting columns that are neutral and donot contribute to the model building
df = f[f.columns.difference(['director_name', 'actor_2_name','genres','actor_1_name','movie_title','actor_3_name',                             'plot_keywords','movie_imdb_link','title_year','language','country'])]

# Removing rows with null values of 1/3rd or more than the number of columns 
master = df[df.isnull().sum(axis=1) < 5].reset_index(drop = True)


# In[4]:


# Fill Null values with median value for the numerical attributes and mode for the categorical attribiutes
master = master.fillna(master.median())
master = master.fillna(master.mode().iloc[0])
train_data = master


# ### Data Exploration

# In[87]:


master['imdb_buc'] = pd.cut(master['imdb_score'], bins=[0,5,6,7,8,9,10],
    labels=['[0,5]','[5-6]','[6-7]','[7-8]','[8-9]','[9-10]'])

master['content_rating'].value_counts()


# ##### R and PG-13 rated movies are the highest of all ratings. we'll look at the imdb score buckets with these two.

# In[92]:


cr = master.groupby(['content_rating','imdb_buc'])['imdb_score'].count().reset_index().sort_values('imdb_buc', ascending = False)
rcr = cr[cr['content_rating'] == 'R']
pg13cr = cr[cr['content_rating'] == 'PG-13']

def plots(df,column,titlee, clr = 'green'):
    fig, ax = plt.subplots()
    ax.barh(df[column], df['imdb_score'], 0.75, color = clr)
    for i, v in enumerate(df['imdb_score']):
        ax.text(v + 3, i + .25, str(v), 
                color = 'blue', fontweight = 'bold')
    plt.title(titlee)
    plt.xlabel('Count')
    plt.ylabel('IMDB Score buckets') 
    plt.show()

plots(rcr,'imdb_buc','R-Rated Movies')
plots(pg13cr,'imdb_buc','PG-13 Rated Movies')


# In[96]:


# num_user_for_reviews has been the most correlated attribute for the imdb score
plt.scatter(master['num_user_for_reviews'][:100],master['imdb_score'][:100])
plt.xlabel('num_user_for_reviews')
plt.ylabel('imdb_score')
plt.title('Scatter plot on number of users vs imdb score')


# ### Encoding and Scaling

# In[5]:


from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce


# In[7]:


# Converting the categorical variables to numerical using target encoding
tenc=ce.TargetEncoder() 
df_tenc = tenc.fit_transform(train_data[['color','content_rating']],train_data['imdb_score'])
train_data = df_tenc.join(train_data.drop(['color','content_rating'],axis = 1))


# In[8]:


corr = []
for col in train_data.columns:
    corr.append([col,train_data[col].corr(train_data['imdb_score'])])

# dropping attributes with correlation less than 0.1 
corr = pd.DataFrame(corr, columns = ['Attribute', 'correlation with IMDB Score'])
selected = corr[corr['correlation with IMDB Score'] > 0.09].sort_values('correlation with IMDB Score', ascending = False).reset_index(drop=True)[1:]
selected


# In[ ]:


TargetVariable=['imdb_score']
Predictors=list(selected['Attribute'])
 
X=train_data[Predictors].values
y=train_data[TargetVariable].values


# In[10]:


# Sandardization of data
from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()
 
# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y)
 
# Generating the standardized values of X and y
X=PredictorScalerFit.transform(X)
y=TargetVarScalerFit.transform(y) 


# split into train test sets

# In[11]:


# split into train test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# ### Train

# #### Random forest Regressor 

# In[21]:


# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor
 
# create regressor object
regressor = RandomForestRegressor(n_estimators = 500, random_state = 42)
 
# fit the regressor with x and y data
regressor.fit(X_train, y_train)


# In[22]:


# Generating Predictions on testing data
Predictions=regressor.predict(X_test)
 
# Scaling the predicted data back to original scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)
 
# Scaling the y_test data back to original scale
y_test_orig=TargetVarScalerFit.inverse_transform(y_test)
 
# Scaling the test data back to original scale
Test_Data=PredictorScalerFit.inverse_transform(X_test)


# In[43]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("Mean Absolute Error",mean_absolute_error(y_test_orig,Predictions))
print("Mean Squared Error",mean_squared_error(y_test_orig,Predictions))

# Scaling the test data back to original scale
Test_Data=PredictorScalerFit.inverse_transform(X_test)
TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
TestingData['imdb_score']=y_test_orig

TestingData['RF_Predicted_imdb_score']=Predictions

APE=100*(abs(TestingData['imdb_score']-TestingData['RF_Predicted_imdb_score'])/TestingData['imdb_score'])
TestingData['APE']=APE
 
print('The Accuracy of Random forest model is:', 100-np.mean(APE))


# In[44]:


# Feature Importances
pd.DataFrame({'attribute':Predictors,'importance':regressor.feature_importances_}).sort_values('importance',ascending = False)


# #### Support Vector Regression

# In[29]:


from sklearn.svm import SVR, LinearSVR
reg = SVR(kernel="rbf")
reg.fit(X_train, y_train)


# In[34]:


# Generating Predictions on testing data
Preds=reg.predict(X_test)
 
# Scaling the predicted data back to original scale
Preds=TargetVarScalerFit.inverse_transform(Preds)
 
# Scaling the y_test data back to original scale
y_test_org=TargetVarScalerFit.inverse_transform(y_test)
 
# Scaling the test data back to original scale
Test_Dat=PredictorScalerFit.inverse_transform(X_test)


# In[36]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("MAE",mean_absolute_error(y_test_org,Preds))
print("MSE",mean_squared_error(y_test_org,Preds))

TestingData['SVR_Predicted_imdb_score']=Preds

APE_SVR=100*(abs(TestingData['imdb_score']-TestingData['SVR_Predicted_imdb_score'])/TestingData['imdb_score'])
TestingData['APE_SVR']=APE_SVR
 
print('The Accuracy of Support vector model is:', 100-np.mean(APE_SVR))


# #### Neural network regression

# In[12]:


from keras.models import Sequential
from keras.layers import Dense

# Defining a function to find the best parameters for ANN
def BestParams(X_train, y_train, X_test, y_test):
    
    # Defining the list of hyper parameters to try
    batch_size_list=[10, 15, 20]
    epoch_list  =   [5, 10, 50]
    
    import pandas as pd
    SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])
    
    # initializing the trials
    TrialNumber=0
    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber+=1
            # create ANN model
            model = Sequential()
            model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
            model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal'))
            # Compiling the model
            model.compile(loss='mean_squared_error', optimizer='adam')
            # Fitting the ANN to the Training set
            model.fit(X_train, y_train ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)
            
            MAPE = np.mean(100 * (np.abs(y_test-model.predict(X_test))/y_test))
            
            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:','batch_size:', batch_size_trial,'-', 'epochs:',epochs_trial, 'Accuracy:', 100-MAPE)
            
            SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), 100-MAPE]],
                                                                    columns=['TrialNumber', 'Parameters', 'Accuracy'] ))
    return(SearchResultsData)


ResultsData=BestParams(X_train, y_train, X_test, y_test)


# ###### Fitting the ANN to the Training set with the best parameteres obtained from the above

# In[13]:


model = Sequential()
model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train ,batch_size = 15 , epochs = 50, verbose=1)


# In[14]:


# Generating Predictions on testing data
Pred=model.predict(X_test)
 
# Scaling the predicted data back to original scale
Pred=TargetVarScalerFit.inverse_transform(Pred)
 
# Scaling the y_test data back to original scale
y_test_og=TargetVarScalerFit.inverse_transform(y_test)
 
# Scaling the test data back to original scale
TestData=PredictorScalerFit.inverse_transform(X_test)


# In[45]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("MSE",mean_squared_error(y_test_og,Pred))
print("MAE",mean_absolute_error(y_test_og,Pred))

TestingData['ANN_Predicted_imdb_score']=Pred
APE_ANN=100*(abs(TestingData['imdb_score']-TestingData['ANN_Predicted_imdb_score'])/TestingData['imdb_score'])
TestingData['APE_ANN']=APE_ANN
 
print('The Accuracy of Artificial Neural network model is:', 100-np.mean(APE_ANN))


# ## Conclusion

# #### Looking at the predictions, all the three models have performed considerably equal with an average accuracy of ~0.88 and MAE of 0.60. 
# 
# #### The number of parameters in the Artificial neural networks that need be trained are too huge and is higher compared to the other two algorithms. Same is case with Support vector regression where in the number of parameters and training time is considerably less compared to random forest regression. 
# 
# #### Hence in any case if the performance of multiple algorithms on a dataset are close to each other then model which would make more sense in this case would be the simplistic one. According to occams razor, simpler the model the better it is. 

# #### "Hence Random forest regressor would be ideal for deployment in this case." 
