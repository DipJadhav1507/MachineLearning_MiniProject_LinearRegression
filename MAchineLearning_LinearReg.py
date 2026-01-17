import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

## Creating the dataset and dataframe
df=pd.read_csv('HousePricePrediction.csv')
df1=df.copy()
## Cleaning the dataset
df=df.dropna()
df=df.drop(['Id','Exterior1st','LotConfig'],axis=1)
df=pd.get_dummies(df,columns=['MSZoning','BldgType'],dtype='int64')

x=df.drop('SalePrice',axis=1)
y=df['SalePrice']

## Feature engineering
x['YearRemodAdd']=2025-x['YearRemodAdd']
x['YearBuilt']=2025-x['YearBuilt']

x['LotArea']/=1000

x['is_renew']=x['YearBuilt']==x['YearRemodAdd']
x['is_renew'] = x['is_renew'].astype('int64')

x['is']=x['BsmtFinSF2']==0
x['is'] = x['is'].astype('int64')

x['TotalBsmtSF']/=100
x['Area']=x['TotalBsmtSF']*x['LotArea']

x['value']=x['TotalBsmtSF']*x['YearBuilt']

# Creating the train and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## Model Training and testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

## Model Testing
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print('r2 score is:',r2)
n=x_test.shape[0]
p=x_test.shape[1]
adj_r=1-((1-r2)*(n-1)/(n-p-1))
print('ajuted r2 score is:',adj_r)
         
