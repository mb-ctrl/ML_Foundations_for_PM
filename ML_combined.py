#!/usr/bin/env python
# coding: utf-8

# # Preparation

# In[147]:


import pandas as pd
import seaborn as sns
import numpy as np
from fast_ml.model_development import train_valid_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn import metrics


# ### Import Data

# In[148]:


df = pd.read_csv('C:/Users/Mauro/Desktop/Coursera_ML_Product_Manager/CCPP_data.csv')
df.rename(columns={"AT":"Temperatur"}, inplace=True)
df.rename(columns={"V":"Exhaust Vacuum"}, inplace=True)
df.rename(columns={"AP":"Ambient Pressure"}, inplace=True)
df.rename(columns={"RH":"Relative Humidity"}, inplace=True)
df.rename(columns={"PE":"PE"}, inplace=True)
df.head(3)


# In[149]:


data_ml = df [["Temperatur", "Exhaust Vacuum","PE"]]
data_ml.head(3)


# ### Splitting Dataset

# In[150]:


X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(data_ml, 
                                                                            target = 'PE', 
                                                                            train_size=0.6, 
                                                                            valid_size=0.2, 
                                                                            test_size=0.2)

for data in [X_train, y_train, X_valid, y_valid, X_test, y_test]:
    data.reset_index(drop=True, inplace=True)
    
X_train.head(3)


# ## Linear Regression Model

# ### Data Preparation

# In[151]:


X_train_01 = X_train['Temperatur']
X_train_01.head(3)


# In[152]:


y_train_01 = y_train.values.reshape(-1, 1)


# In[153]:


X_train_01 = X_train_01.values.reshape(-1, 1)


# ### Regression Model Training

# In[154]:


model = LinearRegression()


# In[155]:


model.fit(X_train_01, y_train_01)
print(model.coef_)


# ### Regression Model Validation

# In[156]:


X_valid_01 = X_valid['Temperatur']
X_valid_01.head(3)


# In[157]:


X_valid_01 = X_valid_01.values.reshape(-1, 1)


# In[158]:


predictions = model.predict(X_valid_01)


# ## Linear Regression Model Metrics

# In[159]:


y_valid_01 = y_valid


# ### MAE

# In[160]:


metrics.mean_absolute_error(y_valid_01, predictions)


# ### MSE
# 
# 

# In[178]:


mse_linear = metrics.mean_squared_error(y_valid_01, predictions)
print(mse_linear)


# ### R2

# In[174]:


r2_linear = model.score(X_valid_01, y_valid_01)
print(f"R^2 Score: {r2_linear}")


# ## Polynominal Regression

# ### Data Preparation

# In[163]:


X_train_02 = X_train['Exhaust Vacuum']
X_train_02.head(3)


# ### Polynominal Regression Training Model

# In[164]:


model_poly = np.poly1d(np.polyfit(X_train_02, y_train, 2))


# ### Polynominal Regression Validation 

# In[165]:


X_valid_02 = X_valid['Exhaust Vacuum']
X_valid_02.head(3)


# In[166]:


predictions_02 = model_poly(X_valid_02)
predictions_02


# ## Polynominal Regression Model Metrics

# In[167]:


y_valid_02 = y_valid


# ### MSE

# In[175]:


mse_polynominal = ((predictions_02 - y_valid_02) ** 2).mean()
print(f"Mean Squared Error: {mse_polynominal}")


# ### R2

# In[170]:


from sklearn.metrics import r2_score


# In[173]:


r2_polynominal = (r2_score(y_valid_02, predictions_02))
print(r2_polynominal)


# # Comparison

# In[179]:


comparetable = {
    'Model': ['Linear', 'Polynomial'],
    'MSE': [mse_linear, mse_polynominal],
    'R2': [r2_linear, r2_polynominal]
}

ct = pd.DataFrame(comparetable)


# In[180]:


print(ct)


# # Testing

# In[184]:


X_test_01 = X_test['Temperatur']
X_test_01.head(3)


# In[185]:


X_test_01 = X_test_01.values.reshape(-1, 1)


# In[186]:


predictions_test = model.predict(X_test_01)


# ### Testing Metrics

# In[187]:


mae_test = metrics.mean_absolute_error(y_test, predictions_test)


# In[193]:


mse_test


# In[197]:


mse_test = metrics.mean_squared_error(y_test, predictions_test)


# In[196]:


rmse_test = np.sqrt(metrics.mean_squared_error(y_test, predictions_test))


# In[194]:


r2_score_test = model.score(X_test_01, y_test)
print(f"R^2 Score: {r2_score}")


# In[ ]:


testing


# In[198]:


testing_result = {
    'Model': ['Linear'],
    'MAE': [mae_test],
    'MSE': [mse_test],
    'RMSE': [rmse_test],
    'R2': [r2_score_test]
}

tr = pd.DataFrame(testing_result)


# In[199]:


print(tr)

