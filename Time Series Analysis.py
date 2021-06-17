#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np   
from sklearn.linear_model import LinearRegression
import pandas as pd    
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split # Sklearn package's randomized data splitting function, splits data into training and testing data


# In[2]:


data = pd.read_csv("Covid_data_Indian_states.csv")  
data.shape


# In[3]:


reqdata=data.drop(['new_cases','Date'],axis=1)


# In[4]:


reqdata.head()


# In[5]:


data.describe()


# In[6]:


reqdata.dtypes


# In[14]:


reqdata.dropna(inplace=True)


# In[15]:


np.any(np.isnan(reqdata))


# In[16]:


np.all(np.isfinite(reqdata))


# In[17]:


data_attr = data.iloc[:, 0:4] 
sns.pairplot(data, diag_kind='kde')  


# In[18]:


X = reqdata.drop(['Weekly_new_infections'], axis=1) #all variables except mpg
# the dependent variable
y = reqdata[['Weekly_new_infections']]


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[20]:


reg_model = LinearRegression()
reg_model.fit(X_train, y_train) 


# In[21]:


for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, reg_model.coef_[0][idx]))


# In[22]:


intercept = reg_model.intercept_[0]
print("The intercept for our model is {}".format(intercept))


# In[23]:


reg_model.score(X_train, y_train)


# In[24]:


reg_model.score(X_test, y_test)


# In[ ]:


#y=a+bx is the equation of the trendline with a=-194677.54086707125,b=17410.704848928614, where y= Weekly new infections, x=Weeks(Time paramter)

