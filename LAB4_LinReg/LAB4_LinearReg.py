#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary.csv')
X = dataset.iloc[:, :-1].values #get a copy of dataset exclude last column
y = dataset.iloc[:, 1].values #get array of dataset in column 1st


# In[35]:


X


# In[36]:


y


# In[37]:


count(X)


# In[38]:


X.shape[0]


# In[39]:


y.shape[0]


# In[46]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# In[47]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[48]:


viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='green')
viz_test.title('Area VS Selling Price (training set)')
viz_test.xlabel('Area (in hectare)')
viz_test.ylabel('Selling Price')
viz_train.show()

# Visualizing the Test set results
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='green')
viz_test.title('Area VS Selling Price (Test set)')
viz_test.xlabel('Area (in hectare)')
viz_test.ylabel('Selling Price')
viz_test.show()


# In[ ]:





# In[ ]:





# In[ ]:




