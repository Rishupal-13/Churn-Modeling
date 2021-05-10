#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[5]:


df = pd.read_csv("Desktop\coding\ML\sush\Churn_Modelling.csv")
df.head(20)


# In[17]:


X = df.iloc[:, 3:13]
y = df.iloc[:, 13]


# In[18]:


geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)


# In[19]:


X=pd.concat([X,geography,gender],axis=1)
X=X.drop(['Geography','Gender'],axis=1)


# In[20]:


X.head(20)


# In[21]:


y.head()


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[24]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# In[ ]:




