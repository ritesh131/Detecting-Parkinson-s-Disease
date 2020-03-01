#!/usr/bin/env python
# coding: utf-8

# #                   Detecting Parkinson’s Disease with XGBoost Classfier

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from sklearn.preprocessing import MinMaxScaler              # It's use for feature scaling
from xgboost import XGBClassifier                           # It's use to buil classify model
from sklearn.model_selection import train_test_split        # It's use to split data set in two part
from sklearn.metrics import accuracy_score


# In[2]:


#DataFlair - Read the data
df=pd.read_csv('parkinsons.data')
df.head()


# In[3]:


# check all entity is any nul in data information
df.info()


# In[4]:


# Check the short description(mean, median) in data set
df.describe()


# ### Visualize the for more information

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df.hist(bins=10, grid=False, figsize=(20,15), color='#8A2BE2', zorder=4)
plt.show()


# ### Split Data in features and labels

# In[7]:


features = df.loc[:,df.columns!='status'].values[:,1:]
labels = df.loc[:,'status'].values


# In[8]:


#DataFlair - Get the count of each label (0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])


# ### # Showing Correlation in data

# In[9]:


corr_mattrix = df.corr()


# In[10]:


corr_mattrix['status'].sort_values(ascending=True)


# ### Featue Scaling =>
#                          Initialize a MinMaxScaler and scale the features to between -1 and 1 to normalize them. The MinMaxScaler transforms features by scaling them to a given range. The fit_transform() method fits to the data and then transforms it. We don’t need to scale the labels.
#                         
#                         # Formula -: (value - min)/(max-min) , sklearn provides calss name MinMaxScaler

# In[11]:


scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels


# #### Split the data using sklearn class

# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


# ###### Using  XGBClassifier and train the model for predict output

# In[13]:


#DataFlair - Train the model
model=XGBClassifier()
model.fit(x_train,y_train)


# # Finally Model is Ready to Predation

# In[14]:


y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)


# In[15]:


print(f'Real Values of test label is {y_test} \n\n\n Predicted Values of test label is {y_pred}')


# ## Evaluate the model

# In[16]:


from sklearn.metrics import mean_squared_error


# In[17]:


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error in model is {mse}')


# In[18]:


from sklearn.metrics import confusion_matrix


# In[20]:


confusionmtrix = confusion_matrix(y_test, y_pred)
confusionmtrix

