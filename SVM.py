#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA


# In[4]:


df=pd.read_csv('processed.cleveland.data', header=None)


# In[5]:


df.head()


# In[6]:


df.columns=['age',
           'sex',
           'cp',
           'restbp',
           'chol',
            'fbs',
            'restecg',
            'thalach',
            'exang',
            'oldpeak',
            'slope',
            'ca',
            'thal',
            'hd'            
        ]


# In[7]:


df.head()


# In[9]:


df.dtypes


# In[12]:


df['ca'].unique()


# In[13]:


df['thal'].unique()


# In[14]:


len(df.loc[(df['ca']=='?') | (df['thal']=='?')])


# In[16]:


df.loc[(df['ca']=='?') | (df['thal']=='?')]


# In[8]:


df.isnull()


# In[17]:


len(df)


# In[20]:


df_no_missing=df.loc[(df['ca'] !='?') & (df['thal'] !='?')]


# In[21]:


len(df_no_missing)


# In[11]:


df.describe()


# In[22]:


df_no_missing['ca'].unique()


# In[23]:


df_no_missing['thal'].unique()


# In[24]:


X=df_no_missing.drop('hd', axis=1).copy()
X.head()


# In[25]:


y=df_no_missing['hd'].copy()
y.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




