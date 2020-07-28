#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


wine = pd.read_csv('/cxldata/datasets/project/wine_quality_red.csv') # load dataset


# In[4]:


wine.head() # view first 5 rows of dataset


# In[5]:


wine.info() # get all info about dataset i.e total rows,columns and missing values 


# In[6]:


bins = (2, 6.5, 8)  # Classify wine into good or bad
group_names = ['bad', 'good']  
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names) 


# In[7]:


label_quality = LabelEncoder()


# In[8]:


wine['quality'].value_counts() # Check total good and bad wined in dataset


# In[9]:


X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[16]:


sc = StandardScaler()


# In[17]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[18]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))


# In[19]:


print(confusion_matrix(y_test, pred_rfc))


# ---Wine has 88% accuracy with Random Forest Classifier---

# In[ ]:




