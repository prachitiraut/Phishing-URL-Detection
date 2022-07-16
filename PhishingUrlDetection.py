#!/usr/bin/env python
# coding: utf-8

# # Phishing URL Detection

# #### Importing necessary libraries

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import metrics


# In[19]:


import warnings
warnings.filterwarnings("ignore")


# #### Uploading urldata.csv file  

# In[20]:


df = pd.read_csv('urldata.csv', encoding="latin-1")
df.head()


# #### Shows total columns in dataseet

# In[21]:


df.columns


# #### Find shape of the dataset

# In[22]:


df.shape


# #### Tells the datatype of each column present in the dataset

# In[23]:


df.dtypes


# In[24]:


df["label"].value_counts()


# ### Data visualization

# In[25]:


df['label'].value_counts()


# In[26]:


labels ='benign', 'malicious'
sizes = [345738,104438]
colors=['lightskyblue','yellow']
fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels, autopct='%1.1f%%',colors=colors,
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# #### Splitting data into dependent and independent variables

# In[27]:


X = df['url']
y = df['label']


# ####  Extract Feature With CountVectorizer 

# In[28]:


cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data


# #### Spliting data into training and testing

# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[30]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('\nLogistic Regression')
print("------------------------------------------------------------------------")
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print("------------------------------------------------------------------------")
print('Confusion Matrix: \n',metrics.confusion_matrix(y_test,y_pred), sep = '\n')
print("Classification report - \n", classification_report(y_test,y_pred))


# In[31]:


from sklearn.svm import LinearSVC
SVM = LinearSVC()
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)

print('\nSupport Vector Machine')
print("------------------------------------------------------------------------")
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print("------------------------------------------------------------------------")
print('Confusion Matrix: \n',metrics.confusion_matrix(y_test,y_pred), sep = '\n')
print("Classification report - \n", classification_report(y_test,y_pred))


# In[32]:


#Accuracy using Naive Bayes Model

NB = MultinomialNB()
NB.fit(X_train, y_train)
y_pred = NB.predict(X_test)

print('\nNaive Bayes')
print("------------------------------------------------------------------------")
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print("------------------------------------------------------------------------")
print('Confusion Matrix: \n',metrics.confusion_matrix(y_test,y_pred), sep = '\n')
print("Classification report - \n", classification_report(y_test,y_pred))

