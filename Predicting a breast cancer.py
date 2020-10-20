#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
import numpy as np


# In[123]:


dataset=pd.read_csv('C:/Users/DELL/Desktop/New folder/Python/data.csv')


# In[124]:


dataset.head(20)


# In[125]:


dataset.shape


# In[126]:


dataset.isnull().sum()


# In[127]:


dataset.info()


# In[128]:


dataset.describe()


# In[129]:


dataset['diagnosis']=np.where(dataset['diagnosis']=='M',1,0)


# In[130]:


dataset.head(20)


# In[131]:


dataset.drop(['id'], axis=1, inplace=True)


# In[132]:


dataset.head()


# In[133]:


dataset.shape


# In[134]:


dataset.drop(['Unnamed: 32'], axis=1, inplace=True)


# In[135]:


dataset.head()


# In[136]:


continues_feature=[feature for feature in dataset.columns]


# In[137]:


continues_feature


# In[138]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[139]:


for feature in continues_feature:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.title(feature)
    plt.show()


# In[140]:


##Create a pair plot


# In[141]:


#sns.pairplot(dataset.iloc[:,0:18], hue='diagnosis')


# In[142]:


##Get the corr


# In[143]:


dataset.corr()


# In[144]:


##Visualize the correlation


# In[145]:


plt.figure(figsize=(10,10))
sns.heatmap(dataset.iloc[:,0:12].corr(), annot=True, fmt='.0%')


# In[146]:


X=dataset.drop(['diagnosis'], axis=1)


# In[147]:


X.head()


# In[148]:


y=dataset.iloc[:,0]


# In[149]:


y.head()


# In[150]:


from sklearn.model_selection import train_test_split


# In[151]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)


# In[152]:


from sklearn.preprocessing import StandardScaler


# In[153]:


sc=StandardScaler()


# In[154]:


X_train=sc.fit_transform(X_train)


# In[155]:


X_test=sc.fit_transform(X_test)


# In[156]:


###Selecting model


# In[157]:


##Using a function


# In[158]:


def model(X_train,y_train):
    ##Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train,y_train)
    ##Decsion tree
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train,y_train)
    ##Random Forest Clasffier
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    forest.fit(X_train,y_train)
    print('[0]Logistic Regression Accuracy',log.score(X_train,y_train))
    print('[1]Desision Tree',tree.score(X_train,y_train))
    print('[2]Random Classifier Accuracy',forest.score(X_train,y_train))
    return log,tree,forest
    


# In[159]:


##Getting all te model
models=model(X_train,y_train)


# In[160]:


###Test the model accuracey


# In[161]:


from sklearn.metrics import confusion_matrix 

for i in range(len(models)):
    print('Model',i)

    cm = confusion_matrix(y_test, models[i].predict(X_test))
    print(cm)
 
    print('testing accuracy=',(65+44)/(65+44+3+3))
    print()


# In[164]:


from sklearn.metrics import classification_report


# In[165]:


from sklearn.metrics import accuracy_score


# In[166]:


print(classification_report(y_test, models[0].predict(X_test)))


# In[167]:


print(classification_report(y_test, models[1].predict(X_test)))


# In[168]:


print(classification_report(y_test, models[2].predict(X_test)))


# In[169]:


pred=models[2].predict(X_test)


# In[170]:


print(pred)


# In[173]:


print()


# In[176]:


print(y_test.head(12))


# In[ ]:




