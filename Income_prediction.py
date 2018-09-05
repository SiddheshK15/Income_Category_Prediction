# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:04:59 2018

@author: Siddhesh
"""
#%%
import pandas as pd
import numpy as np
#%%
adult_df= pd.read_csv(r'C:\Users\Siddhesh\adult_data.csv',header= None, delimiter=' *, *',engine='python')
adult_df.head()
#%%
adult_df.shape
#%%
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']
#%%
adult_df.head()
#%%
print(adult_df.isnull().sum())
#%%
for value in ['workclass','education','marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']:
    print(value, sum(adult_df[value] =='?'))

#%%
pd.set_option('display.max_columns',None)
#%%
#create a copy of the dataframe
adult_df_rev=pd.DataFrame.copy(adult_df)
temp=adult_df_rev.describe(include='all')
#%%
for value in ['workclass','occupation','native_country']:
    adult_df_rev[value].replace(['?'], adult_df_rev[value].mode()[0],inplace=True)
#%%
for value in ['workclass','education','marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']:
    print(value, sum(adult_df_rev[value] =='?'))
#%%
adult_df_rev.workclass.value_counts()
#%%
adult_df_rev.education.value_counts()
#%%
colname = ['workclass','education','marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']
colname
0
#%%
adult_df_rev.head()

#0--> <=50K
#1--> >50K
#%%
X= adult_df_rev.values[:,:-1]#all the rows all the cols but income
Y= adult_df_rev.values[:,-1]#all the rows and only income col
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X= scaler.transform(X)
print(X)
#%%
Y=Y.astype(int)#precationary not mandatory
type(Y)
#%%
from sklearn.model_selection import train_test_split

#split the data inti test and train

X_train, X_test,Y_train, Y_test =train_test_split(X,Y, test_size=0.3,random_state=10)
#%%
from sklearn.linear_model import LogisticRegression 
#crete a model
classifier=(LogisticRegression())
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred= classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))
#%%
temp2=list(zip(Y_test,Y_pred))

#%%
from sklearn.metrics import confusion_matrix, accuracy_score, \
classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)
#%%
print(Y_pred.shape)
#%%
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)
#%%
y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value < 0.6:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
#print(y_pred_class)
#%%
from sklearn.metrics import confusion_matrix, accuracy_score

cfm=confusion_matrix(Y_test.tolist(),y_pred_class)
print(cfm)



acc=accuracy_score(Y_test.tolist(), y_pred_class)
print("Accuracy of the model: ",acc)
print(classification_report(Y_test.tolist(),Y_pred))
#%%
for a in np.arange(0,1,0.05):
    predict_mine = np.where(y_pred_prob[:,0] < a, 1, 0)
    cfm = confusion_matrix(Y_test.tolist(),predict_mine)
    total_err= cfm[0,1]+cfm[1,0]
    print("Error at threshold", a, ":",total_err,"type 2 error:", cfm[1,0]," , type 1 error",cfm[0,1])
#%%
from sklearn import metrics
fpr, tpr, threshhold= metrics.roc_curve(Y_test.tolist(),y_pred_class)
auc=metrics.auc(fpr,tpr)
print(auc)
#%%
import matplotlib.pyplot as plt
plt.title('Reaceiver  Operating Characteristics')
plt.plot(fpr,tpr,'b',label =auc) 
plt.legend(loc = 'lower right') 
plt.plot([0, 1], [0 ,1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False  positive Rate')
plt.ylabel('True positive Rate')

plt.show()  