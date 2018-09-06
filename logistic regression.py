# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 10:40:24 2018

@author: madhavi
"""
#logistic regression
#%%
import pandas as pd
import numpy as np
#%%
adult_df=pd.read_csv('adult_data.csv',header=None,delimiter=' *, *',engine='python')
adult_df.head()
#%%
adult_df.shape
#%%
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']
#%%
#adult_df.isnull().sum() #used only when NaN valuezi.e. integer
#%%
#all variables will get assigned to 0 value
for value in['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income']:
            print(value,sum(adult_df[value]=='?'))
#%%            
#copy of dataframe
adult_df_rev=pd.DataFrame.copy(adult_df)
adult_df_rev.describe(include='all')
#%%
for value in['workclass','occupation','native_country']:
    adult_df_rev[value].replace(['?'],adult_df_rev[value].mode()[0],inplace=True)
#%%
    for value in['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income']:
            print(value,",",sum(adult_df_rev[value]=='?'))
#%%
#convert ctegorical to numerical data
colname=['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income']  
colname
#%%          

from sklearn import preprocessing
le={}
for x in colname:
       le[x]=preprocessing.LabelEncoder()
for x in colname:
       adult_df_rev[x]=le[x].fit_transform(adult_df_rev.__getattr__(x))
#%%       
adult_df_rev.head()
# 0 -> <= 50k
# 1-> >50k
#%%   
x= adult_df_rev.values[:,:-1]
y= adult_df_rev.values[:,-1]
#%%    
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
print(x)
#%%
y=y.astype(int)
#%%
#building the model
#splitting into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
#%%
from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression())
#fitting training data to the model
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(list(zip(y_test,y_pred)))
#%%
#evaluating the model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test,y_pred)
print(cfm)
print("Classification report: ")
print(classification_report(y_test,y_pred))
acc=accuracy_score(y_test,y_pred)
print("Accuracy of the model: ",acc)
#%%
print(y_pred.shape)
#%%
#tuning the model
y_pred_prob=classifier.predict_proba(x_test)
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
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test.tolist(),y_pred_class)
print(cfm)
acc=accuracy_score(y_test.tolist(),y_pred_class)
print("Accuracy of the model: ",acc)
print(classification_report(y_test.tolist(),y_pred_class))
#%%         
y_pred_class=[]
for value in y_pred_prob[:,0]:
     if value < 0.5:
         y_pred_class.append(1)
     else:
         y_pred_class.append(0)
#print(y_pred_class)
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test.tolist(),y_pred_class)
print(cfm)
acc=accuracy_score(y_test.tolist(),y_pred_class)
print("Accuracy of the model: ",acc)
print(classification_report(y_test.tolist(),y_pred_class))
#%%
for a in np.arange(0,1,0.05):
       predict_mine=np.where(y_pred_prob[:,0]<a,1,0)
       cfm=confusion_matrix(y_test.tolist(),predict_mine)
       total_err=cfm[0,1]+cfm[1,0]
       print("Errors at threshold",a,":",total_err,"type 2 error:",cfm[1,0],"type 1 error:",cfm[0,1])
#%%       
from sklearn import metrics
#preds=classifier.predict_proba(x_test)[:,0]
fpr,tpr,threshold=metrics.roc_curve(y_test.tolist(),y_pred_class)
auc=metrics.auc(fpr,tpr)
print(auc)
#%%       
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristics')
plt.plot(fpr,tpr,'b',label=auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#%%
#performing cross validation
classifier=(LogisticRegression())

from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(x_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=x_train,
y=y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

for train_value, test_value in kfold_cv:
        classifier.fit(x_train[train_value], y_train[train_value]).predict(x_train[test_value])

y_pred=classifier.predict(x_test)
print(list(zip(y_test,y_pred)))
#%%
#feature selection using recursive feature elimination
x= adult_df_rev.values[:,:-1]
y= adult_df_rev.values[:,-1]
#print(x)
#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
print(x)
#%%
y=y.astype(int)
#%%
#building the model
#splitting into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
#%%
colname=adult_df_rev.columns[:]
#%%
#recursive feature elimia=nation
from sklearn.feature_selection import RFE
rfe = RFE(classifier, 7)
model_rfe = rfe.fit(x_train, y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ") 
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_)
#%%
y_pred=model_rfe.predict(x_test)
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test,y_pred)
print(cfm)
print("Classification report: ")
print(classification_report(y_test,y_pred))
acc=accuracy_score(y_test,y_pred)
print("Accuracy of the model: ",acc)
#%%
x= adult_df_rev.values[:,:-1]
y= adult_df_rev.values[:,-1]
#%%
#select k best
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


test = SelectKBest(score_func=chi2, k=10)
fit1 = test.fit(x, y)

print(fit1.scores_)
print(list(zip(colname,fit1.get_support())))
features = fit1.transform(x)

print(features)
#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(features)
x=scaler.transform(features)
print(features)
#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
#%%
from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression())
#fitting training data to the model
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(list(zip(y_test,y_pred)))
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test,y_pred)
print(cfm)
print("Classification report: ")
print(classification_report(y_test,y_pred))
acc=accuracy_score(y_test,y_pred)
print("Accuracy of the model: ",acc)
#%%





