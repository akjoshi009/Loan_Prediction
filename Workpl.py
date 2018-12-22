# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:54:05 2018

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification



data=pd.read_csv("D://24projects//Project 2//train_u6lujuX_CVtuZ9i.csv")
datatest=pd.read_csv("D://24projects//Project 2//test_Y3wMUE5_7gLdaTN.csv")
#filling emty or nan with appropriate value
data.isnull().sum()
data.dtypes
data.Gender.value_counts()

data.Gender.value_counts()
data.Gender.fillna("Male",inplace=True)
gender = {'Male': 1,'Female': 2} 
data.Gender = [gender[item] for item in data.Gender] 

data.Married.value_counts()
data.Married.fillna("Yes",inplace=True)
gender = {'Yes': 1,'No': 2} 
data.Married = [gender[item] for item in data.Married] 

data.Dependents.value_counts()
data.Dependents.fillna("0",inplace=True)
gender = {'0': 0,'1': 1,'2': 2,'3+': 4} 
data.Dependents = [gender[item] for item in data.Dependents] 

data.Self_Employed.value_counts()
data.Self_Employed.fillna("No",inplace=True)
gender = {'Yes': 1,'No': 2} 
data.Self_Employed = [gender[item] for item in data.Self_Employed] 

data.LoanAmount.value_counts()
a=round(data.LoanAmount.median())
data.LoanAmount.fillna(a,inplace=True)

data.Loan_Amount_Term.value_counts()
data.Loan_Amount_Term.fillna(360.0,inplace=True)


data.Credit_History.value_counts()
data.Self_Employed.fillna(1.0,inplace=True)

data.Education.value_counts()
gender = {'Graduate': 1,'Not Graduate': 2} 
data.Education = [gender[item] for item in data.Education]

data.Property_Area.value_counts()
gender = {'Semiurban': 1,'Urban': 2,'Rural':3} 
data.Property_Area = [gender[item] for item in data.Property_Area]


data.Loan_Status.value_counts()
gender = {'Y': 1,'N': 0} 
data.Loan_Status = [gender[item] for item in data.Loan_Status]

data.Credit_History.value_counts()
data.Credit_History.fillna(1.0,inplace=True)


#data[data.columns[0:12]]
X_train, X_test, y_train, y_test = train_test_split(data[data.columns[1:5]],data['Loan_Status'], test_size=0.20, random_state=42)

#using Logistic Regression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
pred=clf.predict(X_test)
accuracy_score(y_test, pred)

X2 = sm.add_constant(data[data.columns[1:5]])
est = sm.OLS(data['Loan_Status'], X2)
est2 = est.fit()
print(est2.summary())


data = data.drop('Gender', 1)
data = data.drop('Dependents', 1)
data= data.drop('Self_Employed', 1)
data=data.drop('ApplicantIncome',1)
data=data.drop('Education',1)
data=data.drop('Loan_Amount_Term',1)
data=data.drop('LoanAmount',1)


#using Random Forest

clf = RandomForestClassifier(n_estimators=1000, max_depth=5,
                             random_state=2)
clf.fit(X_train, y_train)
pred=clf.predict(X_test)
accuracy_score(y_test, pred)


#Data Cleaning for Test Data


datatest = datatest.drop('Gender', 1)
datatest = datatest.drop('Dependents', 1)
datatest= datatest.drop('Self_Employed', 1)
datatest=datatest.drop('ApplicantIncome',1)
datatest=datatest.drop('Education',1)
datatest=datatest.drop('Loan_Amount_Term',1)
datatest=datatest.drop('LoanAmount',1)






datatest.Married.value_counts()
datatest.Married.fillna("Yes",inplace=True)
gender = {'Yes': 1,'No': 2} 
datatest.Married = [gender[item] for item in datatest.Married] 


datatest.LoanAmount.value_counts()
a=round(datatest.LoanAmount.median())
datatest.LoanAmount.fillna(a,inplace=True)

datatest.Credit_History.value_counts()
datatest.Credit_History.fillna(1.0,inplace=True)

datatest.Property_Area.value_counts()
gender = {'Semiurban': 1,'Urban': 2,'Rural':3} 
datatest.Property_Area = [gender[item] for item in datatest.Property_Area]

datatest.Credit_History.value_counts()
datatest.Credit_History.fillna(1.0,inplace=True)


pred=clf.predict(datatest[datatest.columns[1:5]])
#accuracy_score(y_test, pred)

print(pred)

datatest["Loan_Status"]=pred
nwdf=datatest[['Loan_ID','Loan_Status']]
nwdf.head()
gender = {1:'Y',0:'N'} 
nwdf.Loan_Status = [gender[item] for item in nwdf.Loan_Status]


nwdf.to_csv("D://24projects//Project 2//output.csv", encoding='utf-8', index=False)


