#Step 1: Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
%matplotlib inline

#Step 2: Read ad Analyse the dataset
df = pd.read_csv('churn.csv')

#Dimension of the dataset
df.shape

#Info on the dataset
df.info()

#Step 3 : Data Visualization
#Plotting the target variable
df.Exited.value_counts().plot(kind='bar')
plt.xticks(ticks=[0,1],labels=['Stayed','Exited'])

#Plotting Churn vs Categorical variables
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
sns.countplot(x='Geography', hue = 'Exited',data = df, ax=axes[0][0])
sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axes[0][1])
sns.countplot(x='HasCrCard', hue = 'Exited',data = df, ax=axes[1][0])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axes[1][1])

#Plotting Churn vs Numerical Varibles
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
sns.boxplot(y=df['CreditScore'], x=df['Exited'], hue=df['Exited'], ax=axes[0][0])
sns.boxplot(y=df['Age'], x=df['Exited'], hue=df['Exited'], ax=axes[0][1])
sns.boxplot(y=df['Tenure'], x=df['Exited'], hue=df['Exited'], ax=axes[1][0])
sns.boxplot(y=df['Balance'], x=df['Exited'], hue=df['Exited'], ax=axes[1][1])
sns.boxplot(y=df['NumOfProducts'], x=df['Exited'], hue=df['Exited'], ax=axes[2][0])
sns.boxplot(y=df['EstimatedSalary'], x=df['Exited'], hue=df['Exited'], ax=axes[2][1])

#Step 4: Data Manipulation
#Continous variables
df['CreditScore'] = df['CreditScore'].astype('float')
df['Age'] = df['Age'].astype('float')
df['Tenure'] = df['Tenure'].astype('float')
df['Balance'] = df['Balance'].astype('float')
df['NumOfProducts'] = df['NumOfProducts'].astype('float')
df['EstimatedSalary'] = df['EstimatedSalary'].astype('float')

#Categorical Variables
df['HasCrCard'] = df['HasCrCard'].astype('category')
df['IsActiveMember'] = df['IsActiveMember'].astype('category')
df['Exited'] = df['Exited'].astype('category')

#STep 5: Feature Creation
df['CSperSal'] = df['CreditScore']/df['EstimatedSalary']

#Step 6: Data Wrangling 
df = df.drop(['Surname','CustomerId'],axis=1)
df.head(10)

#Step 7:Normalization of Continous Variables
num_cols = df.select_dtypes('float')
df_scaled = df.copy() 

for col in num_cols: 
    df_scaled[col] = (df_scaled[col] - df_scaled[col].mean()) / df_scaled[col].std()   

#Step 8 : Label Encoding
df.Gender = df.Gender.replace({'Male':1,'Female':0}) 
df.Gender = df.Gender.astype('category')
cont = pd.get_dummies(df['Geography'])
df_scaled = df_scaled.drop('Geography',axis=1)
df_scaled = pd.concat([cont,df_scaled],axis=1)

#Step 9: Selecting the features and the target variable
X = df_scaled.drop('Exited',axis=1)
Y = df_scaled['Exited']

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

#Step 10: Modelling and Evaluation
clf_entropy = DecisionTreeClassifier(criterion='entropy',random_state=0)

clf_entropy.fit(X_train,y_train)

y_pred = clf_entropy.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('Accuracy of the model: ',acc*100)

