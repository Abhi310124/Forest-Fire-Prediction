import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from warnings import filterwarnings
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
filterwarnings('ignore')
%matplotlib inline

df=pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv",header=1)

df.head()

df.tail()

df.shape

df.isnull().sum()

df.describe()

df.iloc[121:125,:]

#DATA CLEANING

df.drop([122,123],inplace=True)
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)
df.loc[:122,"Region"]=0
df.loc[122:,"Region"]=1
df.columns

df.columns=df.columns.str.strip()
df.columns

df.dropna(inplace=True)
df.dtypes

df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws',"Region"]]=df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws',"Region"]].astype(int)

df.dtypes

df.Classes.unique()
df.Classes=df.Classes.str.strip()
df.Classes.unique()

df.columns

df[['Rain', 'FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']]=df[['Rain', 'FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']].astype('float')
df.dtypes

df1=df.drop(['year'],axis=1)
df1.describe().T

set(df1.Classes)
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Classes'.
df1 ['Classes']= label_encoder.fit_transform(df1 ['Classes'])
df1.head()

set(df1.Classes)
df1.corr()

sns.pairplot(df1)

sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(df1.corr(),annot=True)

df1.hist(figsize=(20,14),color='r')
percentage=df.Classes.value_counts(normalize=True)*100
percentage

classes_labels=['Fire','Not Fire']
plt.figure(figsize=(15,10))
plt.pie(percentage,labels=classes_labels,autopct="%1.1f%%")
plt.title("Pie Chart of Classes",fontsize=15)
plt.show()

#Model Building Using Logistic Regression
df1

X = df1[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC',
       'ISI', 'BUI', 'FWI','Region']]
X

y=df1['Classes']
y

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=32,test_size=.33)
#Feature Scaling
def Feature_Scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train_after_Standardisation = scaler.fit_transform(X_train)
    X_test_after_Standardisation = scaler.transform(X_test)
    return X_train_after_Standardisation, X_test_after_Standardisation

X_train_after_Standardisation,X_test_after_Standardisation=Feature_Scaling(X_train, X_test)
logistic_regression=LogisticRegression()
logistic_regression.fit(X_train_after_Standardisation,y_train)
print('Intercept is :',logistic_regression.intercept_)
print('Coefficient is :',logistic_regression.coef_)

print("Training Score:",logistic_regression.score(X_train_after_Standardisation, y_train))
print("Test Score:",logistic_regression.score(X_test_after_Standardisation,y_test))

Logistic_Regression_Prediction=logistic_regression.predict(X_test_after_Standardisation)
accuracy_score(y_test,Logistic_Regression_Prediction)
Actual_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': Logistic_Regression_Prediction})
Actual_predicted['Report']=abs(Actual_predicted['Actual']-Actual_predicted['Predicted'])
Actual_predicted['Classes']= np.where(Actual_predicted['Report']== 0,'Matched','Unmatched')
Actual_predicted_group_df=Actual_predicted.groupby(['Classes']).agg({'Classes':['count']})
Actual_predicted_group_df.reset_index()

#Confusion Matrix

conf_mat = confusion_matrix(y_test,Logistic_Regression_Prediction)
conf_mat

#Plotting Confusion Matrix

conf_matrix = confusion_matrix(y_true=y_test, y_pred=Logistic_Regression_Prediction)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

#Splitting the Confusion Matrix

true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]

#ACCURACY
Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy

#Precision
Precision = true_positive/(true_positive+false_positive)
Precision

