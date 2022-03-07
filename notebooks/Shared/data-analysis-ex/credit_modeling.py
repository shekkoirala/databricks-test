# Databricks notebook source
import pandas as pd 
import numpy as np 
import snowflake.connector  

ctx = snowflake.connector.connect(
    account = 'dy06286.central-us.azure',
    user = 'Shekhar Koirala',
    password = 'NLP/}<u3A;Ld{Wtp',
    schema = 'PUBLIC',
    warehouse = 'Test_Warehouse',
    role = 'SYSADMIN'
    )

cur = ctx.cursor()

cur.execute('''Use Database TEST''')

query = '''select * from cred_model'''

cur.execute(query)

data = pd.DataFrame.from_records(iter(cur), columns = [x[0] for x in cur.description])
   

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Data Description 
# MAGIC 
# MAGIC 
# MAGIC ID: ID of each client
# MAGIC 
# MAGIC LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# MAGIC 
# MAGIC SEX: Gender (1=male, 2=female)
# MAGIC 
# MAGIC EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# MAGIC 
# MAGIC MARRIAGE: Marital status (1=married, 2=single, 3=others)
# MAGIC 
# MAGIC AGE: Age in years
# MAGIC 
# MAGIC PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
# MAGIC 
# MAGIC PAY_2: Repayment status in August, 2005 (scale same as above)
# MAGIC 
# MAGIC PAY_3: Repayment status in July, 2005 (scale same as above)
# MAGIC 
# MAGIC PAY_4: Repayment status in June, 2005 (scale same as above)
# MAGIC 
# MAGIC PAY_5: Repayment status in May, 2005 (scale same as above)
# MAGIC 
# MAGIC PAY_6: Repayment status in April, 2005 (scale same as above)
# MAGIC 
# MAGIC BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# MAGIC BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# MAGIC BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# MAGIC BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# MAGIC BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# MAGIC BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# MAGIC 
# MAGIC PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# MAGIC PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# MAGIC PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# MAGIC PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# MAGIC PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# MAGIC PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# MAGIC 
# MAGIC default.payment.next.month: Default payment (1=yes, 0=no)

# COMMAND ----------



# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import random 
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

data.rename(columns={'default payment next month':'def_pay'}, inplace=True) # changing the column name

# COMMAND ----------

data.shape

# COMMAND ----------

pd.value_counts(data['def_pay']).plot.bar()
plt.title("Credit Card Default Counts")

# COMMAND ----------

print(data['LIMIT_BAL'].value_counts().nlargest(5))
print('\nNANs found:', sum(data['LIMIT_BAL']==0) )

# COMMAND ----------

print ("Gender: ")
print(data['SEX'].value_counts()) 
print('NANs found: ', sum(data['SEX']==0))

# COMMAND ----------

print(data['EDUCATION'].value_counts())
print('NANs found:', sum(data['EDUCATION']==0) )

# COMMAND ----------

print(data['MARRIAGE'].value_counts())
print('NANs found:', sum(data['MARRIAGE']==0) )

# COMMAND ----------

data['AGE'] = data['AGE'].astype(int)

# COMMAND ----------

print(data['AGE'].value_counts().nlargest(10))
print('NANs found:', sum(data['AGE']==0)) 
plt.boxplot(data['AGE']) 
plt.title("Age Distribution")

# COMMAND ----------

print(data['PAY_0'].value_counts())
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(12,4))
sns.countplot(x="PAY_0", data=data, ax=ax[0,0])
sns.countplot(x="PAY_2", data=data, ax=ax[0,1])
sns.countplot(x="PAY_3", data=data, ax=ax[0,2])
sns.countplot(x="PAY_4", data=data, ax=ax[1,0])
sns.countplot(x="PAY_5", data=data, ax=ax[1,1])
sns.countplot(x="PAY_6", data=data, ax=ax[1,2])

# COMMAND ----------

#things to consider, are the 0's here for the actual amount or representing nan
print(data['BILL_AMT1'].value_counts().nlargest(5))
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize = (12,4))
sns.distplot(data['BILL_AMT1'], ax=ax[0,0])
sns.distplot(data['BILL_AMT2'], ax=ax[0,1])
sns.distplot(data['BILL_AMT3'], ax=ax[0,2])
sns.distplot(data['BILL_AMT4'], ax=ax[1,0])
sns.distplot(data['BILL_AMT5'], ax=ax[1,1])
sns.distplot(data['BILL_AMT6'], ax=ax[1,2])

# COMMAND ----------

def plot_corr(df,size=15):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr,cmap=plt.get_cmap('jet'))
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(cax)
plot_corr(data)

# COMMAND ----------

data['def_pay'].value_counts()
plt.figure(figsize=(12, 4))
sns.barplot(x="EDUCATION", y="def_pay", hue="SEX", data=data)
plt.title("Default by Education and Sex")

# COMMAND ----------

data['EDUCATION'] = data['EDUCATION'].astype(int)
data['def_pay'] = data['def_pay'].astype(int)

# COMMAND ----------

plt.figure(figsize=(8, 4))
sns.boxplot(x="def_pay", y="AGE", data=data)
plt.title("Distribution of Default by Age")

# COMMAND ----------

plt.figure(figsize=(8, 4))
sns.violinplot(x="def_pay", y="AGE", hue="SEX", data=data, split=True)
plt.title("Distribution of Default by Sex and Age")

# COMMAND ----------

fig, ax1 = plt.subplots(ncols=1, figsize=(16,6))
sns.boxplot(ax=ax1, x="AGE", y="LIMIT_BAL", data=data,hue='SEX')
plt.ylabel('Credit Card limit')
plt.xlabel("AGE")
plt.title("Default amount of credit card limit,group by age and sex")
plt.show()

# COMMAND ----------

sns.countplot(x="MARRIAGE", data=data,hue="def_pay", palette="muted")
plt.legend(["Default","Not Default"])

# COMMAND ----------

# Preprocessing

# COMMAND ----------

data['EDUCATION'].loc[data['EDUCATION'] == 6]=5
data['def_pay'].value_counts()

# COMMAND ----------

total_balance=data[['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]
data['total_balance']=total_balance.sum(axis=1)

# COMMAND ----------

total_pay_amount=data[['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4','PAY_AMT5']]
data['total_pay_amount']=total_pay_amount.sum(axis=1)

# COMMAND ----------

#categorical and continuous variable

# COMMAND ----------

cat_v = []
con_v = []
for c in data.columns:
    if len(data[c].value_counts().index)<=15:
        cat_v.append(c)
    else:
        con_v.append(c)

# COMMAND ----------

print("The continuous variables: ", con_v, "\n")
print("The categorical variables: ", cat_v,"\n")
print("There are ",len(con_v)," continuous variables")
print("There are ",len(cat_v)," categorical variables")

# COMMAND ----------

data[con_v].describe()


# COMMAND ----------

pd.DataFrame(data[cat_v], dtype='object').describe()

# COMMAND ----------

as_category = ['PAY_0', 'PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','SEX','EDUCATION','MARRIAGE']

# COMMAND ----------

as_numeric = ['ID','LIMIT_BAL','AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
             'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4','PAY_AMT5', 'PAY_AMT6','total_balance','total_pay_amount',
             ]

# COMMAND ----------

labels = ['default.payment.next.month']

# COMMAND ----------

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
         return X[self.attribute_names]

# COMMAND ----------

class DummyEncoder(TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    def transform(self, X, y=None, **kwargs):
        return pd.get_dummies(X, columns=self.columns)
    def fit(self, X, y=None, **kwargs):
        return self

# COMMAND ----------

num_pipe=Pipeline([
        ("selector", DataFrameSelector(as_numeric)),
        ("scale", StandardScaler())
        ])

# COMMAND ----------

cat_pipe=Pipeline([
        ("selector", DataFrameSelector(as_category)),
        ("convert", DummyEncoder(columns=as_category)),
        ("scale", StandardScaler())
        ])

# COMMAND ----------

full_pipeline = FeatureUnion(transformer_list=[
        ("cat_pipeline", cat_pipe ),
        ("num_pipeline", num_pipe ),
        ])

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
data_prepared = full_pipeline.fit_transform(data)

# COMMAND ----------

import random
#split the train data and test data
random.seed(777)
train_data = data.sample(frac=5/6, replace=False)
test_data = data.sample(frac=1/6, replace=False)
print (train_data.shape)
print (test_data.shape)

# COMMAND ----------

y_train = train_data['def_pay']
y_test=test_data['def_pay']
X_train = train_data.drop(['def_pay', 'ID'], axis= 1)
X_test = test_data.drop(['def_pay', 'ID'], axis= 1)
print (y_train.shape)
print (y_test.shape)
print (X_train.shape)
print (X_test.shape)

# COMMAND ----------

# Random forrest

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(class_weight = {0:1, 1:3})
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
acc_RF = round(accuracy_score(y_test,y_pred) * 100, 2)
print(RF.score(X_train, y_train))
print (acc_RF,"%")

# COMMAND ----------

features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = RF.feature_importances_
features = features.sort_values(by='importance',ascending=False)
plt.figure(figsize = (12,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='feature',y='importance',data=features)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
def search_model(X_reduced, y_train, est, param_grid, n_jobs, cv):
    model = GridSearchCV(estimator  = est,
                                     param_grid = param_grid,
                                     scoring = 'f1_weighted',
                                     verbose = 10,
                                     n_jobs = n_jobs,
                                     iid = True,
                                     cv = cv)
    # Fit Grid Search Model
    model.fit(X_reduced, y_train)   
    return model

# COMMAND ----------



# COMMAND ----------

y_scores_reg = cross_val_predict(RF, X_test, y_test, cv=3)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
def plot_roc_curve(fpr, tpr, label=None):
  plt.plot(fpr, tpr, linewidth=2, label=label)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.axis([0, 1, 0, 1])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()

# COMMAND ----------

#SVM

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

