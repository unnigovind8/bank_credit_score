#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


credit_data = pd.read_csv("credit.csv")


# In[3]:


credit_data.head()


# In[4]:


credit_data.columns


# In[5]:


credit_data.shape


# In[6]:


credit_data.info()


# In[ ]:





# In[7]:


credit_data["Customer_ID"].nunique()


# In[ ]:





# #### Data cleaning

# * Missing Values 

# In[8]:


credit_data.isna().sum()


# In[ ]:





# * Duplicated values

# In[9]:


credit_data.duplicated().sum()


# In[10]:


#sns.boxplot(x = credit_data["Annual_Income"])


# In[ ]:





# * Inconsistant Data

# In[11]:


credit_data.loc[41488:41490]


# In[12]:


# proper format for name column

credit_data["Name"] = credit_data["Name"].str.capitalize()


# In[13]:


credit_data[credit_data["Customer_ID"] == 30454].head()


# In[14]:


# Dealing with "Payment_of_Min_Amount" column

credit_data["Payment_of_Min_Amount"].value_counts()


# In[15]:


minpay_mode = credit_data["Payment_of_Min_Amount"].mode()[0]

credit_data["Payment_of_Min_Amount"].replace("NM", minpay_mode, inplace = True)


# In[16]:


credit_data[credit_data["Payment_of_Min_Amount"] == "NM"]


# In[17]:


# Dealing with "Num_bnk_account" column

sns.histplot(x = credit_data["Num_Bank_Accounts"], kde=True)


# In[18]:


mode_bnkacc = credit_data["Num_Bank_Accounts"].mode()[0]
print(mode_bnkacc)
credit_data["Num_Bank_Accounts"].replace(0, mode_bnkacc, inplace=True)


# In[19]:


credit_data[credit_data["Num_Bank_Accounts"] == 0]


# In[ ]:





# ### EDA

# In[20]:


credit_data.describe()


# In[21]:


# distribution of occupation

work = credit_data.groupby("Customer_ID")["Occupation"].first().value_counts().reset_index(name = "Count")
work.rename({"index":"Occupation"}, axis=1, inplace=True)
work



# In[22]:


plt.figure(figsize=(18,8))
ax = sns.barplot(y = work["Count"], x = work["Occupation"], palette="Reds")
ax.bar_label(ax.containers[0])
plt.xticks(rotation = 45)
plt.show()


# In[ ]:





# In[23]:


# Number of Bank account

bank_acc = credit_data.groupby('Customer_ID')['Num_Bank_Accounts'].value_counts().reset_index(name='Count')

bank_acc


# In[24]:


#for same account there are different number of bank accounts
bank_acc[bank_acc["Customer_ID"] == 27857]


# In[25]:


#bank_acc["Num_Bank_Accounts"].value_counts()


# In[26]:


plt.figure(figsize=(10,4))
ax = sns.barplot(x = bank_acc["Num_Bank_Accounts"].value_counts().index, y = bank_acc["Num_Bank_Accounts"].value_counts().values, palette = "copper", width=0.5)
ax.bar_label(ax.containers[0])
plt.ylabel("Total count")
plt.xlabel("Number of Bank Accounts")

plt.tight_layout()
plt.show()


# In[ ]:





# In[27]:


# Payment of Min Amount

min_pay = credit_data["Payment_of_Min_Amount"].value_counts()
#print(min_pay)

plt.figure()
plt.pie(x = min_pay.values, labels=["Yes","No"], explode = [0.03,0.03], colors=["green","olive"],  autopct='%1.1f%%', startangle=140)
plt.title("Payment of minimum amount by customers")
plt.show()


# In[ ]:





# In[28]:


# Annual income

annual_income = credit_data.groupby('Customer_ID')['Annual_Income'].value_counts().reset_index(name='Count')


# In[29]:


plt.figure(figsize=(15, 6))
plt.plot(annual_income.index, annual_income['Annual_Income'], alpha=0.5)
plt.title('Annual Income')
plt.xlabel('Index')
plt.ylabel('Annual Income')
plt.grid(True)
plt.show()



# In[30]:


# Heat map

plt.figure(figsize=(18,8))
sns.heatmap(credit_data.corr(), annot=True)


# In[ ]:





# In[31]:


# Distribution of Credit_Score

credit_score = credit_data.groupby("Customer_ID")["Credit_Score"].value_counts().reset_index(name = "count")
credit_score["Credit_Score"].value_counts()


# In[32]:


plt.figure(figsize=(10,4))
ax = sns.pointplot(x = credit_score["Credit_Score"].value_counts().index, y = credit_score["Credit_Score"].value_counts().values,linestyles='--',color="purple" )
plt.xlabel("Credit Score")
plt.ylabel("Total count")
plt.tight_layout()
plt.show()


# In[ ]:





# ### Feature Selection

# #### Feature Important from Tree-Based Models

# In[91]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

X = credit_data.drop(columns=["ID","Customer_ID","Credit_Score","Name","Occupation","Type_of_Loan","Payment_Behaviour"])
y = credit_data["Credit_Score"]

X["Credit_Mix"] = encoder.fit_transform(X["Credit_Mix"])
X["Payment_of_Min_Amount"] = encoder.fit_transform(X["Payment_of_Min_Amount"])

rf = RandomForestClassifier()
rf.fit(X,y)

feature_importance = rf.feature_importances_

indices = feature_importance.argsort()[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], feature_importance[indices[f]]))


# In[ ]:





# In[92]:


col = X.columns
for i in range(0,21):
    print(i, " ---> ", col[i])


# In[ ]:





# #### Recursive Feature Elimination

# In[94]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# seperating feature and target class

x = credit_data.drop(columns=["ID","Customer_ID","SSN","Credit_Score","Name","Occupation","Type_of_Loan","Payment_Behaviour"])
y = credit_data["Credit_Score"]

x["Credit_Mix"] = encoder.fit_transform(x["Credit_Mix"])
x["Payment_of_Min_Amount"] = encoder.fit_transform(x["Payment_of_Min_Amount"])

estimator_rf = RandomForestClassifier()

selector = RFE(estimator = estimator_rf, n_features_to_select=10)

selector = selector.fit(x,y)

selected_indices = selector.get_support(indices=True)

selected_features = x.columns[selected_indices]

print(selected_features)


# In[ ]:





# In[143]:


#Plotiing feature ranks

d = {"Features": x.columns, "Feature ranking" : ranking}

data = pd.DataFrame(d)


# In[144]:


data.head()


# In[146]:


# Plot feature importances
plt.figure(figsize=(12, 6))
plt.bar(x = data["Features"], height = data["Feature ranking"], align='center', color='skyblue', alpha=0.7)
plt.xticks(rotation=80)
plt.xlabel('Features')
plt.ylabel('Feature Ranking')
plt.title('Feature Ranking using Recursive Feature Elimination')
plt.tight_layout()
plt.show()


# In[ ]:





# In[20]:


# Selected Feature

selected_features = credit_data[["Annual_Income","Num_Bank_Accounts","Num_Credit_Card","Num_of_Delayed_Payment","Num_Credit_Inquiries","Credit_Mix","Outstanding_Debt","Credit_Utilization_Ratio","Credit_History_Age","Payment_of_Min_Amount","Credit_Score"]]


# In[21]:


credit_features = selected_features.copy()


# In[ ]:





# ### Data Preprocessing

# #### Encoding and Scaling

# In[22]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

encoder_le = LabelEncoder()
oneh_encoder = OneHotEncoder()

scaler = StandardScaler()


# In[23]:


X = selected_features.drop(columns=("Credit_Score"))
y = selected_features["Credit_Score"]


# In[ ]:





# In[24]:


#Encoding and Scaling predictive features

#Onehot encoding

from sklearn.preprocessing import OneHotEncoder
oneh_encoder = OneHotEncoder()


#Encoding and Scaling predictive features

creditmix = X["Credit_Mix"].values.reshape(-1,1)

en_creditmix = oneh_encoder.fit_transform(creditmix).toarray()
encoded_credtmix = pd.DataFrame(en_creditmix, columns = oneh_encoder.get_feature_names_out(['Creditmix_']))



pay_minamt = X["Payment_of_Min_Amount"].values.reshape(-1,1)

en_paymin= oneh_encoder.fit_transform(pay_minamt).toarray()
encoded_paymin = pd.DataFrame(en_paymin, columns = oneh_encoder.get_feature_names_out(['Paymin_']))



# In[25]:


# merging encodedvalues 

encoded_credtmix.set_index(X.index, inplace = True)
X = pd.concat([X, encoded_credtmix], axis=1)

encoded_paymin.set_index(X.index, inplace = True)
X = pd.concat([X,encoded_paymin], axis=1)

X.drop(columns=["Credit_Mix","Payment_of_Min_Amount"], inplace=True)


# In[30]:


#tobe_scaled = X[["Annual_Income","Outstanding_Debt","Credit_Utilization_Ratio","Credit_History_Age"]]
#tobe_scaled.to_csv("E:/DA/ict_dsa/aa_Internship/week4/web_ onehot encoder/bank_web_xg/tobe_scaled.csv")

#X.to_csv("E:/DA/ict_dsa/aa_Internship/week4/web_ onehot encoder/bank_web_xg/tobe_scaled.csv")


# In[41]:


X = scaler.fit_transform(X)


# In[42]:


# Encoding target feature
y = encoder_le.fit_transform(y)


# In[43]:


credit_features["Labelled"] = y
credit_features.head()


# In[ ]:





# #### Train test Split

# In[44]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Modeling

# * SVM

# In[46]:


#Initializing and Importing SVM classifier
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)


# In[47]:


svm_predict = svm_model.predict(X_test)


# In[48]:


# EVALUATION

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy score : ", accuracy_score(svm_predict, y_test))
print("Classification report:\n ", classification_report(y_test, svm_predict))


# In[ ]:





# * Decision Tree

# In[45]:


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# In[46]:


dt_predict = dt_model.predict(X_test)


# In[47]:


# EVALUATION
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy score: ", accuracy_score(dt_predict, y_test))
print("Classification report:\n ", classification_report(y_test, dt_predict))


# In[31]:





# * Random forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)


# In[49]:


rf_predict = rf_model.predict(X_test)


# In[50]:


# EVALUATION
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy Score: ", accuracy_score(rf_predict, y_test))
print("Classification report:\n ", classification_report(y_test, rf_predict))


# In[35]:


#import pickle
#pickle.dump(rf_model, open("rf_model.pkl", "wb"))


# In[ ]:





# * Gradient Boost Classifier

# In[55]:


from sklearn.ensemble import GradientBoostingClassifier

gbc_model = GradientBoostingClassifier()
gbc_model.fit(X_train, y_train)


# In[56]:


gbc_predict = gbc_model.predict(X_test)


# In[57]:


# EVALUATION

print("Accuracy score: ", accuracy_score(gbc_predict, y_test))
print("Classification report:\n ", classification_report(y_test, gbc_predict))


# In[ ]:





# * XGBOOST

# In[30]:


from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)


# In[31]:


xgb_predict = xgb_model.predict(X_test)


# In[33]:


# EVALUATION
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy score: ", accuracy_score(xgb_predict, y_test))
print("Classification report:\n ", classification_report(y_test, xgb_predict))


# In[ ]:





# ### Cross validation

# In[61]:


from sklearn.model_selection import KFold, cross_val_score

classification_model = [("SVM Classifier",SVC()),
                       ("Decision Tree Classifier"  , DecisionTreeClassifier()),
                       ("Random Forest Classifier", RandomForestClassifier()),
                       ("Gradient Boost Classifier",GradientBoostingClassifier()),
                       ("XGBOOST Classifier", XGBClassifier())]

kfold = KFold(n_splits=5, random_state=42, shuffle=True)

cv_results = pd.DataFrame(columns=["Model", "CV_mean_score"])

for mname, model in classification_model:
    cross_val = cross_val_score(model, X_train,y_train, cv = kfold)
    mean_score = cross_val.mean()
    cv_results = cv_results.append({"Model": mname, "CV_mean_score": mean_score}, ignore_index=True)

print(cv_results)


# In[ ]:





# In[ ]:





# ### Hyperparameter Tuning

# * Decision Tree

# In[31]:


parameter_dt = {"criterion" : ["gini", "entropy", "log_loss"],
               'max_depth': [40, 50, 60, 70],
                'min_samples_split': (2, 10)}


# In[32]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

dt_grid_search = GridSearchCV(DecisionTreeClassifier(),
                                  parameter_dt,
                                  cv=5,
                                  scoring="accuracy",
                                  n_jobs=-1,
                                  verbose=1)


# In[65]:


dt_grid_result = dt_grid_search.fit(X_train, y_train)


# In[66]:


dt_grid_search.best_params_
dt_grid_result.best_score_


# In[33]:


# training Decision tree with tuned parameters

dt_tuned = DecisionTreeClassifier(criterion="entropy", max_depth=60, min_samples_split=2)
dt_tuned.fit(X_train, y_train)

dt_tunrd_pred = dt_tuned.predict(X_test)


# In[34]:


print("Accuracy of tuned DT model: ", accuracy_score(dt_tunrd_pred,y_test))


# In[35]:





# * Ranfom Forest

# In[69]:


parameters ={'max_depth': [40, 50, 60, 70],
              'criterion' : ['gini', 'entropy'],
              'n_estimators': [50,100,200,400] }


# In[70]:


#Bayesian optimization
from skopt import BayesSearchCV

bayes_search = BayesSearchCV(rf_model, parameters, n_iter=10, cv=5, scoring='accuracy')
bayes_search.fit(X_train, y_train)

# best hyperparameters
best_params = bayes_search.best_params_
print("Best Hyperparameters:", best_params)

#best parameters
best_model = bayes_search.best_estimator_


# In[71]:


bayes_search.best_score_


# In[45]:


# Training with tuned parameter

rf_tuned = RandomForestClassifier(criterion="gini", max_depth=50, n_estimators = 400)
rf_tuned.fit(X_train, y_train)


# In[46]:


rf_tuned_pred = rf_tuned.predict(X_test)
print("Accuracy  score on tuned model : ", accuracy_score(y_test,rf_tuned_pred ))


# In[47]:





# * XGBoost

# In[39]:


xg_parameter = {"learning_rate": (0.01, 1.0),  
    'max_depth': (1, 20), 
    'min_child_weight': (1, 10), 
    'colsample_bytree': (0.5, 1.0),  
    'n_estimators': (50, 200),  
    }


# In[40]:


#Bayesian optimization
from skopt import BayesSearchCV

xg_bayes_search = BayesSearchCV(xgb_model,
                               xg_parameter, 
                               n_iter=10, 
                               cv=5, 
                               scoring='accuracy')
xg_bayes_search.fit(X_train, y_train)

# best hyperparameters
xg_best_params = xg_bayes_search.best_params_
print("Best Hyperparameters:", xg_best_params)

#best parameters
xg_best_model = xg_bayes_search.best_estimator_


# In[40]:


# Training with tuned parameter

xgb_tuned = XGBClassifier(colsample_bytree = 0.8, learning_rate = 0.05, max_depth = 15, n_estimators = 50 )
xgb_tuned.fit(X_train, y_train)
xgb_tuned_pred = xgb_tuned.predict(X_test)


# In[41]:


print("Accuracy of tuned XGB: ", accuracy_score(y_test, xgb_tuned_pred))


# In[ ]:





# In[42]:


#import pickle
#pickle.dump(xgb_tuned, open("xg_tuned.pkl", "wb"))


# #### Cross validation on Random Forest Classifier

# In[80]:


from sklearn.model_selection import KFold, cross_val_score

kfold_tuned = KFold(n_splits=5, random_state=42, shuffle=True)

rf_crossval = cross_val_score(rf_tuned, X,y, cv = kfold_tuned)

rf_crossval.mean()


# In[ ]:





# #### Threshold Selection

# In[150]:


from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve

# Converting labels to binary format
y_test_binary = label_binarize(y_test, classes=np.unique(y_test))

# Predicting probabilities for the test set
y_scores = rf_tuned.predict_proba(X_test)



n_classes = len(rf_tuned.classes_)

# Calculateing precision-recall curve for each class
precision = dict()
recall = dict()
thresholds = dict()
for i in range(n_classes):
    precision[i], recall[i], thresholds[i] = precision_recall_curve(y_test_binary[:, i], y_scores[:, i])

# Ploting precision-recall curve 
plt.figure()
for i in range(n_classes):
    plt.plot(recall[i], precision[i], label='Class {}'.format(i))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


# In[149]:


f1_scores = dict()

# Calculate F1-score for each class
for i in range(n_classes):
    f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

# Finding the threshold value for each class that maximizes the F1-score
optimal_thresholds = dict()
for i in range(n_classes):
    optimal_idx = np.argmax(f1_scores[i])
    optimal_thresholds[i] = thresholds[i][optimal_idx]


for i in range(n_classes):
    print("Optimal Threshold for Class {}: {}".format(i, optimal_thresholds[i]))


# In[151]:





# * Serializing and saving the Random forest model

# In[152]:


import pickle
pickle.dump(rf_tuned, open("rf_tunedmodel.pkl", "wb"))


# In[ ]:




