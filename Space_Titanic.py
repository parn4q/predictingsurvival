

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("D:/Kaggle/Space Titanic/train.csv")
test = pd.read_csv("D:/Kaggle/Space Titanic/test.csv")


data.info() #there are null values

data['Transported'] = data['Transported'].astype(object)

# Count the number of NA values in the DataFrame
na_count = data.isna().sum()

# Display the result
print("Number of NA values in each column:")
print(na_count)


numdf = data.select_dtypes(include = 'float64')

numdf = numdf.drop('Age', axis = 1)

numdf.fillna(0, inplace = True)

numdf = pd.concat([numdf, data['Age']], axis = 1)

sns.histplot(numdf, x = 'Age')
### Cat Variables

catdf = data.select_dtypes(include = 'object')
catdf = catdf.drop(['PassengerId', 'Name'], axis = 1) #Probably will turn cabin into three separate variables for analyzing
catdf[['deck', 'room_num', 'side']] = catdf['Cabin'].str.split('/', expand = True)

catdf = catdf.drop(['Cabin'], axis = 1)
###Hardly anyone is in VIP.  All cat variables will be used except VIP

for a in catdf: 
    catdf[a] = LabelEncoder().fit_transform(catdf[a])
    
agemod = pd.concat([numdf, catdf], axis = 1)

agenavalues = agemod[agemod['Age'].isna()] # contains the data I'll use to predict age

#Now we will predict age from the other dataset without na values to predict on agenavalues

age_no_na = agemod.dropna()

for i in age_no_na.columns:
   age_no_na.plot.scatter(x = i, y = 'Age')
    



xagemod = age_no_na.drop('Age', axis = 1)
yagemod = age_no_na['Age']


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection as skm
import statsmodels.api as sm
from sklearn.metrics import accuracy_score


# Create a Random Forest classifier
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
rf_reg.fit(xagemod, yagemod)

kfold = skm.KFold(5, shuffle = True)
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

# Instantiate the grid search model
grid_search = skm.GridSearchCV(estimator = rf_reg, param_grid = param_grid, 
                          cv = kfold, n_jobs = -1, verbose = 2)

grid_search.fit(xagemod, yagemod)

grid_search.best_estimator_


new_ages = grid_search.predict(agenavalues.drop('Age', axis = 1))
new_ages = pd.DataFrame(new_ages)
agenavalues = agenavalues.reset_index()
agefull = pd.merge(agenavalues.drop('Age', axis = 1), new_ages, left_index=True, right_index=True)
agefull = agefull.rename(columns = {0: 'Age'})

agefull['Age'] = round(agefull['Age'])
agefull = agefull.drop(['index', 'level_0'], axis = 1)

agefull = pd.concat([age_no_na, agefull], axis = 0)

# creating the model dataset

mod_df = agefull

# mod_df['Adult'] = np.where(mod_df['Age'] < 18, 0, 1)
# mod_df['Age_Bin'] = pd.cut(mod_df['Age'], bins = [0, 10, 20, 30, 40, 50, 60, 70, 80], 
#                           labels = ['<10', '10s', '20s', '30s','40s', '50s', '60s', '70s'])
### Explore the new variables




y = mod_df['Transported']
x = mod_df.drop(['Transported', 'Adult', 'Age_Bin'], axis = 1)
x = sm.add_constant(x)


mnlmod = sm.MNLogit(y,x).fit()

mnlmod.summary()


x = mod_df.drop(['Transported', 'Age', 'Age_Bin'], axis = 1)


mnlmod = sm.MNLogit(y,x).fit()

mnlmod.summary()


x = mod_df.drop(['Transported', 'Age', 'Adult'], axis = 1)
x['Age_Bin'] = LabelEncoder().fit_transform(x['Age_Bin'])

mnlmod = sm.MNLogit(y,x).fit()

mnlmod.summary()


#AFter using MLN as a baseline model, we will not use 'Adult' or 'Age bin' Variable
#As 'Adult' is not significant and age_bin does not change the results of the model
#Vip is useless

#Create train and test set

#Train test split
x_train, x_test, y_train, y_test = train_test_split(mod_df, mod_df['Transported'], test_size=0.1, random_state=20)

x_train = x_train.drop(['Transported', 'VIP'], axis = 1)
x_test = x_test.drop(['Transported', 'VIP'], axis = 1)


x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

mnlmod = sm.MNLogit(y_train, x_train).fit()

mnlmod.summary()

mnlpred = mnlmod.predict(x_test)
mnlpred = np.argmax(mnlpred, axis = 1)


accuracy_score(y_test, mnlpred)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

x_train = x_train.drop('const',axis = 1)
ldamod = LinearDiscriminantAnalysis().fit(x_train, y_train)

ldapred = ldamod.predict(x_test.drop('const', axis = 1))

accuracy_score(y_test, ldapred)

qdamod = QuadraticDiscriminantAnalysis().fit(x_train, y_train)

qdapred = qdamod.predict(x_test.drop('const', axis = 1))

accuracy_score(y_test, qdapred)

from sklearn.naive_bayes import BernoulliNB

nbmod = BernoulliNB().fit(x_train, y_train)

nbpred = nbmod.predict(x_test.drop('const', axis = 1))

accuracy_score(y_test, nbpred)



# Create a Random Forest 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

rf = RandomForestClassifier()

# Fit the model on the training data
rf.fit(x_train, y_train)

kfold = skm.KFold(5, shuffle = True)
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [3, 6, 9,12],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 300, 500, 700, 1000]
}

# Instantiate the grid search model
grid_search = skm.GridSearchCV(estimator = rf,param_grid = param_grid, 
                          cv = kfold, n_jobs = -1, verbose = 2)

grid_search.fit(x_train, y_train)

grid_search.best_estimator_

rfpred = grid_search.predict(x_test.drop('const', axis = 1))

accuracy_score(y_test, rfpred)


gbmod = GradientBoostingClassifier()

param_grid = {
    'learning_rate': [0.1, 0.3, 0.6, 0.9],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2,4,6,8],
    'min_samples_leaf': [1,2,3,4],
    'min_samples_split': [2,4,6,8],
    'n_estimators': [100, 300, 500, 700]
}

gbmod = skm.GridSearchCV(gbmod, param_grid=param_grid, cv = kfold, verbose = 3,
                         n_jobs=-1)

gbmod.fit(x_train, y_train)

gbmod.best_estimator_





















