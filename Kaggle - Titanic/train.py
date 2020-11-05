# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:02:18 2020
@author: kkoni
"""
# importy
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
# from scipy import stats

# wczytanie danych
train=pd.read_csv("C:/Users/kkoni/Desktop/Kaggle/train.csv")
test=pd.read_csv("C:/Users/kkoni/Desktop/Kaggle/test.csv")

# opcje formatujące
pd.options.display.float_format = "{:.2f}".format
pd.options.display.max_columns = None

# prosta EDA
print(train.head())
print("*********** TRAIN PODSUMOWANIE ***********")
print(train.describe(include='all'))
print("*********** TEST PODSUMOWANIE ***********")
print(test.describe(include='all')) 
print(train.isnull().sum())
print(train.dtypes)



x1 = train.loc[train.Survived==1, 'Pclass']
x2 = train.loc[train.Survived==0, 'Pclass']
kwargs = dict(alpha=0.5, bins=30)
plt.hist(x1, **kwargs, color='g', label='Survived', normed=True)
plt.hist(x2, **kwargs, color='b', label='Died', normed=True)
plt.gca().set(title='Frequency Histogram of Diamond Depths', ylabel='Frequency')
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.legend();

# imputacja

# def imp_ind(var):
#     if var=='None':
#         return 1
#     else:
#           return 0
# train['I_Age'] = train['Age'].apply(lambda x : 1 if x == 'nan' else 0)  # Convert to numeric
# train['I_Age']=train['Age'].apply(imp_ind)
# train['I_Cabin']=train['Cabin'].apply(imp_ind)
# train['I_Embarked']=train['Embarked'].apply(imp_ind)
# train['I_Fare']=train['Fare'].apply(imp_ind)

def word_count(message):
    wrdcount = 0
    for i in message.split():
        eawrdlen = len(i) / len(i)
        wrdcount = wrdcount + eawrdlen
    return wrdcount

train['Cabin'] = train['Cabin'].replace(np.NaN, 'U')
train['Cabin_Count'] = train['Cabin'].apply(word_count)

# def cabin_type_flg(message,cat):
#     if cat in message:
#         return 1
#     else:
#         return 0

# train['Cabin_A'] = train['Cabin'].str.contains('A')

def salut(message):
    if "Mr." in message :
        return "Mr"
    if "Mrs." in message :
        return "Mrs"
    if "Miss." in message :
        return "Miss"
    if "Master." in message :
        return "Master"
    if "Rev." in message :
        return "Don"
    if "Dr." in message :
        return "Dr"
    else:
        return "Other"
    
 # Mr.	516  # Miss.	182  # Mrs.	125  # Master.	40  # Dr.	7  # Rev.	6  # Col.	2
 # Major.	2  # Mlle.	2  # the Countess.	1  # Capt.	1  # Don.	1   # Mme.	1
 # Sir.	1  # Jonkheer.	1  # Lady.	1  # Ms.	1

train['Salut'] = train['Name'].apply(salut)
train['Family_size'] = train['SibSp']+train['Parch']+1
train['Cabin_per_fm'] = train['Cabin_Count']/train['Family_size']
train['Family_wealth'] = train['Family_size']*train['Pclass']
train['C_type'] = train['Cabin'].str.slice(0,1)

# cechy dalej:
# family size 
# fare per family size, fare per liczba kabin
# no of cabins per family size + kombinacje
# prefixy do biletów
# cena vs zamożnosc
# rozmiar rodziny vs zamoznosc

# model/segment rodziny do której należy, jego rola w rodzinie
# czy jest ktos na pokladzie o tym samym nazwisku kto przezyl/nie przezyl
# passenger id?
    
    
# def substrings_in_string(big_string, substrings):
#     for substring in substrings:
#         if string.find(big_string, substring) != -1:
#             return substring
#     return np.nan

# cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
# train['Deck']=train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

imp_custom_num=SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=False, copy=False)
patch1 = train.select_dtypes(include=['number'])
I_patch1 = pd.DataFrame(imp_custom_num.fit_transform(patch1))
I_patch1.columns = patch1.columns   

# for arr in patch1: #do not need the loop at this point, but looks prettier
#     print(stats.describe(arr))
    
imp_custom_cat=SimpleImputer(missing_values=np.nan, strategy='most_frequent', add_indicator=False, copy=False)
patch2 = train[['Sex','Embarked','Salut','C_type']]
I_patch2 = pd.DataFrame(imp_custom_cat.fit_transform(patch2))
I_patch2.columns = patch2.columns   
I_patch3 = pd.get_dummies(I_patch2, sparse=False)

x_train = pd.concat([I_patch1, I_patch2, I_patch3], axis=1)
x_train['Fare_vs_wealth'] = x_train['Fare']*x_train['Pclass']
x_train['Fare_per_size'] = x_train['Fare']/x_train['Family_size']

print("FARE", x_train.groupby('Survived')['Fare'].describe())
print("AGE", x_train.groupby('Survived')['Age'].describe())
# print("CABIN A", x_train.groupby('Survived')['Cabin_A_True'].describe())
# print("CABIN B", x_train.groupby('Survived')['Cabin_B_True'].describe())
# print("CABIN C", x_train.groupby('Survived')['Cabin_C_True'].describe())
# print("CABIN D", x_train.groupby('Survived')['Cabin_D_True'].describe())
# print("CABIN E", x_train.groupby('Survived')['Cabin_E_True'].describe())
# print("CABIN F", x_train.groupby('Survived')['Cabin_F_True'].describe())
# print("CABIN G", x_train.groupby('Survived')['Cabin_G_True'].describe())
print(x_train.isnull().sum())

sql=x_train.groupby(['Sex_female', 'C_type']).agg({'Survived': [np.mean]})
pd.merge(x_train, sql, how='left', left_on=['Sex_female','C_type'], right_on=['Index 0', 'Index 1'])


y_train = x_train["Survived"]
x_train.drop(['Survived','PassengerId','Sex','Sex_male','Embarked_C','Embarked','C_type','Salut'] , axis=1, inplace=True)





# final_iv, IV = data_vars(x_train , x_train.Survived)


# 'Salut_Don','Salut_Master','Salut_Miss','Salut_Mr','Salut_Dr','Salut_Mrs','Salut_Other'

test_size = 0.70  
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=42)

lr_list = [0.05, 0.1, 1]
ne_list = [10, 20, 50, 100, 200]
gb_perf = pd.DataFrame(columns=["acc","nest","l_rate","ofit"])

for learning_rate in lr_list:
    for n_estimators in ne_list:
        gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, 
                                            min_samples_leaf=30, min_weight_fraction_leaf=0, max_features=None, max_depth=2, random_state=10)
        gb_clf.fit(x_train, y_train)
        ofit=gb_clf.score(x_train, y_train)/gb_clf.score(x_val, y_val)
        a_row = pd.DataFrame({"acc":[gb_clf.score(x_val, y_val)], "nest":[n_estimators], "l_rate":[learning_rate], "ofit":[ofit]}) 
        gb_perf = pd.concat([gb_perf, a_row])
        print("GB *** LR=", learning_rate, "N=", n_estimators, "TR={0:.3f}".format(gb_clf.score(x_train, y_train)),
              "VAL={0:.3f}".format(gb_clf.score(x_val, y_val)), "OF={0:.3f}".format(ofit))

# plt.scatter(gb_perf["nest"], gb_perf["l_rate"], s=gb_perf["ofit"], alpha=0.5)

# df = px.data.gapminder()

lr_list = [0.05, 0.1, 1]
ne_list = [10, 20, 50]
ld_list = [0.5, 1, 5 ]
al_list = [0, 0.1, 2, 5]

xgb_perf = pd.DataFrame(columns=["acc","nest","l_rate","ofit"])

for learning_rate in lr_list:
    for n_estimators in ne_list:
        # for alpha in al_list:
        #     for lb in ld_list:
        xgb_clf2 = XGBClassifier(n_estimators=n_estimators,learning_rate=learning_rate, random_state=10)
        xgb_clf2.fit(x_train, y_train)
        score = xgb_clf2.score(x_val, y_val)
        ofit = xgb_clf2.score(x_train, y_train)/score
        b_row = pd.DataFrame({"acc":[xgb_clf2.score(x_val, y_val)], "nest":[n_estimators], "l_rate":[learning_rate], "ofit":[ofit]}) 
        xgb_perf = pd.concat([xgb_perf, b_row])
        print("XGB *** LR=", learning_rate, "N=", n_estimators, "TR={0:.3f}".format(xgb_clf2.score(x_train, y_train)),
              "VAL={0:.3f}".format(score), "OF={0:.3f}".format(ofit))
        
# gb_champ=GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)
# gb_champ.fit(x_train, y_train)
# print(gb_champ.feature_importances_)

gb_champ = XGBClassifier(n_estimators=100,learning_rate=1)
gb_champ.fit(x_train, y_train)

stat=x_train.describe(include='all')
# stat_test=x_test.describe(include='all')