# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:12:18 2020

@author: kkoni
"""



########################################################################################
# scoring
# test['I_Age']=test['Age'].apply(imp_ind)
# test['I_Cabin']=test['Cabin'].apply(imp_ind)
# test['I_Embarked']=test['Embarked'].apply(imp_ind)
# test['I_Fare']=test['Fare'].apply(imp_ind)

test['Cabin'] = test['Cabin'].replace(np.NaN, 'U')
test['Cabin_Count'] = test['Cabin'].apply(word_count)
test['Cabin_A'] = test['Cabin'].str.contains('A')
test['Cabin_B'] = test['Cabin'].str.contains('B')
test['Cabin_C'] = test['Cabin'].str.contains('C')
test['Cabin_D'] = test['Cabin'].str.contains('D')
test['Cabin_E'] = test['Cabin'].str.contains('E')
test['Cabin_F'] = test['Cabin'].str.contains('F')
test['Cabin_G'] = test['Cabin'].str.contains('G')
test['Salut'] = test['Name'].apply(salut)
test['Family_size'] = test['SibSp']+test['Parch']+1
test['Cabin_per_fm'] = test['Cabin_Count']/test['Family_size']
test['Family_wealth']= test['Family_size']*test['Pclass']

tpatch1 = test.select_dtypes(include=['number'])
I_tpatch1 = pd.DataFrame(imp_custom_num.fit_transform(tpatch1))
I_tpatch1.columns = tpatch1.columns   
    
tpatch2 = test[['Sex','Embarked','Cabin_A','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Salut']]
I_tpatch2 = pd.DataFrame(imp_custom_cat.fit_transform(tpatch2))
I_tpatch2.columns = tpatch2.columns   
I_tpatch3 = pd.get_dummies(I_tpatch2, sparse=False)

x_test = pd.concat([I_tpatch1, I_tpatch2, I_tpatch3], axis=1)
x_test.drop(['PassengerId','Sex','Embarked','Cabin_A','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Sex_male','Embarked_C',
              'Cabin_A_False','Cabin_B_False','Cabin_C_False','Cabin_D_False','Cabin_E_False','Cabin_F_False','Cabin_G_False','Salut'], axis=1, inplace=True)

x_test['Fare_vs_wealth'] = x_test['Fare']*x_test['Pclass']
x_test['Fare_per_size'] = x_test['Fare']/x_test['Family_size']

pred = pd.DataFrame(gb_champ.predict(x_test))
csv=pd.concat([test['PassengerId'], pred], axis=1)
# csv[1] = csv[1].astype(int)
# csv.rename(columns={"0": "Survived"})
# csv.to_csv('C:/Users/kkoni/Desktop/Kaggle/submit_190420_1.csv', index=False)

# Fare tylko w teście
# Pomysł na pipeline:
# prosta analiza - histogramy, braki danych
# brudne dane w numerze kabiny
# zmienne wyliczane
# - długosc name
# - interakcje (wiek<18 i bez rodziców, wiek+cena biletu, wiek i klasa, klasa i płeć, wiek/płeć/pokład)
# - czy wiek estymowany?
# - czy na pokładzie był ktos o tym samym nazwisku
# prefixy z kodu biletu?

# zmienne w oparciu o kod kabiny

# modele    
    

# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower

# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)

# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.