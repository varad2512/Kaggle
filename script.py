import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression

titanic_data = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")

titanic_data = titanic_data.drop(['Name'],axis = 1)
titanic_data = titanic_data.drop(['PassengerId'], axis = 1)
titanic_data['Embarked'] = titanic_data['Embarked'].fillna("S")
titanic_data['Cabin'] = titanic_data['Cabin'].fillna("S")

titanic_data = titanic_data.drop(['Embarked','Ticket','Cabin'],axis = 1)
titanic_test = titanic_test.drop(['Embarked','Ticket','Cabin','PassengerId','Name'],axis = 1)

titanic_data['Fare'] = titanic_data['Fare'].astype("int")
titanic_test["Fare"].fillna(titanic_test["Fare"].median(), inplace=True)
titanic_test["Fare"] = titanic_test["Fare"].astype("int")

count_nan_age_titanic = titanic_data["Age"].isnull().sum()
std_age  = titanic_data["Age"].std()
mean_age = titanic_data["Age"].mean()
random_created = np.random.randint(mean_age - std_age,mean_age + std_age, size = count_nan_age_titanic)
titanic_data["Age"][np.isnan(titanic_data["Age"])] = random_created
titanic_data["Age"] = titanic_data["Age"].astype("int")
count_for_test_age = titanic_test["Age"].isnull().sum()
std_age_test = titanic_test["Age"].std()
std_age_mean = titanic_test["Age"].mean()
random_test_age = np.random.randint(std_age_mean - std_age_test,std_age_mean + std_age_test, count_for_test_age)
titanic_test["Age"][np.isnan(titanic_test["Age"])] = random_test_age
titanic_test["Age"] = titanic_test["Age"].astype(int)

titanic_data["Family"] = titanic_data["Parch"] + titanic_data["SibSp"]
titanic_data["Family"].loc[titanic_data["Family"] > 0]  = 1
titanic_data["Family"].loc[titanic_data["Family"] == 0] = 0
titanic_data = titanic_data.drop(['Parch','SibSp'],axis=1)
titanic_test["Family"] = titanic_test["Parch"] + titanic_test["SibSp"]
titanic_test["Family"].loc[titanic_test["Family"] > 0]  = 1
titanic_test["Family"].loc[titanic_test["Family"] == 0] = 0
titanic_test = titanic_test.drop(['Parch','SibSp'],axis=1)

titanic_data["Sex"].loc[titanic_data["Sex"] == 'male'] = 1
titanic_data["Sex"].loc[titanic_data["Sex"] == 'female'] = 0
titanic_test["Sex"].loc[titanic_test["Sex"] == 'male'] = 1
titanic_test["Sex"].loc[titanic_test["Sex"] == 'female'] = 0

dum = pd.get_dummies(titanic_data["Pclass"])
dum.columns = ['1','2','3']
dum_test = pd.get_dummies(titanic_test["Pclass"])
dum_test.columns = ['1','2','3']
titanic_test = titanic_test.join(dum_test)
titanic_test = titanic_test.drop("Pclass",axis=1)
titanic_data = titanic_data.join(dum)
titanic_data = titanic_data.drop("Pclass",axis=1)

X_train = titanic_data.drop("Survived",axis=1)
Y_train = titanic_data["Survived"]
X_test  = titanic_test

'''Logistic Regression'''
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print(logreg.score(X_train, Y_train))









