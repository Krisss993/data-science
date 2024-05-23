import numpy as np
import pandas as pd

from sklearn.neighbors    import KNeighborsClassifier
from sklearn.ensemble     import AdaBoostClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.tree         import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import seaborn as sns

# PRZYGOTOWANIE DANYCH 
dataset = load_iris()
data = dataset.data
target = dataset.target

cols = dataset.feature_names
cols = [x[:-5].replace(' ','_') for x in cols]


df = pd.DataFrame(data=np.c_[data,target], columns=cols+['target'])
print(df)
df.info()



# usunięcie duplikatów
df[df.duplicated()]
df.drop_duplicates(inplace=True)
df.info()

# sprawdzenie brakujących danych
df.isnull().sum()

# wywietlenie statystyk
df.describe().T





# macierz korelacji danych
corr = df.corr()
corr
# Najmniejszą istotnosc ma cecha sepal_width




X = df.copy()
y = X.pop('target')

X.head()
y.head()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



#-----------------------------------------------------------------------------
# Testowanie różnych algorytmów regresji
scores = []
models = ['Support Vector Classifier', 'Naive Bayess classifier', 'AdaBoost Classifier', 'Decision Tree Classifier', 'RandomForest Classifier', 
          'KNeighbours Classifier']
 
#-----------------------------------------------------------------------------

# Support Vector Classifier
svc = SVC()

params = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1], 
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
    'degree': [2, 3, 4],  
    'coef0': [0, 0.1, 0.5, 1]  
}

grid_search = GridSearchCV(svc, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

r2 = best_model.score(X_test, y_test)
print("SVC R2:", r2)

scores.append(r2)



# Naive Bayess classifier
model = GaussianNB()

model.fit(data,target)

r2 = model.score(data, target)

print("Bayess R2:", r2)

scores.append(r2)




# Adaboost classifier
adaboost = AdaBoostClassifier(n_estimators=1000)

params = {
    'n_estimators': [50, 100, 200],  # Number of boosting stages
    'learning_rate': [0.01, 0.1, 1.0],  # Learning rate
    'algorithm': ['SAMME'] 
}

grid_search = GridSearchCV(adaboost, params, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)
print("Best Model:", best_model)
 
r2 = best_model.score(X_test, y_test)
print("Adaboost R2:", r2)

scores.append(r2)




# Tree classifier
tree = DecisionTreeClassifier()

params = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [10, 20, 30],  
    'min_samples_split': [2, 10, 20],  
    'min_samples_leaf': [3, 5, 10], 
    'max_features': ['sqrt', 'log2']  
}

grid_search = GridSearchCV(tree, params, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

r2 = best_model.score(X_test, y_test)
print("Tree regressor R2:", r2)

scores.append(r2)




# Random forest classifier
randomforest = RandomForestClassifier()

params = {
    'n_estimators': [100, 200],  
    'max_depth': [10, 20],  
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4],  
    'max_features': ['auto', 'sqrt', 'log2'],  
    'bootstrap': [True, False]  
}

grid_search = GridSearchCV(randomforest, params, cv=5, scoring='r2', n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
 
r2 = best_model.score(X_test, y_test)
print("Random Forest R2:", r2)

scores.append(r2)




# K-Neighbours classifier
kneighbours = KNeighborsClassifier()
params = {
    'n_neighbors': [3, 5, 7, 10],  
    'weights': ['uniform', 'distance'],  
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
    'leaf_size': [20, 30, 40],  
    'p': [1, 2]  
}

grid_search = GridSearchCV(kneighbours, params, cv=5, scoring='r2', n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

r2 = best_model.score(X_test, y_test)
print('K-Neighbours R2: {0:.2f}'.format(r2))

scores.append(r2)




#-----------------------------------------------------------------------------
ranking = pd.DataFrame({'Algorithms' : models , 'R2-Scores' : scores})
ranking = ranking.sort_values(by='R2-Scores' ,ascending=False)
ranking
 
sns.barplot(x='R2-Scores' , y='Algorithms' , data=ranking)




