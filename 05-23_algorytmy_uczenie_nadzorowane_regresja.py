import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors    import KNeighborsRegressor
from sklearn.ensemble     import AdaBoostRegressor
from sklearn.ensemble     import RandomForestRegressor
from sklearn.tree         import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import PolynomialFeatures

from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns


# PRZYGOTOWANIE DANYCH 
dataset = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/insurance.csv')

df = dataset.copy()
df.info()

df['sex'].value_counts()
df['smoker'].value_counts()
df['region'].value_counts()

# zmiana typów danych na kategoryczne
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')
df.info()

# usunięcie duplikatów
df[df.duplicated()]
df.drop_duplicates(inplace=True)
df.info()

# sprawdzenie brakujących danych
df.isnull().sum()

# zamiana danych 
df = pd.get_dummies(df, drop_first=True, dtype='int')
df
df.info()


# wywietlenie statystyk
df.describe().T

plt.pie(df.sex_male.value_counts())
df.sex_male.value_counts().plot(kind='pie')

plt.hist(df['charges'], bins=50)
plt.hist(df['smoker_yes'], bins=50)


# macierz korelacji danych
corr = df.corr()
corr
# najmniejsza istotnosc maja zmienne region_...
#TODO
# w zaleznosci od potrzeb mozna sprobowac usunac te dane i zbudowac model bez nich


# WARTOSCI KORELACJI POSZCZEGOLNYCH KOLUMN Z KOLUMNA CHARGES POSORTOWANE MALEJACO
df.corr()['charges'].sort_values(ascending=False)
sns.set()
df.corr()['charges'].sort_values(ascending=True).plot(kind='barh')


X = df.copy()
y = X.pop('charges')

X.head()
y.head()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



#-----------------------------------------------------------------------------
# Testowanie różnych algorytmów regresji
scores = []
models = ['Linear Regression', 'Lasso Regression', 'AdaBoost Regression', 
          'Ridge Regression', 'Decision Tree Regressor','RandomForest Regression', 
          'KNeighbours Regression']
 
#-----------------------------------------------------------------------------
# Linear regression
lr = LinearRegression()
lr.fit(X_train , y_train)
y_pred = lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
scores.append(r2)
print('Linear Regression R2: {0:.2f}'.format(r2))


# GridSearchCV - LR

pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('regressor', LinearRegression())
])


params = {
    'poly__degree': [1, 2, 3],
    'regressor__fit_intercept': [True, False],
}

grid_search = GridSearchCV(pipeline, params, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best Parameters:", best_params)
print("Best Model:", best_model)

r2 = best_model.score(X_test, y_test)
print("GridSearchCV R2:", r2)

scores.append(r2)
models.insert(1, f'GS: {best_model[0]}')







# Lasso regressor
lasso = Lasso()

params = {
    'alpha': [0.1, 1.0, 10.0],
    'fit_intercept': [True, False],  
    'max_iter': [1000, 5000, 10000] 
}

grid_search = GridSearchCV(lasso, params, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)
print("Best Model:", best_model)
 
r2 = best_model.score(X_test, y_test)
print("Lasso R2:", r2)

scores.append(r2)

 
 

# Adaboost regressor
adaboost = AdaBoostRegressor(n_estimators=1000)

params = {
    'n_estimators': [50, 100, 200],  # Number of boosting stages
    'learning_rate': [0.01, 0.1, 1.0]  # Learning rate
}

grid_search = GridSearchCV(adaboost, params, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)
print("Best Model:", best_model)
 
r2 = best_model.score(X_test, y_test)
print("Test R2:", r2)

scores.append(r2)




# Ridge regressor
ridge = Ridge()

params = {
    'alpha': [0.1, 1.0, 10.0], 
    'fit_intercept': [True, False],   
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'] 
}


grid_search = GridSearchCV(ridge, params, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)
print("Best Model:", best_model)

r2 = best_model.score(X_test, y_test)
print("Ridge R2:", r2)

scores.append(r2)









# Tree regressor
tree = DecisionTreeRegressor()

params = {
    'criterion': ['absolute_error', 'poisson', 'squared_error'],
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




# Random forest regressor
randomforest = RandomForestRegressor()

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




# K-Neighbours regressor
kneighbours = KNeighborsRegressor()
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




