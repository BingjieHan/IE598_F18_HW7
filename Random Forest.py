import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/wine/wine.data',
         header=None)

df.columns = ['Class label', 'Alcohol',
                    'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']

X, y = df.iloc[:,1:].values, df.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                      test_size=0.1,
                      random_state=0,
                      stratify=y)

n_space=np.arange(1,30,1)
param_grid = {'n_estimators': n_space}

'''
score=[]
for i in n_space:
    forest = RandomForestClassifier(criterion='gini',
                                 n_estimators=i,
                                 random_state=1,
                                 n_jobs=2)
    forest.fit(X_train, y_train)
    scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=5, n_jobs=1)
    score.append(np.mean(scores))
print(score)    

forest = RandomForestClassifier(criterion='gini',
                                 random_state=1,
                                 n_jobs=2)
forest_cv = GridSearchCV(forest, param_grid, cv=5)
forest_cv.fit(X_train, y_train)
print(forest_cv.bo est_params_)
#print(forest_cv.cv_results_)
 
'''

best_forest = RandomForestClassifier(criterion='gini',
                                 n_estimators=28,
                                 random_state=1,
                                 n_jobs=2)

best_forest.fit(X_train, y_train)
importances = pd.Series(data=best_forest.feature_importances_,
                        index= df.columns[1:])
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='red')
plt.title('Features Importances')
plt.show()

print("My name is Bingjie Han")
print("My NetID is: bingjie5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


