# -*- coding: utf-8 -*-

import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()

iris.keys()

#Dividimos el data set para entrenar y testear
X_train,X_test,Y_train,Y_test=train_test_split(iris['data'],iris['target'])
#X tiene las flores con sus datos
#Y tiene las clasificaciones
knn=KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train,Y_train)
print(knn.score(X_test,Y_test))

array=knn.predict([[5.2,3.4,1.6,1.1]])

print(array)