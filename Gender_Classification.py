# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 23:07:25 2018

@author: kisku
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 02:47:18 2018

@author: kisku
"""
import sklearn
import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# 1. Decision Tree Classifier 
classifier1 = tree.DecisionTreeClassifier()

# 2. Logistic Classifier
classifier2 = LogisticRegression(random_state = 0)

# 3. KNN Classifiers
classifier3 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# 4 SVM Classifiers
classifier4 = SVC(kernel = 'linear', random_state = 0)

# 5 Kernel SVM Classifiers
classifier5 = SVC(kernel = 'rbf', random_state = 0)

# 6 Naive Bayes Classifiers
classifier6 = GaussianNB()

# 7 Random Forest Classifiers
classifier7 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


#classifier = np.array([classifier2, classifier3, classifier4, classifier5, classifier6, classifier7])
#classifier = classifier.fit(X,Y)

#for i in classifier:
#    prediction = classifier[i].predict([[190, 70, 43]])
#    print(prediction)

classifier1 = classifier1.fit(X, Y)
classifier2 = classifier2.fit(X, Y)
classifier3 = classifier3.fit(X, Y)
classifier4 = classifier4.fit(X, Y)
classifier5 = classifier5.fit(X, Y)
classifier6 = classifier6.fit(X, Y)
classifier7 = classifier7.fit(X, Y)

prediction1 = classifier1.predict([[190, 70, 43]])
prediction2 = classifier2.predict([[190, 70, 43]])
prediction3 = classifier3.predict([[190, 70, 43]])
prediction4 = classifier4.predict([[190, 70, 43]])
prediction5 = classifier5.predict([[190, 70, 43]])
prediction6 = classifier6.predict([[190, 70, 43]])
prediction7 = classifier7.predict([[190, 70, 43]])

prediction = [prediction1, prediction2, prediction3, prediction4, prediction5, prediction6, prediction7]
print(prediction)