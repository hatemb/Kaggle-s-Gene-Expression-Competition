#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:10:33 2017

@author: hatem
"""

import load_data as ld
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

x_train, y_train, x_test = ld.load_data('Data')

#LR = linear_model.LogisticRegression()
#LR.fit(x_train, y_train)
#probabilities = LR.predict_proba(x_test)
#
#for i, prob in enumerate(probabilities):
#    if prob[0] > prob[1]:
#        print(str(i)+", "+ str(prob[0]))
#    else:
#        print(str(i)+", "+ str(prob[1]))
    
power = list(range(-7, 0, 1))

parameters = {'penalty':['l1','l2'], 'C':[10**i for i in power]}
LR_param = linear_model.LogisticRegression()
gscv = GridSearchCV(LR, parameters, n_jobs=3, cv=10, scoring='roc_auc')
gscv.fit(x_train, y_train)
print(gscv.best_score_, gscv.best_params_)

LR_param = linear_model.LogisticRegression(C=gscv.best_params_['C'], penalty=gscv.best_params_['penalty'])
LR_param.fit(x_train, y_train)
prediction = LR_param.predict(x_test)





