#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:46:56 2021

@author: manisha
"""

from sklearn.svm import SVC
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from chefboost import Chefboost as chef
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from utilities import *
from models import *
from graph_plots import *


config = {'algorithm': 'C4.5'}

def knnTrainer(X_train,Y_train):
    #Train the model and predict value using KNN
    knn = KNeighborsClassifier(n_neighbors=8)
    Y_train = Y_train.reshape(-1,)
    knn.fit(X_train[:,1:], Y_train)
    return knn
    

def knnPredictor(knn,X_test):
    y_pred_knn = knn.predict(X_test[:,1:])
    #y_pred_knn_np = np.array(y_pred_knn)
    return y_pred_knn


def randomForestTrainer(X_train,Y_train):
    #Random Forest Classifier training and prediction
    regressor = RandomForestClassifier(n_estimators=100, random_state=0)
    Y_train = Y_train.reshape(-1,)
    regressor.fit(X_train[:,1:], Y_train)
    return regressor


def randomForestPredictor(regressor, X_test):
    y_pred_rf = regressor.predict(X_test[:,1:])
    return y_pred_rf


def decisionTreeEntropyTrainer(X_train,Y_train):
    #Predict using Decision Tree - Entropylassifier
    clf_gini = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train[:,1:], Y_train)
    return clf_gini


def decisionTreeEntropyPredictor(clf_gini,X_test):
    y_pred_dt = clf_gini.predict(X_test[:,1:])
    return y_pred_dt


def c4_5_Trainer(X_train_c):
    #Predict using C4.5
    X_train_c = X_train_c[:,1:] #removing index from set
    cols=['C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Decision']
    data11 = getDataFrameFromNParray(X_train_c,cols)
    data11['Decision'] = data11.Decision.astype(object)
    model = chef.fit(data11, config)
    chef.save_model(model,"model.pkl")
    model = chef.load_model("model.pkl")
    return model


def c4_5_Predictor(model1,X_test):
    X_test = X_test[:,1:] #removing index from set
    X_test = X_test[:,:-1] #removing decision from test set
    y_pred_c_temp = []
    for row in X_test:
        prediction = chef.predict(model1,row)
        y_pred_c_temp.append(float(prediction))
    y_pred_c = np.array(y_pred_c_temp)
    y_pred_c = y_pred_c.astype(int)
    return y_pred_c


def naiveBayesTrainer(X_train,Y_train):
    naive = GaussianNB()
    Y_train = Y_train.reshape(-1,)
    naive.fit(X_train, Y_train)
    return naive;
    
    
def naiveBayesPredictor(naive,X_test):
    y_pred_n = naive.predict(X_test)
    return y_pred_n


def xgBoostTrainer(X_train,Y_train):
    xg = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
    Y_train = Y_train.reshape(-1,)
    xg.fit(X_train, Y_train)
    return xg;
    
    
def xgBoostPredictor(xg,X_test):
    y_pred_xg = xg.predict(X_test)
    y_xg = [round(value) for value in y_pred_xg] 
    y_xg = np.array(y_xg)
    return y_xg


def svmTrainer(X_train,Y_train):
    svm = SVC(kernel='linear')
    Y_train = Y_train.reshape(-1,)
    svm.fit(X_train, Y_train)
    return svm;
    
    
def svmPredictor(svm,X_test):
    y_pred_svm = svm.predict(X_test)
    return y_pred_svm
    

def logisticRegressionTrainer(X,Y):
    Y = Y.reshape(-1,)
    mod = LogisticRegression(random_state=0).fit(X,Y)
    return mod


def LogisticRegressionPredictor(mod,X_test):
    FinalPredicton = mod.predict(X_test)
    FinalPredicton = FinalPredicton.reshape(-1,1)
    FinalPredicton = FinalPredicton.astype(int)
    return FinalPredicton