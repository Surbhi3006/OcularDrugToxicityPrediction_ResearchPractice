#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:46:35 2021

@author: manisha
"""


import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from utilities import *
from models import *
from graph_plots import *


list_knn = []
list_rf = []
list_dt = []
list_naive = []
list_xgboost = []
list_svm = []
modelsUsed = ["KNN","Random Forest","Decision Tree","Naive Bayes","XG-Boost","SVM"]

def fetchDataset(sheet,cols):
    dataset11 = pd.read_csv(sheet)
    dataset = dataset11[cols]
    dataset.head()
    return dataset

def fetchDatasetAnn(file_datasheet, file_validation):
    dataset = fetchDataset(file_datasheet,['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Decision'])
    X = dataset.iloc[:,1:-1].values
    Y = dataset.iloc[:, -1:].values
    
    #fetch testing data from validation sheet
    dataset = fetchDataset(file_validation,['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume'])
    X_test = dataset.iloc[:,1:].values
    
    #handle missing values and replace with mean value in train data
    X = handleMissingValues(X)
    X_test = handleMissingValues(X_test)
    
    #feature scale train data
    sc_X, X = scaleSet(X)
    sc_XT, X_test = scaleSet(X_test)
    
    return X,Y,X_test

def fetchTrainDatasetAnn(file_datasheet):
    dataset = fetchDataset(file_datasheet,['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Decision'])
    X = dataset.iloc[:,1:-1].values
    Y = dataset.iloc[:, -1:].values
    
    #handle missing values and replace with mean value
    X = handleMissingValues(X)
    
    #split data into training and testing dataset
    X_train, X_test, Y_train, Y_test = splitDatasetToTrainTest(X,Y);
    
    #feature scale data
    sc_X, X_train, X_test = featureScaleDataSet(X_train,X_test)
    
    return X_train, Y_train, X_test, Y_test
    
def handleMissingValues(X):
    #handle missing values and replace with mean values
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    return X


def splitDatasetToTrainTest(X,Y):
    #split data into Training data and Test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
    return X_train,X_test,Y_train,Y_test
    

def featureScaleDataSet(train,test):
    sc_X = StandardScaler()
    train = sc_X.fit_transform(train)
    test = sc_X.transform(test)
    return sc_X,train,test

def scaleSet(train):
    sc_X = StandardScaler()
    train = sc_X.fit_transform(train)
    return sc_X,train

def calculateAverageAccuracyOfModels(avg_models,list_knn,list_rf,list_dt,list_naive,list_xgboost,list_svm):
    avg_models.append(sum(list_knn) / len(list_knn))
    avg_models.append(sum(list_rf) / len(list_rf))
    avg_models.append(sum(list_dt) / len(list_dt))
    avg_models.append(sum(list_naive) / len(list_naive))  
    avg_models.append(sum(list_xgboost) / len(list_xgboost))
    avg_models.append(sum(list_svm) / len(list_svm))
    
    for c in range(5):
        print("KNN Score:", list_knn[c])
    for c in range(5):
        print("RF Score:", list_rf[c])
    for c in range(5):
        print("DT Score:", list_dt[c])
    for c in range(5):
        print("NB Score:", list_naive[c])
    for c in range(5):
        print("XGB Score:", list_xgboost[c])
    for c in range(5):
        print("SVM Score:", list_svm[c])
        
    print("\nAvg Individual Accuracies :")
    for i in range (6):
        print(modelsUsed[i], " --> " , avg_models[i])    
    avg_accuracy = sum(avg_models)/len(avg_models)
    print("\nAverage Score All:", avg_accuracy)
    return avg_accuracy


def getDataFrameFromNParray(X,cols):
    dataframe = pd.DataFrame(X,columns=cols)
    return dataframe


def appendPredictions(X_backup,y_knn,y_rf,y_dt,y_nb,y_xg,y_svm):
    predValues = []
    for i in range(0,len(X_backup),1):
        a = []
        a.append(y_knn[i])
        a.append(y_rf[i])
        a.append(y_dt[i])
        a.append(y_nb[i])
        a.append(y_xg[i])
        a.append(y_svm[i])
        predValues.append(a)
    predValues = np.array(predValues)
    return predValues;
    

def exportToCSV(data,file):
    data.to_csv(file,index=False, header=True)
    return