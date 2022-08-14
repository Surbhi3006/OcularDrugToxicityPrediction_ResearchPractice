#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:46:35 2021

@author: surbhi sharma
"""


import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from utilities import *
from models import *
from graph_plots import *


def fetchColumns(sheet):
	df = pd.read_csv(sheet)
	columns = list(df.columns)
	return columns


def create_model_columns(models):
	cols = list()
	for n in models:
		cols.append('predict_' + n)
	return cols


def createMetrics(models):
	acc = dict()
	precision = dict()
	recall = dict()
	f1 = dict()
	for m in models:
		acc.update({m:dict()})
		precision.update({m:dict()})
		recall.update({m:dict()})
		f1.update({m:dict()})
	return acc,precision,recall,f1	


def fetchDataset(sheet,cols):
    dataset11 = pd.read_csv(sheet)
    dataset = dataset11[cols]
    dataset.head()
    return dataset


def fetchDatasetAnn(file_datasheet, file_validation):
    dataset = fetchDataset(file_datasheet,fetchColumns(file_datasheet))
    X = dataset.iloc[:,1:-1].values
    Y = dataset.iloc[:, -1:].values
    
    #fetch testing data from validation sheet
    dataset = fetchDataset(file_validation,fetchColumns(file_datasheet)[:-1])
    X_test = dataset.iloc[:,1:].values
    
    #handle missing values and replace with mean value in train data
    X = handleMissingValues(X)
    X_test = handleMissingValues(X_test)
    
    #feature scale train data
    sc_X, X = scaleSet(X)
    sc_XT, X_test = scaleSet(X_test)
    
    return X,Y,X_test


def fetchTrainDatasetAnn(file_datasheet):
    dataset = fetchDataset(file_datasheet,fetchColumns(file_datasheet))
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


def printMetrics(model, y_test, y_pred, a, p, r, f1, fold):
	if model in a:
		#print(f"\t\tAccuracy = {accuracy_score(y_true=y_test, y_pred=y_pred)}")
		a[model].update({"fold-"+str(fold):accuracy_score(y_true=y_test, y_pred=y_pred)})
		#print(f"\t\tPrecision = {precision_score(y_true=y_test, y_pred=y_pred)}")
		p[model].update({"fold-"+str(fold):precision_score(y_true=y_test, y_pred=y_pred)})
		#p[model].append(precision_score(y_true=y_test, y_pred=y_pred))
		#print(f"\t\tRecall = {recall_score(y_true=y_test, y_pred=y_pred)}")
		r[model].update({"fold-"+str(fold):recall_score(y_true=y_test, y_pred=y_pred)})
		#r[model].append(recall_score(y_true=y_test, y_pred=y_pred))
		#print(f"\t\tF1 Score = {f1_score(y_true=y_test, y_pred=y_pred)}")
		f1[model].update({"fold-"+str(fold):f1_score(y_true=y_test, y_pred=y_pred)})
		#f1[model].append(f1_score(y_true=y_test, y_pred=y_pred))


def calculateStatsForModels(models, a, p, r, f1):
	stat = dict()
	a_all = p_all = r_all = f1_all = 0
	
	for m in models:
		temp = dict()
		a_avg = sum(a[m].values()) / len(a[m])
		p_avg = sum(p[m].values()) / len(p[m])
		r_avg = sum(r[m].values()) / len(r[m])
		f1_avg = sum(f1[m].values()) / len(f1[m])
		temp.update({"accuracy":a_avg})
		temp.update({"precision":p_avg})
		temp.update({"recall":r_avg})
		temp.update({"f1":f1_avg})
		stat.update({m:temp})
	print(f"\n\nMetrics generated for all the models (mathematical average after k-fold): ")
	for m in stat:
		a_all = a_all + stat[m]["accuracy"]
		p_all = p_all + stat[m]["precision"]
		r_all = r_all + stat[m]["recall"]
		f1_all = f1_all + stat[m]["f1"]
	df = pd.DataFrame.from_dict(stat, orient='index')
	print(df.to_markdown())
	print(f"\n\nMetrics fenerated for consensus model (mathematical average of all models):")
	print(f"\t\tAccuracy : {a_all/len(models)}")
	print(f"\t\tPrecision : {p_all/len(models)}")
	print(f"\t\tRecall : {r_all/len(models)}")
	print(f"\t\tF1 Score : {f1_all/len(models)}")	
	print()


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


def combineLists(l1, l2):
	l3 = l1 + l2
	return l3
