#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:39:37 2021

@author: manisha
"""

import numpy as np
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from utilities import *
from models import *
from graph_plots import *


#declaring accuracy lists for knn, random forest, decision tree, C4.5
list_knn = []
list_rf = []
list_dt = []
list_naive = []
list_xgboost = []
list_svm = []
avg_models = []
modelsUsed = ["KNN","Random Forest","Decision Tree","Naive Bayes","XG-Boost","SVM"]
config = {'algorithm': 'C4.5'}

#---------------- Datasets of 192 training and 142 validation ------------ #
file_datasheet2 = 'DataSheets/Dataset_unshuffled_192.csv' #not shuffled
file_datasheet = 'DataSheets/90.51_Sheet_192.csv' #shuffled
file_validationSheet = 'DataSheets/Validation2_142.csv'
# ----------------------------------------------------------------------- #


#---------------- Datasets of 172 training and 23 validation ------------ #
# file_datasheet_train = 'DataSheets/final/final_merged/92_Sheet_171.csv'
# file_datasheet_val = 'DataSheets/final/final_merged/ValidationSet_23.csv'
# file_datasheet = 'DataSheets/final/final_merged/92_Sheet_171.csv'
# file_validationSheet = 'DataSheets/final/final_merged/ValidationSet_23.csv'
# ----------------------------------------------------------------------- #


if __name__=="__main__":
    print('\nStarting Execution')
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #working on Training dataset
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    
    dataset = fetchDataset(file_datasheet,['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Decision'])
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:, -1:].values
    #X,Y = shuffle(X,Y)
    X_c = np.append(X,Y,axis=1)
    
    #handle missing values and replace with mean value
    X = handleMissingValues(X)
    X_c = handleMissingValues(X_c)
    
    # ------- Cross Validation -------- #
    
    X_df = pd.DataFrame(X)
    Y_df = pd.DataFrame(Y)
    
    # Definitions of different Cross validation Models
    outer_kfold = KFold(n_splits=10, random_state=50, shuffle=True)
    skf = StratifiedKFold(n_splits=5)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    final_scores = list()
    X_backup_test = np.empty((0,15))
    
    for train, test in skf.split(X_df,Y_df):
        X_train, X_test = X_df.iloc[train].values, X_df.iloc[test].values
        Y_train, Y_test = Y[train], Y[test]
        
        X_backup = X_test
        sc_X,X_train,X_test = featureScaleDataSet(X_train,X_test)
        
        #Predict values using KNN Algorithm
        knn = knnTrainer(X_train, Y_train)
        y_pred_knn = knnPredictor(knn,X_test)
        list_knn.append(accuracy_score(Y_test, y_pred_knn))
        
        #predict values using Random Forest Algorithm
        regressor = randomForestTrainer(X_train,Y_train)
        y_pred_rf = randomForestPredictor(regressor,X_test)
        list_rf.append(accuracy_score(Y_test, y_pred_rf))
        
        #Predict values using Decision tree, entropy classifier
        clf_gini = decisionTreeEntropyTrainer(X_train,Y_train)
        y_pred_dt = decisionTreeEntropyPredictor(clf_gini,X_test)
        list_dt.append(accuracy_score(Y_test, y_pred_dt))
        
        #Predice values using Naive Bayes Classifier
        naive = naiveBayesTrainer(X_train,Y_train)
        y_pred_naive = naiveBayesPredictor(naive,X_test)
        list_naive.append(accuracy_score(Y_test, y_pred_naive))
        
        #Predice values using XG-Boost
        xg = xgBoostTrainer(X_train,Y_train)
        y_pred_xg = xgBoostPredictor(xg,X_test)
        list_xgboost.append(accuracy_score(Y_test, y_pred_xg))
       
        #Predice values using SVM
        svm = svmTrainer(X_train,Y_train)
        y_pred_svm = svmPredictor(svm,X_test)
        list_svm.append(accuracy_score(Y_test, y_pred_svm))
        
        #calculate average accuracies of all models together and combine all results in 2-D array
        predValues = appendPredictions(X_backup,y_pred_knn,y_pred_rf,y_pred_dt,y_pred_naive,y_pred_xg,y_pred_svm)
        
        #merge Y_test and predicted values with X_backup
        X_backup = np.append(X_backup,predValues,axis=1)
        X_backup = np.append(X_backup, Y_test, axis=1)
        
        # X_backup_test ------ contains predictions of all k fold test sets
        # Will be used for training logistic regression model for stacking
        X_backup_test = np.append(X_backup_test, X_backup,axis = 0)
        
        
        print("heyyyy")
        #create dataframe from arraylist
        columns=['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Predict_KNN','Predict_RF','Predict_DT','Predict_Naive','Predict_XG-Boost','Predict_SVM','Y_Actual']
        data_kfold = getDataFrameFromNParray(X_backup,columns)
        filename = datetime.now().strftime('KfoldOutputSheet/CombinedSheet---%Y-%m-%d-%H-%M-%S.%f.csv')
        exportToCSV(data_kfold,filename)
        
        
    #create dataframe from arraylist
    columns=['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Predict_KNN','Predict_RF','Predict_DT','Predict_Naive','Predict_XG-Boost','Predict_SVM','Y_Actual']
    data = getDataFrameFromNParray(X_backup_test,columns)
    filename = datetime.now().strftime('OutputSheet/CombinedSheet_withFeatures&Predictions---%Y-%m-%d-%H-%M-%S.csv')
    exportToCSV(data,filename)
    
    #print(X_backup_test)
    columns=['Index','Predict_KNN','Predict_RF','Predict_DT','Predict_Naive','Predict_XG-Boost','Predict_SVM','Y_Actual']
    data_logReg = getDataFrameFromNParray(data,columns)
    filename = datetime.now().strftime('TrainSheetForLogReg/KFoldCombinedSheet---%Y-%m-%d-%H-%M-%S.csv')
    exportToCSV(data_logReg,filename)
        
    avg_accuracy= calculateAverageAccuracyOfModels(avg_models,list_knn,list_rf,list_dt,list_naive,list_xgboost,list_svm)
    
    
    features1 =['C Log P','ROTB','nON','nOHNH']
    features2 = ['TPSA','Molecular Weight','Molecular Volume']
    
    plotComparisonGraph(modelsUsed, avg_models)
    plotBoxIndividual(dataset,'{0} Distribution')
    plotBoxGrouped(dataset,features1,-4,18)
    plotBoxGrouped(dataset,features2,-4,800)
    


    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #working on Test dataset
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    
    dataVal = fetchDataset(file_validationSheet,['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume'])
    X_validate = dataVal.values
    
    #handle missing values and replace with mean values
    X_validate = np.nan_to_num(X_validate)
    
    #Take backup of validation dataset
    X_validatebackup = X_validate
    X_validatebackup1 = X_validate
    
    #feature scaling on validation set
    X_validate = sc_X.fit_transform(X_validate)
    
    #predicting values using various models
    y_predValidate_knn = knnPredictor(knn, X_validate) 
    y_predValidate_rf = randomForestPredictor(regressor,X_validate)
    y_predValidate_dt = decisionTreeEntropyPredictor(clf_gini,X_validate)
    y_predValidate_naive = naiveBayesPredictor(naive,X_validate)
    y_predValidate_xg = xgBoostPredictor(xg,X_validate)
    y_predValidate_svm = svmPredictor(svm,X_validate)
    
    #append predictions and create dataframe
    pred_Validate_Values = appendPredictions(X_validatebackup, y_predValidate_knn, y_predValidate_rf, y_predValidate_dt,y_predValidate_naive,y_predValidate_xg,y_predValidate_svm)
    X_validatebackup = np.append(X_validatebackup,pred_Validate_Values,axis=1)
    col_Validate = ['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Predict_KNN','Predict_RF','Predict_DT','Predict_Naive','Predict_XG-Boost','Predict_SVM']
    dataValidate = getDataFrameFromNParray(X_validatebackup, col_Validate)
    #filename = datetime.now().strftime('OutputSheet/TestSheet---%Y-%m-%d-%H-%M-%S.csv')
    #exportToCSV(dataValidate,filename)


   #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #Applying Logistic Regression on predictions of various models
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #create training and test dataset for logictic regression
    column1 = ['Predict_KNN','Predict_RF','Predict_DT','Predict_Naive','Predict_XG-Boost','Predict_SVM','Y_Actual']
    #Traindataset = fetchDataset('DataSheets/final/kfold_logReg.csv',column1)
    #Traindataset = fetchDataset('TrainSheetForLogReg/KFoldCombinedSheet1.csv',column1)
    Traindataset = data_logReg[column1]
    Traindataset.head()
    column2 = ['Predict_KNN','Predict_RF','Predict_DT','Predict_Naive','Predict_XG-Boost','Predict_SVM']
    dataValidate = dataValidate[column2]
    dataValidate.head()
    X1 = Traindataset.iloc[:,:-1].values
    Y1 = Traindataset.iloc[:, -1:].values
    X2 = dataValidate.values
    
    #applying logistic regression on predicted values of all models
    mod = logisticRegressionTrainer(X1,Y1)
    finalPredictedValue = LogisticRegressionPredictor(mod,X2)
    
    #appending the output to backup and storing in excel
    finalPredictedValue = np.append(X_validatebackup1,finalPredictedValue,axis=1)
    colss = ['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Predicted_Output']
    outputDataset = getDataFrameFromNParray(finalPredictedValue,colss)
    filename = datetime.now().strftime('OutputSheet/PredictionSheet_ValidationSet_kfold---%Y-%m-%d-%H-%M-%S.csv')
    exportToCSV(outputDataset,filename)
    
    plotPieChart(outputDataset)
    print("Plots created successfully")
    print("\nAnalysis and prediction completed successfully. File has been created that conatins the input and its respective predicted values.\n")