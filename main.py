#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:39:37 2021

@author: surbhi sharma
"""

import os
import numpy as np
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense

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
modelsUsed = ["KNN","Random_Forest","Decision_Tree","Naive_Bayes","XG_Boost","SVM"]
config = {'algorithm': 'C4.5'}

#---------------- Datasets of 192 training and 513 validation ------------ #
file_datasheet = 'inputSheets/90.51_Sheet_192.csv' #shuffled
file_validationSheet = 'inputSheets/ScreeningData_Oct11.csv'
# ----------------------------------------------------------------------- #


if __name__=="__main__":
    print('\nStarting Execution')
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #working on Training dataset
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    
    dataset = fetchDataset(file_datasheet, fetchColumns(file_datasheet))
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
        predValues = appendPredictions(X_backup, y_pred_knn, y_pred_rf, y_pred_dt, y_pred_naive, y_pred_xg, y_pred_svm)
        
        #merge Y_test and predicted values with X_backup
        X_backup = np.append(X_backup, predValues, axis=1)
        X_backup = np.append(X_backup, Y_test, axis=1)
        
        # X_backup_test ------ contains predictions of all k fold test sets
        # Will be used for training logistic regression model for stacking
        X_backup_test = np.append(X_backup_test, X_backup,axis = 0)
        
        
        #create dataframe from arraylist
        columns = combineLists(combineLists(fetchColumns(file_datasheet)[:-1], create_model_columns(modelsUsed)), ['Y_Actual'])
        data_kfold = getDataFrameFromNParray(X_backup,columns)
        #filename = datetime.now().strftime('outSheet/KfoldOutputSheet/CombinedSheet---%Y-%m-%d-%H-%M-%S.%f.csv')
        #exportToCSV(data_kfold,filename)


    #create dataframe from arraylist
    columns = combineLists(combineLists(fetchColumns(file_datasheet)[:-1], create_model_columns(modelsUsed)), ['Y_Actual'])
    data = getDataFrameFromNParray(X_backup_test,columns)
    #filename = datetime.now().strftime('outSheet/OutputSheet/CombinedSheet_withFeatures&Predictions---%Y-%m-%d-%H-%M-%S.csv')
    #exportToCSV(data,filename)

    columns = combineLists(combineLists([fetchColumns(file_datasheet)[0]], create_model_columns(modelsUsed)), ['Y_Actual'])
    data_logReg = getDataFrameFromNParray(data,columns)
    #filename = datetime.now().strftime('outSheet/TrainSheetForLogReg/KFoldCombinedSheet---%Y-%m-%d-%H-%M-%S.csv')
    #exportToCSV(data_logReg,filename)
        
    avg_accuracy= calculateAverageAccuracyOfModels(avg_models,list_knn,list_rf,list_dt,list_naive,list_xgboost,list_svm)
    
    
    #features1 =['C Log P','ROTB','nON','nOHNH']
    #features2 = ['TPSA','Molecular Weight','Molecular Volume']
    
    plotComparisonGraph(modelsUsed, avg_models)
    #plotBoxIndividual(dataset,'{0} Distribution')
    #plotBoxGrouped(dataset,features1,-4,18)
    #plotBoxGrouped(dataset,features2,-4,800)
    


    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #working on Test dataset
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    
    dataVal = fetchDataset(file_validationSheet, fetchColumns(file_validationSheet))
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
    
    # ANN on a train:test split dataset, used to print the accuracy and confusion matrix
    X_ann, Y_ann, X_test_ann, Y_test_ann = fetchTrainDatasetAnn(file_datasheet)
    ann_classifier = ANNTrainer(X_ann, Y_ann)
    y_predValidate_ann = ANNPredictor(ann_classifier,X_test_ann)

    #confusion matrix
    cm = confusion_matrix(Y_test_ann, y_predValidate_ann)
    acc_score_ann = accuracy_score(Y_test_ann,y_predValidate_ann)
    
    # ANN used to give the predictions for the validation dataset
    X_ann, Y_ann, X_test_ann = fetchDatasetAnn(file_datasheet,file_validationSheet)
    ann_classifier = ANNTrainer(X_ann, Y_ann)
    y_predValidate_ann = ANNPredictor(ann_classifier,X_test_ann)
    
    print("Confusion Matrix of ANN: \n",cm)
    print("Accuracy Score for ANN: \n",acc_score_ann)

    
    #append predictions and create dataframe
    pred_Validate_Values = appendPredictions(X_validatebackup, y_predValidate_knn, y_predValidate_rf, y_predValidate_dt,y_predValidate_naive,y_predValidate_xg,y_predValidate_svm)
    X_validatebackup = np.append(X_validatebackup,pred_Validate_Values,axis=1)
    col_Validate = combineLists(fetchColumns(file_validationSheet), create_model_columns(modelsUsed))
    dataValidate = getDataFrameFromNParray(X_validatebackup, col_Validate)
    #filename = datetime.now().strftime('outSheet/OutputSheet/TestSheet---%Y-%m-%d-%H-%M-%S.csv')
    #exportToCSV(dataValidate,filename)


   #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #Applying Logistic Regression on predictions of various models
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    
    #create training and test dataset for logictic regression
    column1 = combineLists(create_model_columns(modelsUsed), ['Y_Actual'])
    #Traindataset = fetchDataset('inputSheets/final/kfold_logReg.csv',column1)
    #Traindataset = fetchDataset('outSheet/TrainSheetForLogReg/KFoldCombinedSheet1.csv',column1)
    Traindataset = data_logReg[column1]
    Traindataset.head()
    column2 = create_model_columns(modelsUsed)
    dataValidate = dataValidate[column2]
    dataValidate.head()
    X1 = Traindataset.iloc[:,:-1].values
    Y1 = Traindataset.iloc[:, -1:].values
    X2 = dataValidate.values
    
    #applying logistic regression on predicted values of all models
    mod = logisticRegressionTrainer(X1,Y1)
    finalPredictedValue = logisticRegressionPredictor(mod,X2)
    
    
    
    # Prediction using Artificial Neural Network
    
    
    #appending the output to backup and storing in excel
    prediction_cols = ['Predicted_Output_supervisedModel','Prediction_ANNModel']
    finalPredictedValue = np.append(X_validatebackup1,finalPredictedValue,axis=1)
    finalPredictedValue = np.append(finalPredictedValue,y_predValidate_ann,axis=1)
    colss = combineLists(fetchColumns(file_validationSheet), prediction_cols)
    outputDataset = getDataFrameFromNParray(finalPredictedValue,colss)
    filename = datetime.now().strftime('outSheet/OutputSheet/PredictionSheet_ValidationSet_kfold---%Y-%m-%d-%H-%M-%S.csv')
    exportToCSV(outputDataset,filename)
    
    #compare confusion matrix for supervised and ANN predictions
    compare_pred = confusion_matrix(outputDataset[prediction_cols[0]], outputDataset[prediction_cols[1]])
    print(f"Confusion matrix for final predictions for supervised learning model and ANN : \n{compare_pred}")
    compare_acc_score_ann = accuracy_score(outputDataset[prediction_cols[0]], outputDataset[prediction_cols[1]])
    print(f"Accuracy score for final predictions for supervised learning model and ANN : \n{compare_acc_score_ann}")
    

    #plotPieChart(outputDataset)
    print("Plots created successfully")
    print(f"\nAnalysis and prediction completed successfully.\nFile has been created and conatins the input with corresponding predictions.\nFile Path - {os.path.relpath(filename)}")
