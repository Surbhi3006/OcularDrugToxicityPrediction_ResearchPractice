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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
import pandas as pd

from utilities import *
from models import *
from graph_plots import *

# declaring accuracy lists for knn, random forest, decision tree, C4.5
modelsUsed = ["KNN", "Random_Forest", "Decision_Tree", "Naive_Bayes", "XG_Boost", "SVM"]
config = {'algorithm': 'C4.5'}

# ---------------- Datasets of 192 training and 513 validation ------------ #
file_datasheet = 'inputSheets/OCT1-Trainingdataset-14Desc-12Aug2022.csv'
file_validationSheet = 'inputSheets/Ocutox-Validationdatset-14Desc-12Aug2022.csv'
# ----------------------------------------------------------------------- #


if __name__ == "__main__":
    print('\nStarting Execution')

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # working on Training dataset
    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    accuracy, precision, recall, f1 = createMetrics(modelsUsed)
    dataset = fetchDataset(file_datasheet, fetchColumns(file_datasheet))
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1:].values
    X, Y = shuffle(X, Y)
    X_c = np.append(X, Y, axis=1)

    # handle missing values and replace with mean value
    X = handleMissingValues(X)
    X_c = handleMissingValues(X_c)

    # ------- Cross Validation -------- #
    X_df = pd.DataFrame(X)
    Y_df = pd.DataFrame(Y)

    # Definitions of different Cross validation Models
    outer_kfold = KFold(n_splits=10, random_state=50, shuffle=True)
    skf = StratifiedKFold(n_splits=5)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    s = len(X[0]) + 7
    X_backup_test = np.empty((0, s))
    fold = 1

    for train, test in skf.split(X_df, Y_df):
        X_train, X_test = X_df.iloc[train].values, X_df.iloc[test].values
        Y_train, Y_test = Y[train], Y[test]

        X_backup = X_test
        sc_X, X_train, X_test = featureScaleDataSet(X_train, X_test)

        # Predict values using KNN Algorithm
        knn = knnTrainer(X_train, Y_train)
        y_pred_knn = knnPredictor(knn, X_test)
        printMetrics("KNN", Y_test, y_pred_knn, accuracy, precision, recall, f1, fold)

        # predict values using Random Forest Algorithm
        regressor = randomForestTrainer(X_train, Y_train)
        y_pred_rf = randomForestPredictor(regressor, X_test)
        printMetrics("Random_Forest", Y_test, y_pred_rf, accuracy, precision, recall, f1, fold)

        # Predict values using Decision tree, entropy classifier
        clf_gini = decisionTreeEntropyTrainer(X_train, Y_train)
        y_pred_dt = decisionTreeEntropyPredictor(clf_gini, X_test)
        printMetrics("Decision_Tree", Y_test, y_pred_dt, accuracy, precision, recall, f1, fold)

        # Predice values using Naive Bayes Classifier
        naive = naiveBayesTrainer(X_train, Y_train)
        y_pred_naive = naiveBayesPredictor(naive, X_test)
        printMetrics("Naive_Bayes", Y_test, y_pred_naive, accuracy, precision, recall, f1, fold)

        # Predice values using XG-Boost
        xg = xgBoostTrainer(X_train, Y_train)
        y_pred_xg = xgBoostPredictor(xg, X_test)
        printMetrics("XG_Boost", Y_test, y_pred_xg, accuracy, precision, recall, f1, fold)

        # Predice values using SVM
        svm = svmTrainer(X_train, Y_train)
        y_pred_svm = svmPredictor(svm, X_test)
        printMetrics("SVM", Y_test, y_pred_svm, accuracy, precision, recall, f1, fold)

        # calculate average accuracies of all models together and combine all results in 2-D array
        predValues = appendPredictions(X_backup, y_pred_knn, y_pred_rf, y_pred_dt, y_pred_naive, y_pred_xg, y_pred_svm)

        # merge Y_test and predicted values with X_backup
        X_backup = np.append(X_backup, predValues, axis=1)
        X_backup = np.append(X_backup, Y_test, axis=1)

        # X_backup_test ------ contains predictions of all k fold test sets
        # Will be used for training logistic regressionpn model for stacking
        X_backup_test = np.append(X_backup_test, X_backup, axis=0)

        # create dataframe from arraylist
        columns = combineLists(combineLists(fetchColumns(file_datasheet)[:-1], create_model_columns(modelsUsed)), ['Y_Actual'])
        data_kfold = getDataFrameFromNParray(X_backup, columns)
        # filename = datetime.now().strftime('outSheet/KfoldOutputSheet/CombinedSheet---%Y-%m-%d-%H-%M-%S.%f.csv')
        # exportToCSV(data_kfold,filename)
        
        fold = fold + 1
    
    print(f"\n\nGenerating Accuracy scores for each model :")
    df = pd.DataFrame.from_dict(accuracy, orient='index')
    print(df.to_markdown())
    
    print(f"\n\nGenerating Precision scores for each model :")
    df = pd.DataFrame.from_dict(precision, orient='index')
    print(df.to_markdown())

    print(f"\n\nGenerating Recall scores for each model :")
    df = pd.DataFrame.from_dict(recall, orient='index')
    print(df.to_markdown())

    print(f"\n\nGenerating F1 scores for each model :")
    df = pd.DataFrame.from_dict(f1, orient='index')
    print(df.to_markdown())

    # create dataframe from arraylist
    columns = combineLists(combineLists(fetchColumns(file_datasheet)[:-1], create_model_columns(modelsUsed)), ['Y_Actual'])
    data = getDataFrameFromNParray(X_backup_test, columns)
    # filename = datetime.now().strftime('outSheet/OutputSheet/CombinedSheet_withFeatures&Predictions---%Y-%m-%d-%H-%M-%S.csv')
    # exportToCSV(data,filename)

    columns = combineLists(combineLists([fetchColumns(file_datasheet)[0]], create_model_columns(modelsUsed)), ['Y_Actual'])
    data_logReg = getDataFrameFromNParray(data, columns)
    # filename = datetime.now().strftime('outSheet/TrainSheetForLogReg/KFoldCombinedSheet---%Y-%m-%d-%H-%M-%S.csv')
    # exportToCSV(data_logReg,filename)

    calculateStatsForModels(modelsUsed, accuracy, precision, recall, f1)

    # features1 =['C Log P','ROTB','nON','nOHNH']
    # features2 = ['TPSA','Molecular Weight','Molecular Volume']

    # plotComparisonGraph(modelsUsed, avg_models)
    # plotBoxIndividual(dataset,'{0} Distribution')
    # plotBoxGrouped(dataset,features1,-4,18)
    # plotBoxGrouped(dataset,features2,-4,800)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # working on Test dataset
    # --------------------------------------------------------------------------------------------------------------------------------------------------------

    dataVal = fetchDataset(file_validationSheet, fetchColumns(file_validationSheet))
    X_validate = dataVal.values

    # handle missing values and replace with mean values
    X_validate = np.nan_to_num(X_validate)

    # Take backup of validation dataset
    X_validatebackup = X_validate
    X_validatebackup1 = X_validate

    # feature scaling on validation set
    X_validate = sc_X.fit_transform(X_validate)

    # predicting values using various models
    y_predValidate_knn = knnPredictor(knn, X_validate)
    y_predValidate_rf = randomForestPredictor(regressor, X_validate)
    y_predValidate_dt = decisionTreeEntropyPredictor(clf_gini, X_validate)
    y_predValidate_naive = naiveBayesPredictor(naive, X_validate)
    y_predValidate_xg = xgBoostPredictor(xg, X_validate)
    y_predValidate_svm = svmPredictor(svm, X_validate)

    # ANN on a train:test split dataset, used to print the accuracy and confusion matrix
    X_ann, Y_ann, X_test_ann, Y_test_ann = fetchTrainDatasetAnn(file_datasheet)
    ann_classifier = ANNTrainer(X_ann, Y_ann)
    y_predValidate_ann = ANNPredictor(ann_classifier, X_test_ann)

    # confusion matrix
    cm = confusion_matrix(Y_test_ann, y_predValidate_ann)
    acc_score_ann = accuracy_score(Y_test_ann, y_predValidate_ann)
    pr_score_ann = precision_score(Y_test_ann, y_predValidate_ann)
    rc_score_ann = recall_score(Y_test_ann, y_predValidate_ann)
    f1_score_ann = f1_score(Y_test_ann, y_predValidate_ann)
    
    # ANN used to give the predictions for the validation dataset
    X_ann, Y_ann, X_test_ann = fetchDatasetAnn(file_datasheet, file_validationSheet)
    ann_classifier = ANNTrainer(X_ann, Y_ann)
    y_predValidate_ann = ANNPredictor(ann_classifier, X_test_ann)
    
    print(f"\nGenerating metrics for ANN :")
    print("\t\tConfusion Matrix: \n", cm)
    print("\t\tAccuracy: ", acc_score_ann)
    print("\t\tPrecision: ", pr_score_ann)
    print("\t\tRecall: ", rc_score_ann)
    print("\t\tF1 Score: ", f1_score_ann)

    # append predictions and create dataframe
    pred_Validate_Values = appendPredictions(X_validatebackup, y_predValidate_knn, y_predValidate_rf, y_predValidate_dt,
                                             y_predValidate_naive, y_predValidate_xg, y_predValidate_svm)
    X_validatebackup = np.append(X_validatebackup, pred_Validate_Values, axis=1)
    col_Validate = combineLists(fetchColumns(file_validationSheet), create_model_columns(modelsUsed))
    dataValidate = getDataFrameFromNParray(X_validatebackup, col_Validate)
    # filename = datetime.now().strftime('outSheet/OutputSheet/TestSheet---%Y-%m-%d-%H-%M-%S.csv')
    # exportToCSV(dataValidate,filename)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # Applying Logistic Regression on predictions of various models
    # --------------------------------------------------------------------------------------------------------------------------------------------------------

    # create training and test dataset for logictic regression
    column1 = combineLists(create_model_columns(modelsUsed), ['Y_Actual'])
    # Traindataset = fetchDataset('inputSheets/final/kfold_logReg.csv',column1)
    # Traindataset = fetchDataset('outSheet/TrainSheetForLogReg/KFoldCombinedSheet1.csv',column1)
    Traindataset = data_logReg[column1]
    Traindataset.head()
    column2 = create_model_columns(modelsUsed)
    dataValidate = dataValidate[column2]
    dataValidate.head()
    X1 = Traindataset.iloc[:, :-1].values
    Y1 = Traindataset.iloc[:, -1:].values
    X2 = dataValidate.values

    # applying logistic regression on predicted values of all models
    mod = logisticRegressionTrainer(X1, Y1)
    finalPredictedValue = logisticRegressionPredictor(mod, X2)

    # Prediction using Artificial Neural Network

    # appending the output to backup and storing in excel
    prediction_cols = ['Predicted_Output_supervisedModel', 'Prediction_ANNModel']
    finalPredictedValue = np.append(X_validatebackup1, finalPredictedValue, axis=1)
    finalPredictedValue = np.append(finalPredictedValue, y_predValidate_ann, axis=1)
    colss = combineLists(fetchColumns(file_validationSheet), prediction_cols)
    outputDataset = getDataFrameFromNParray(finalPredictedValue, colss)
    filename = datetime.now().strftime(
        'outSheet/OutputSheet/PredictionSheet_ValidationSet_kfold---%Y-%m-%d-%H-%M-%S.csv')
    exportToCSV(outputDataset, filename)

    # confusion compare matrix for supervised and ANN predictions
    print(f"\nGenerating metrics by comaring predictions by Supervised learning model and ANN :")
    cmp_conf_mat = confusion_matrix(outputDataset[prediction_cols[0]], outputDataset[prediction_cols[1]])
    cmp_acc_score = accuracy_score(outputDataset[prediction_cols[0]], outputDataset[prediction_cols[1]])
    cmp_pr_score = precision_score(outputDataset[prediction_cols[0]], outputDataset[prediction_cols[1]])
    cmp_rc_score = recall_score(outputDataset[prediction_cols[0]], outputDataset[prediction_cols[1]])
    cmp_f1_score = f1_score(outputDataset[prediction_cols[0]], outputDataset[prediction_cols[1]])
    print(f"\t\tConfusion Matrix: \n{cmp_conf_mat}")
    print(f"\t\tAccuracy: {cmp_acc_score}")
    print(f"\t\tPrecision: {cmp_pr_score}")
    print(f"\t\tRecall: {cmp_rc_score}")
    print(f"\t\tF1 Score: {cmp_f1_score}")

    # plotPieChart(outputDataset)
    print("Plots created successfully")
    print(
        f"\nAnalysis and prediction completed successfully.\nFile has been created and conatins the input with corresponding predictions.\nFile Path - {os.path.relpath(filename)}")
