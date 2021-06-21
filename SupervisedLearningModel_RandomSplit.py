#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:24:55 2021

@author: surbhi
"""

import matplotlib.pyplot as plt
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from chefboost import Chefboost as chef
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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
file_datasheet = 'DataSheets/Dataset.csv'
file_validationSheet = 'DataSheets/Validation2.csv'


def fetchDataset(sheet,cols):
    dataset11 = pd.read_csv(sheet)
    dataset = dataset11[cols]
    dataset.head()
    return dataset
    
    
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


def calculateAverageAccuracyOfModels():
    avg_models.append(sum(list_knn) / len(list_knn))
    avg_models.append(sum(list_rf) / len(list_rf))
    avg_models.append(sum(list_dt) / len(list_dt))
    avg_models.append(sum(list_naive) / len(list_naive))  
    avg_models.append(sum(list_xgboost) / len(list_xgboost))
    avg_models.append(sum(list_svm) / len(list_svm))
    print("\nAvg Individual Accuracies :")
    for i in range (6):
        print(modelsUsed[i], " --> " , avg_models[i])    
    avg_accuracy = sum(avg_models)/len(avg_models)
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


def plotComparisonGraph():
    plt.plot(modelsUsed, avg_models, color='blue', linewidth = 3,marker='o', markerfacecolor='blue', markersize=12)
    plt.xlabel('Supervised Machine Learning Models')
    plt.ylabel('Accuracy')
    plt.title("Accuracy plot for Supervised Machine Learning Models")
    plt.gcf().autofmt_xdate()
    plt.show()
    return


def plotScatterPlots(data):
    plotFeatures(data[['Index','C Log P','Predicted_Output']],'C Log P','C Log P Distribution')
    plotFeatures(data[['Index','TPSA','Predicted_Output']],'TPSA','TPSA Distribution')
    plotFeatures(data[['Index','Molecular Weight','Predicted_Output']],'Molecular Weight','Molecular Weight Distribution')
    plotFeatures(data[['Index','nON','Predicted_Output']],'nON','nON Distribution')
    plotFeatures(data[['Index','nOHNH','Predicted_Output']],'nOHNH','nOHNH Distribution')
    plotFeatures(data[['Index','Molecular Volume','Predicted_Output']],'Molecular Volume','Molecular Volume Distribution')
    return

    
def plotFeatures(df,yaxis,tit):
    col = []
    for index,row in df.iterrows():
        if row['Predicted_Output'] == 0:
            col.append('red')
        else:
            col.append('blue')
    
    df.plot(x = 'Index',y = yaxis, kind = 'scatter', color=col)
    plt.xlabel('Molecule')
    plt.ylabel(yaxis)
    plt.title(tit)
    plt.plot([], c='blue', label='Substrates')
    plt.plot([], c='red', label='Non-Substrates')
    plt.legend()
    plt.show()
    return


def plotBoxIndividual(data,title):
    data['compoundType'] = np.where(data['Decision'] == 0, 'Non-Substrates', 'Substrates')
    plotWhiskerByFeatures(data,'C Log P',title)
    plotWhiskerByFeatures(data,'TPSA',title)
    plotWhiskerByFeatures(data,'Molecular Weight',title)
    plotWhiskerByFeatures(data,'nON',title)
    plotWhiskerByFeatures(data,'nOHNH',title)
    plotWhiskerByFeatures(data,'ROTB',title)
    plotWhiskerByFeatures(data,'Molecular Volume',title)  
    return


def plotWhiskerByFeatures(data,feat,title):
    substrates_feat = data[data['compoundType'] == 'Substrates'][feat]
    nonsubstrates_feat = data[data['compoundType'] == 'Non-Substrates'][feat]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    plt.xticks(fontsize= 24)
    plt.yticks(fontsize= 14)
    ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_title(title.format(feat), fontsize=24)
    dataset = [substrates_feat, nonsubstrates_feat]
    labels = data['compoundType'].unique()
    colors = ['#426A8C','#73020C']
    colors_substrates = dict(color=colors[0])
    colors_nonsubstrates = dict(color=colors[1])
    ax.boxplot(dataset[0], positions=[1], labels=[labels[0]], boxprops=dict(facecolor = "blue"), medianprops=colors_substrates, whiskerprops=colors_substrates, capprops=colors_substrates, flierprops=dict(markeredgecolor=colors[0]),patch_artist=True)
    ax.boxplot(dataset[1], positions=[2], labels=[labels[1]], boxprops=dict(facecolor = "red"), whiskerprops=colors_nonsubstrates, capprops=colors_nonsubstrates, flierprops=dict(markeredgecolor=colors[1]),patch_artist=True)
    plt.show()
    return


def plotPieChart(data):
    c0 = 0
    c1 = 1
    for i in range(0, len(data['Decision'])):
    	if data['Decision'][i] == 0:
    		c0 = c0 + 1
    	else:
    		c1 = c1 + 1
    print(c0 , " " , c1)
    y = np.array([c0,c1])
    mylabels = ["Non-Substrates", "Substrates"]
    mycolors = ["red", "blue"]
    plt.pie(y, labels = mylabels, colors = mycolors,autopct='%.0f%%')
    plt.show() 
    return
    
    
if __name__=="__main__":
    print('\nStarting Execution')
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    #working on Training dataset
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    
    dataset = fetchDataset(file_datasheet,['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Decision'])
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:, -1:].values
    X,Y = shuffle(X,Y)
    X_c = np.append(X,Y,axis=1)
    
    #handle missing values and replace with mean value
    X = handleMissingValues(X)
    X_c = handleMissingValues(X_c)
    
    #split dataset into train set and test set
    X_train,X_test,Y_train,Y_test = splitDatasetToTrainTest(X,Y)
    X_train_c,X_test_c,Y_train_c,Y_test_c = splitDatasetToTrainTest(X_c,Y)
    
    #take backup of X_test, and perform feature scaling
    X_backup = X_test
    sc_X,X_train,X_test = featureScaleDataSet(X_train,X_test)
    
    #Predict values using KNN Algorithm
    knn = knnTrainer(X_train, Y_train)
    y_pred_knn = knnPredictor(knn,X_test)
    list_knn.append(accuracy_score(Y_test, y_pred_knn))
    print("\nY_pred_KNN : ",y_pred_knn)
    
    #predict values using Random Forest Algorithm
    regressor = randomForestTrainer(X_train,Y_train)
    y_pred_rf = randomForestPredictor(regressor,X_test)
    list_rf.append(accuracy_score(Y_test, y_pred_rf))
    print("\nY_pred_RF : ",y_pred_rf)
    
    #Predict values using Decision tree, entropy classifier
    clf_gini = decisionTreeEntropyTrainer(X_train,Y_train)
    y_pred_dt = decisionTreeEntropyPredictor(clf_gini,X_test)
    list_dt.append(accuracy_score(Y_test, y_pred_dt))
    print("\nY_pred_DT : ",y_pred_dt)
    
    
    """
    #Predict values using C-4.5 single decision tree
    model1 = c4_5_Trainer(X_train_c)
    y_pred_c = c4_5_Predictor(model1,X_test_c)
    list_c.append(accuracy_score(Y_test_c, y_pred_c))
    print("Y_pred_c : ",y_pred_c)
    """
    
    
    #Predice values using Naive Bayes Classifier
    naive = naiveBayesTrainer(X_train,Y_train)
    y_pred_naive = naiveBayesPredictor(naive,X_test)
    list_naive.append(accuracy_score(Y_test, y_pred_naive))
    print("\nY_pred_Naive : ",y_pred_naive)
    
    
    #Predice values using XG-Boost
    xg = xgBoostTrainer(X_train,Y_train)
    y_pred_xg = xgBoostPredictor(xg,X_test)
    list_xgboost.append(accuracy_score(Y_test, y_pred_xg))
    print("\nY_pred_XG-Boost : ",y_pred_xg)
    
    
    #Predice values using SVM
    svm = svmTrainer(X_train,Y_train)
    y_pred_svm = svmPredictor(svm,X_test)
    list_svm.append(accuracy_score(Y_test, y_pred_svm))
    print("\nY_pred_SVM : ",y_pred_svm)
    
    
    #calculate average accuracies of all models together and combine all results in 2-D array
    predValues = appendPredictions(X_backup,y_pred_knn,y_pred_rf,y_pred_dt,y_pred_naive,y_pred_xg,y_pred_svm)
    avg_accuracy= calculateAverageAccuracyOfModels()
    print("\nAverage Accuracy for all models : ",avg_accuracy)
    
    #merge Y_test and predicted values with X_backup
    X_backup = np.append(X_backup, Y_test, axis=1)
    X_backup = np.append(X_backup,predValues,axis=1)
    
    #create dataframe from arraylist
    columns=['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Y_Actual','Predict_KNN','Predict_RF','Predict_DT','Predict_Naive','Predict_XG-Boost','Predict_SVM']
    data = getDataFrameFromNParray(X_backup,columns)
    #filename = datetime.now().strftime('OutputSheet/CombinedSheet---%Y-%m-%d-%H-%M-%S.csv')
    #exportToCSV(data,filename)
    
    #plotComparisonGraph()
    #plotBoxIndividual(dataset,'{0} Distribution')
    
    

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
    data1 = data[column1]
    data1.head()
    column2 = column1[:-1]
    dataValidate = dataValidate[column2]
    dataValidate.head()
    X1 = data1.iloc[:,:-1].values
    Y1 = data1.iloc[:, -1:].values
    X2 = dataValidate.values
    
    #applying logistic regression on predicted values of all models
    mod = logisticRegressionTrainer(X1,Y1)
    finalPredictedValue = LogisticRegressionPredictor(mod,X2)
    
    #appending the output to backup and storing in excel
    finalPredictedValue = np.append(X_validatebackup1,finalPredictedValue,axis=1)
    colss = ['Index','C Log P','TPSA','Molecular Weight','nON','nOHNH','ROTB','Molecular Volume','Decision']
    outputDataset = getDataFrameFromNParray(finalPredictedValue,colss)
    filename = datetime.now().strftime('OutputSheet/PredictionSheet_ValidationSet---%Y-%m-%d-%H-%M-%S.csv')
    exportToCSV(outputDataset,filename)
    
    #plotBoxIndividual(outputDataset,'{0} Distribution on Validation Set')
    plotPieChart(outputDataset)
    print("Plots created successfully")
    print("\nAnalysis and prediction completed successfully. File has been created that conatins the input and its respective predicted values.\n")
