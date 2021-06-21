#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:49:02 2021

@author: manisha
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utilities import *
from models import *


def plotComparisonGraph(modelsUsed,avg_models):
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

def plotBoxGrouped(data,columns,a,b):
    all_feat = []
    data['compoundType'] = np.where(data['Decision'] == 0, 'Non-Substrates', 'Substrates')
    # columns3=['C Log P','ROTB','nON','nOHNH']
    # columns4 = ['TPSA','Molecular Weight','Molecular Volume']
    
    for feat in columns:
        substrates_feat = data[data['compoundType'] == 'Substrates'][feat]
        nonsubstrates_feat = data[data['compoundType'] == 'Non-Substrates'][feat]
        
        all_feat.append(substrates_feat)
        all_feat.append(nonsubstrates_feat)
        labels = data['compoundType'].unique()
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 14)
    
   # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#2C7BB6', label='Substrates')
    plt.plot([], c='#D7191C', label='Non-Substrates')
    plt.legend()


    # Add major gridlines in the y-axis
    ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
    # Set plot title
    ax.set_title('Distribution of Molecules by Features and Decision value')    
    
    # Set the colors for each distribution
    colors = ['#426A8C','#73020C']
    colors_substrates = dict(color=colors[0])
    colors_nonsubstrates = dict(color=colors[1])
        
    i = 0
    for feat in columns: 
        ax.boxplot(all_feat[i], positions=[i], labels=[feat], boxprops=dict(facecolor = "blue"), medianprops=colors_substrates, whiskerprops=colors_substrates, capprops=colors_substrates, flierprops=dict(markeredgecolor=colors[0]),patch_artist=True,widths = 0.35)
        ax.boxplot(all_feat[i+1], positions=[i+0.5], boxprops=dict(facecolor = "red"), whiskerprops=colors_nonsubstrates, capprops=colors_nonsubstrates, flierprops=dict(markeredgecolor=colors[1]), patch_artist=True,widths = 0.35)
        i = i+2
        
    
    ticks = columns
    plt.xticks(range(0, len(ticks)*2, 2), ticks)
    plt.xlim(-0.5, len(ticks)*2)
    #plt.ylim(-2, 18)
    #plt.ylim(0, 800)
    plt.ylim(a,b)
    plt.tight_layout()
    plt.show()


def plotPieChart(data):
    c0 = 0
    c1 = 1
    for i in range(0, len(data['Predicted_Output'])):
    	if data['Predicted_Output'][i] == 0:
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
    