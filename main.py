# -*- coding: utf-8 -*-
"""
@author: Nehal
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from Linear import LinearModels
from Ensemble import EnsembleModels
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
def main():
    
    filepath = f'Automobile.csv'
    
    df = pd.read_csv(filepath)
    
#    Data pre-processing
    df.drop(['X','Y','Index_','ACCNUM','DATE','STREET1','STREET2','OFFSET','LATITUDE','LONGITUDE', 
             'ACCLOC','YEAR','TIME','INVAGE','IMPACTYPE','INVTYPE','INITDIR','DRIVACT','DRIVCOND','CYCLISTYPE','PEDCOND','PEDACT',
             'PEDTYPE','CYCACT','CYCCOND','TRSN_CITY_','EMERG_VEH','Hood_ID','ObjectId','Division'],axis=1,inplace=True)
    df.dropna(axis = 1, inplace=True)
    df.replace(['Yes',' '],[1,0],inplace=True)
    
    y = df['PEDESTRIAN']
    df.drop('PEDESTRIAN',axis=1,inplace=True)
    #encoding labels to numbers
    X = pd.get_dummies(df)
    
#    Feature selection
    feat = feature_selection(X,y)
    
    print(feat['Features'].values)
    
    X_f = X
    for f in X.columns:
        if f not in feat['Features'].values:
#            print(f)
            X_f = X_f.drop(f,axis=1)
            
#   Remove this feature as it is overpowering feature for dataset.
#    X_f.drop('IMPACTYPE_Pedestrian Collisions',axis=1,inplace=True)
#    X_f has new features for training
    X_train, X_test, y_train, y_test = train_test_split(X_f, y, test_size =0.20, random_state=89 )

    print("Result using Decision Tree Classification:\n")
    y_decision, scores_dt = decisionTree(X_train,y_train,X_test)
    printReport(y_test,y_decision)
    
    print("Result using KNN Classification:\n")
    y_knn, scores_knn = kNeighbors(X_train,y_train,X_test,y_test)
    printReport(y_test,y_knn)
    
    print("Result using Rule based Classification:\n")
    y_dc, scores_dc = dummyClass(X_train,y_train,X_test)
    printReport(y_test,y_dc)


    LinearModels(X_train,y_train,X_test, y_test)
    EnsembleModels(X_train,y_train,X_test, y_test)
    
def printReport(y_test,y_predict):
    #finding the accuracy of results
    acc = accuracy_score(y_test, y_predict)
    print("Accuracy comes out to be: ",acc)
    #calculating precision and recall
    print(classification_report(y_test,y_predict))
    
    #creating confusion matrix
    print("Confusion Matrix for the Classification result:")
    print(pd.DataFrame(
            confusion_matrix(y_test, y_predict),
            columns =['Predicted Pedestrian Safe','Predicted Pedestrian Affected'],
            index=['True Pedestrian Safe','True Pedestrian Affected']))
    
    fpr, tpr , thresholds = roc_curve(y_test, y_predict, pos_label=1)
    plt.plot([0, 1],[0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='o')
    plt.show()
    return

####### KNN ##########
def kNeighbors(X_train,y_train,X_test,y_test):
    #trying to find best value for k
    k_range = range(5,30)
    scores ={}
    scores_list=[]
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        y_predict = knn.predict(X_test)
        scores[k] = accuracy_score(y_test, y_predict)
        scores_list.append(scores[k])
    
    #creating and fitting the model to training data based on selected k value
    knn = KNeighborsClassifier(n_neighbors = 8)
    scores_knn = cross_val_score(knn, X_train, y_train, cv =10 )
    knn.fit(X_train, y_train)
    #predicting the result based on trained model
    y_predict = knn.predict(X_test)
    
    return y_predict, scores_knn

def decisionTree(X_train,y_train,X_test):
    #creating and fitting the model to training data
    decision_tree = DecisionTreeClassifier(criterion="entropy")
    scores_dt = cross_val_score(decision_tree, X_train, y_train, cv =10 )
    decision_tree = decision_tree.fit(X_train, y_train)  
    #predicting the result based on trained model
    y_predict = decision_tree.predict(X_test)

    return y_predict, scores_dt

########## Rule Based ##############
def dummyClass(X_train, y_train, X_test):
    dc = DummyClassifier(strategy="stratified")
    scores_dc = cross_val_score(dc, X_train, y_train, cv =10)
    dc.fit(X_train, y_train )
    y_predict = dc.predict(X_test)
    
    return y_predict, scores_dc
    
def feature_selection(X,y):
    #---------- SelectKBest -----------#
    features = SelectKBest(score_func=f_classif, k=10)
    fit = features.fit(X,y)
    feat_scores = pd.DataFrame(fit.scores_)
    feat_cols = pd.DataFrame(X.columns)
    bestFeatures = pd.concat([feat_cols,feat_scores], axis = 1)
    bestFeatures.columns = ['Features','Score']
    best = bestFeatures.nlargest(20, 'Score' )
    print(bestFeatures.nlargest(20, 'Score' ))
    
    
    #---------- ExtraTreesClassifier -----------#
    etcmodel=ExtraTreesClassifier()
    etcmodel.fit(X,y)
    #print(etcmodel.feature_importances_)
    feat_imp = pd.Series(etcmodel.feature_importances_,index = X.columns)
    feat_imp.nlargest(20).plot(kind='barh')
    plt.show()
    
    return best


if __name__ =="__main__":
    main()
