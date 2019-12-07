# -*- coding: utf-8 -*-
"""
@author: Nehal
"""


from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def Linear_SVC(X_train,y_train,X_test):
    clf = LinearSVC(random_state = 100,tol = 1e-5, max_iter=3000)
    scores = cross_val_score(clf, X_train, y_train, cv =10 )
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    return y_pred,scores

def SGD_Classifier(X_train,y_train,X_test):
    clf = SGDClassifier(max_iter=1000, random_state = 70,tol = 1e-3)
    scores = cross_val_score(clf, X_train, y_train, cv =10 )
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    return y_pred,scores

def Ridge_Classifier(X_train,y_train,X_test):
    clf = RidgeClassifier(max_iter=1000, random_state = 40,tol = 1e-3)
    scores = cross_val_score(clf, X_train, y_train, cv =10 )
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    return y_pred,scores

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
    
def LinearModels(X_train,y_train,X_test,y_test):
    print("Result using Linear SVC Classification:\n")
    y_lsvc, scores_lsvc = Linear_SVC(X_train,y_train,X_test)
    printReport(y_test,y_lsvc)
    
    print("Result using Stochastic Gradient Descent Classification:\n")
    y_sgdc, scores_sgdc = SGD_Classifier(X_train,y_train,X_test)
    printReport(y_test,y_sgdc)
    
    print("Result using Ridge Classification:\n")
    y_rc, scores_rc = SGD_Classifier(X_train,y_train,X_test)
    printReport(y_test,y_rc)
    
    
    data2 = {'Linear SVC':scores_lsvc,'SGD':scores_sgdc, 'Ridge':scores_rc}
    df2 = pd.DataFrame(data=data2)
    
    paired_t_test(df2)
    print(df2)
    
    return

def paired_t_test(df2):
    for x in range(0,len(df2.columns)):
        for y in range(x+1,len(df2.columns)):
            col1 = df2.columns[x]
            col2 = df2.columns[y]
            before = df2[col1]
            after = df2[col2]
            print(col1,"-",col2)
            print(stats.ttest_rel(a=before.values,b=after.values))
            print("")
    