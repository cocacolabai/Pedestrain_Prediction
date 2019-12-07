# -*- coding: utf-8 -*-
"""
@author: Nehal
"""
# Bagged Decision Trees for Classification

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def BaggingClassif(X_train,y_train,X_test):
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    classif = DecisionTreeClassifier()
    num_trees = 50
    model = BaggingClassifier(base_estimator=classif, n_estimators=num_trees, random_state=seed)
    scores = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    return y_pred,scores

def RandomForest(X_train,y_train,X_test):
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    num_trees = 50
    model = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
    scores = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    return y_pred,scores

def ExtraTreeClassif(X_train,y_train,X_test):
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    num_trees = 50
    model = ExtraTreesClassifier(n_estimators=num_trees, random_state=seed)
    scores = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    return y_pred,scores

def AdaBoost(X_train,y_train,X_test):
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = AdaBoostClassifier(n_estimators=100, random_state=seed)
    scores = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    return y_pred,scores

def GradientTreeBoost(X_train,y_train,X_test):
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,random_state=seed)
    scores = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    return y_pred,scores

def VotingClassif(X_train,y_train,X_test):
    clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                    random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft')
    scores = model_selection.cross_val_score(model, X_train, y_train, cv=10)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_pred,scores

def EnsembleModels(X_train,y_train,X_test, y_test):
    print("Result using Bagging Classification:\n")
    y_bag, scores_bag = BaggingClassif(X_train,y_train,X_test)
    printReport(y_test,y_bag)
    
    print("Result using Random Forest Classification:\n")
    y_rf, scores_rf = RandomForest(X_train,y_train,X_test)
    printReport(y_test,y_rf)
    
    print("Result using Extra Tree Classification:\n")
    y_et, scores_et = ExtraTreeClassif(X_train,y_train,X_test)
    printReport(y_test,y_et)
    
    print("Result using AdaBoost Classification:\n")
    y_ab, scores_ab = AdaBoost(X_train,y_train,X_test)
    printReport(y_test,y_ab)
    
    print("Result using Gradient Tree Boosting Classification:\n")
    y_gtb, scores_gtb = GradientTreeBoost(X_train,y_train,X_test)
    printReport(y_test,y_gtb)
    
    print("Result using Voting Classification:\n")
    y_vc, scores_vc = VotingClassif(X_train,y_train,X_test)
    printReport(y_test,y_vc)
    
    data2 = {'Bagging':scores_bag,'Random Forest':scores_rf, 'Extra Tree':scores_et, 'AdaBoost':scores_ab,'GradientBoost':scores_gtb,'Voting':scores_vc}
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