"""
随机森林里的随机包含的意思是： 
样本随机 
特征随机 
参数随机 
模型随机（ID3 ,C4.5） 
extratree 极限树/极端随机树里的随机包含的意思是： 
特征随机 
参数随机 
模型随机（ID3 ,C4.5） 
分裂随机 
GBDT 基于梯度的boosting,单位为CADT,CART
xgboosting 基于梯度的boosting,单位为CADT,CART,LR
adamboosing 基于adam的boosting 单位为CADT,CART,LR
GCforest RF的集成

"""
# 加载包
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing 
import matplotlib.pyplot as plt


#读取数据
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def load_data(file_list):
    print("=======Loading Data=======")
    if len(file_list)==1:
        data=pd.read_csv(file_list[0])
    elif len(file_list)>1:
        data1=pd.read_csv(file_list[0])
        data2=pd.read_csv(file_list[1])
        data=pd.concat([data1,data2])
        for file in file_list[2:]:
            data3=pd.read_csv(file)
            data=pd.concat([data,data3])
    else:
        print('file_list is empty!')
    # 数据划分
    data=data.sample(frac=1)
    train_x=data.iloc[:,:-1].values
    train_y=data.iloc[:,-1].values
    x_train, x_test, y_train, y_test = train_test_split(
        train_x,
        train_y,
        test_size=0.30,
        random_state=1,
        stratify=train_y # 这里保证分割后y的比例分布与原数据一致
    )
    return x_train[:,1:],x_test[:,1:],y_train,y_test



def get_model(model):
    print('=======get_model_%x======='%model)
    clf={
        0:RandomForestClassifier(n_estimators=800),
        1:DecisionTreeClassifier(),
        2:KNeighborsClassifier(),
        3:GaussianNB(),
        4:SVC(),
        5:LogisticRegression(),
        6:GradientBoostingClassifier(init=None,n_estimators=1000,learning_rate=0.1, subsample=0.8,loss='deviance',max_features='sqrt',criterion='friedman_mse',min_samples_split =1200, min_impurity_split=None,min_impurity_decrease=0.0,max_depth=7,max_leaf_nodes=None,min_samples_leaf =60, warm_start=False,random_state=10),
        7:GradientBoostingRegressor(init=None,n_estimators=1000,learning_rate=0.1, subsample=0.8,loss='ls',max_features='sqrt',criterion='friedman_mse',min_samples_split =1200, min_impurity_split=None,min_impurity_decrease=0.0,max_depth=7,max_leaf_nodes=None,min_samples_leaf =60, warm_start=False,random_state=10),
        8:ExtraTreesClassifier(n_estimators=400, max_depth=None,min_samples_split=2, random_state=0),
        9:AdaBoostClassifier(n_estimators=1000)
    }
    return clf[model]



def train(clf,x_train,y_train):
    print('=======train=======')
    clf.fit(x_train,y_train)
    return clf


def test(clf,x_test,y_test):
    print("=======test=======")
    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:,1]
    return y_pred,y_predprob


def evaluate_nomal(y_test,clf,y_pred,y_predprob):
    print("\n\n=======evaluate_nomal=======")
    confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
    accuracy=metrics.accuracy_score(y_test, y_pred)
    p_r_f=metrics.classification_report(y_test,y_pred)
    feature_importances=[round(i, 4) for i in clf.feature_importances_]
    AUC=metrics.roc_auc_score(y_test, y_predprob)
    MSE=metrics.mean_squared_error(y_test, y_pred)
    
    print("\n\n confusion_matrix:\n",confusion_matrix,
          "\n\n Accuracy : %.4g "%accuracy,
         "\n\n precision_recall_f1-score:\n",p_r_f,
         "\n\n Feature importances:\n",feature_importances,
         "\n\n AUC Score (Train): %.4g "%AUC,
         "\n\n mean_squared_error: %.4g "%MSE)
   
    
def evaluate_test(y_test,y_predprob):
    print("\n\n=======evaluate_test=======")
    T,P,TP=0,0,0
    for i in range(len(y_test)):
        if y_predprob[i]>0.6:
            T+=1.0
        if y_predprob[i]>0.6 and y_test[i]==1:
            TP+=1.0
        if y_test[i]==1:
            P+=1
    
    recall=TP/P
    precision=TP/T
    matrix=[[TP,P-TP],[T-TP,len(y_test)-T-P+TP]]
    fscore=2*recall*precision/(recall+precision+0.0001)
    print("\n\n recall:%.4g"%recall,
          "\n\n precision:%.4g"%precision,
          '\n\n f-score:%.4g'%fscore,
          '\n\n confusion_matrix:',matrix)



def Gridsearch(clf,x_train,y_train,x_test,y_test):
    print("\n\n=======Gridsearch=======")
    gsearch = GridSearchCV(estimator = clf, param_grid = {'n_estimators':[100,200,300,400,500,800,1000]}, scoring='roc_auc',iid=False,cv=5)
    gsearch.fit(x_train,y_train)
    print('\n\n gsearch.grid_scores_:', gsearch.grid_scores_,
          '\n\n gsearch.best_params_:', gsearch.best_params_,
          '\n\n gsearch.best_score_:',gsearch.best_score_)

    

def CV(clf,x_train,y_train):
    print('\n\n=======CV=======')
    scores = cross_val_score(clf, x_train,y_train)
    print('\n\n CV_scores:',scores,
          '\n\n scores.mean():%.4g'%scores.mean(),
          '\n\n scores.max():%.4g'%scores.max()) 
    

    
def measure_result():
    w=open('data/measure.csv','w')
    column=['acc','pre_0','recall_0','fscore_0','pre_1','recall_1','fscore_1','AUC','MSE',
            'precision_test','recall_test','fscore_test','CV_mean','CV_max']
    w.write(column[0])
    for measure in column[1:]:
        w.write(','+measure)
    w.write('\n')
    w.close()
    
    
    
def plot(y_true,y_pred):
    x = range(100)
    plt.figure()
    plt.plot(x, y_pred[:100], c="r", label="n_estimators=300", linewidth=2)
    plt.plot(x, y_true[:100], c="k", label="training samples", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show()
    
    
    
if __name__ == '__main__':
    file_list=["data/malicious.csv","data/benign.csv"]  
    x_train,x_test,y_train,y_test=load_data(file_list)
    clf=get_model(0)
    _clf=train(clf,x_train,y_train)
    y_pred,y_predprob=test(_clf,x_test,y_test)
    evaluate_nomal(y_test,_clf,y_pred,y_predprob)
    evaluate_test(y_test,y_predprob)
    plot(y_test,y_pred)
    Gridsearch(clf,x_train,y_train,x_test,y_test)
    CV(clf,x_train,y_train)
    