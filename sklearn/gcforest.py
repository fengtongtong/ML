from GCForest import gcForest
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt



# loading the data
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



def evaluate_nomal(y_test,y_pred,y_predprob):
    print("\n\n=======evaluate_nomal=======")
    confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
    accuracy=metrics.accuracy_score(y_test, y_pred)
    p_r_f=metrics.classification_report(y_test,y_pred)
    AUC=metrics.roc_auc_score(y_test, y_predprob)
    MSE=metrics.mean_squared_error(y_test, y_pred)
    
    print("\n\n confusion_matrix:\n",confusion_matrix,
          "\n\n Accuracy : %.4g "%accuracy,
         "\n\n precision_recall_f1-score:\n",p_r_f,
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
    
    
    
def plot(y_test,y_pred):
    x = range(100)
    plt.figure()
    plt.plot(x, y_pred[:100], c="r", label="n_estimators=300", linewidth=2)
    plt.plot(x, y_test[:100], c="k", label="training samples", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show()



def train(x_train,x_test,y_train,y_test):
    print("\n\n=======trainning=======")
    gcf = gcForest(shape_1X=72, n_mgsRFtree=300, window=[5,10,15,20], stride=5,
                     cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=800, cascade_layer=np.inf,
                     min_samples_mgs=10, min_samples_cascade=50, tolerance=0.0, n_jobs=-1)
    x_tr_mgs = gcf.mg_scanning(x_train, y_train)
    x_te_mgs = gcf.mg_scanning(x_test)
    _ = gcf.cascade_forest(x_tr_mgs, y_train)
    pred_proba = gcf.cascade_forest(x_te_mgs)
    y_predprob = np.mean(pred_proba, axis=0)
    y_pred = np.argmax(y_predprob, axis=1)
    return y_pred,y_predprob[:,1]
    

if __name__=='__main__':
    file_list=["data/malicious.csv","data/benign.csv"]  
    x_train,x_test,y_train,y_test=load_data(file_list)
    y_pred,y_predprob=train(x_train,x_test,y_train,y_test)
    evaluate_nomal(y_test,y_pred,y_predprob)
    evaluate_test(y_test,y_predprob)
    plot(y_test,y_pred)
    
    '''
    gcf.fit(x_train, y_train)
    pred_X = gcf.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=pred_X)
    print('gcForest accuracy : {}'.format(accuracy))
    '''