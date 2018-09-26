
# 加载包
import numpy as np
import pandas as pd
import os

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


#读取数据
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.chdir("data_9_feature")

def data_preprocess(filename):
    data = pd.read_csv(filename)
   
    '''
    data[data<0]=0
    data=data.fillna(0)
    print(data.max().max())
    column=data.columns
    for i in column[:-1]:
        data[i] = data[i].map(lambda x: np.log(x+2)*10000)
    data=data.astype('int')
    '''
    '''
    min_max_scaler = preprocessing.MaxAbsScaler()
    data=min_max_scaler.fit_transform(data)*10000

    data=data.drop_duplicates(keep='first')
    '''

    return data


def load_data_np():
    Adware = np.loadtxt('adware.csv',skiprows=1,delimiter=',')
    Begin = np.loadtxt('begin.csv',skiprows=1,delimiter=',')
    GM = np.loadtxt('GM.csv',skiprows=1,delimiter=',')
    np.random.shuffle(Adware)
    np.random.shuffle(Begin)
    np.random.shuffle(GM)
    train=np.row_stack((Adware[:125000],Begin[:380000]))
    # train=np.row_stack((Adware[:125000],train))
    test=np.row_stack((Adware[125000:],Begin[380000:]))
    np.random.shuffle(train)
    np.random.shuffle(test)
    x_train,y_train=train[:,:9],train[:,9].ravel()
    x_test,y_test=test[:,:9], test[:,9]
    print(train.shape,test.shape)
    return x_train,y_train,x_test,y_test



def load_data_pd():
    Adware = data_preprocess('adware.csv')
    GM = data_preprocess('GM.csv')
    Begin = data_preprocess('begin.csv')

    Adware = Adware.sample(frac=1.0)
    GM = GM.sample(frac=1.0)
    Begin = Begin.sample(frac=1.0)
    
    print(len(Adware),len(GM),len(Begin))

    train = pd.merge(GM[:int(0.8*len(GM))],Adware[:int(0.8*len(Adware))], how='outer')
    test = pd.merge(GM[int(0.8*len(GM)):],Adware[int(0.8*len(Adware)):], how='outer')
    train =pd.merge(train,Begin[:int(0.8*len(Begin))], how='outer')
    test =pd.merge(test,Begin[int(0.8*len(Begin)):], how='outer')

    train=train.values
    test=test.values

    x_train,y_train=train[:,:9],train[:,9].ravel()
    x_test,y_test=test[:,:9], test[:,9]

    # min_max_scaler = preprocessing.MaxAbsScaler()
    # x_train=min_max_scaler.fit_transform(x_train)*10000
    # x_test=min_max_scaler.transform(x_test)*10000

    return x_train,y_train,x_test,y_test



def load_data_split():
    Adware = pd.read_csv('adware.csv')
    GM =pd.read_csv('GM.csv')
    Begin = pd.read_csv('begin.csv')
    train_data = pd.merge(Adware,GM, how='outer')
    train_data=train_data.drop_duplicates(keep='first')
    train_data = train_data.sample(frac=1.0)
    train_data = train_data.values
    print(sum(train_data[:,9].ravel()),len(train_data)-sum(train_data[:,9].ravel()))
    num_features = train_data.shape[0]
    print("Number of all features: \t", num_features)
    split = int(num_features * 0.8)
    train = train_data[:split]
    test = train_data[split:]
    x_train,y_train=train[:,:9],train[:,9].ravel()
    x_test,y_test=test[:,:9], test[:,9]
    print(train.shape,test.shape)
    return x_train,y_train,x_test,y_test



def model():
    clf = RandomForestClassifier(n_estimators=1000)
    # clf = DecisionTreeClassifier()
    # clf = KNeighborsClassifier()
    # clf = GaussianNB()
    # clf = SVC()
    # clf = LogisticRegression()
    # clf = GradientBoostingClassifier(init=None,n_estimators=1000,learning_rate=0.1, subsample=0.8,loss='deviance',max_features='sqrt',criterion='friedman_mse',min_samples_split =1200, min_impurity_split=None,min_impurity_decrease=0.0,max_depth=7,max_leaf_nodes=None,min_samples_leaf =60, warm_start=False,random_state=10)
    # clf = GradientBoostingRegressor(init=None,n_estimators=1000,learning_rate=0.1, subsample=0.8,loss='ls',max_features='sqrt',criterion='friedman_mse',min_samples_split =1200, min_impurity_split=None,min_impurity_decrease=0.0,max_depth=7,max_leaf_nodes=None,min_samples_leaf =60, warm_start=False,random_state=10)  
    # clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)
    # clf = AdaBoostClassifier(n_estimators=1000)
    return clf



def train():
    print('RandomForestClassifier')
    x_train,y_train,x_test,y_test=load_data_pd()
    clf=model()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:,1]
    print("Accuracy : %.4g" % clf.score(x_test,y_test))
    print("precision_recall_f1-score_accuracy:\n",metrics.classification_report(y_test,y_pred))
    print("confusion_matrix:\n",metrics.confusion_matrix(y_test,y_pred))
    print("Feature importances:",clf.feature_importances_)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(test[:,9].ravel(), y_predprob))
    # print(metrics.mean_squared_error(test[:,9].ravel(), output))

    '''
    m,n=0.0,0.0
    for i in range(len(test)):
    	if predict_proba[i][1]>0.5:
    		m+=1.0
    	if predict_proba[i][1]>0.5 and test[:,9].ravel()[i]==1:
    		n+=1.0
    k=sum(test[:,9].ravel())
    recall=n/k
    precision=n/m
    print("recall:",recall,"precision:",precision)
    print(k,m,n)
    '''


def Gridsearch():
    x_train,y_train,x_test,y_test=load_data_pd()
    clf=model()
    gsearch = GridSearchCV(estimator = clf, param_grid = {'n_estimators':[100,200,300,500,800,1000]}, scoring='accuracy',iid=False,cv=5)
    gsearch.fit(x_train,y_train)
    print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)

def CV():
    train,test=load_data_pd()
    clf=model()
    scores = cross_val_score(clf, train[:,:9],train[:,9].ravel())
    print(scores)
    print(scores.mean()) 

for i in range(10):
    print('Gridsearch()-----RF')
    Gridsearch()
