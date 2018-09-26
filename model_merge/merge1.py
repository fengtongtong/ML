# 通用的预处理框架
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_csv_file(f, logging=False):
    print("==========读取数据=========")
    data =  pd.read_csv(f)
    data.astype("float")
    if logging:
        print(data.head(5)) #前5行信息
        print(f, "包含以下列")
        print(data.columns.values) #特征的列名
        print(data.describe()) #数量，均值，方差，分位点
        print(data.info()) #数据类型
        print(data.corr()) #相关    系数
    return data


def load_data():
    print("Loading Data ... ")
    data1=read_csv_file("data_9_feature/GM.csv")
    data2=read_csv_file("data_9_feature/adware.csv")
    data3=read_csv_file("data_9_feature/begin.csv")
    data=pd.concat([data1,data2,data3])
    data=data.sample(frac=1)
    train_x=data.iloc[:,:-1].values
    train_y=data.iloc[:,-1].values
    x_train, x_test, y_train, y_test = train_test_split(
        train_x,
        train_y,
        test_size=0.30,
        random_state=1,
        stratify=train_y ## 这里保证分割后y的比例分布与原数据一致
    )
    return x_train,x_test,y_train,y_test


def evaluate(n,m,num):
    recall=m/(num)
    precision=m/(n)
    f_score=2*recall*precision/(recall+precision)
    print("TP=%d,TP+FN=%d,TP+FP=%d"%(m,n,num))
    print("recall=%.4f,precision=%.4f,f_score=%.4f"%(recall,precision,f_score))


# 模型训练
from sklearn.ensemble import RandomForestClassifier
def train(threshold,x_train,y_train):
    for i in range(5):
        lr = RandomForestClassifier(n_estimators=5)
        lr.fit(x_train, y_train)
        proba_test = lr.predict_proba(x_train)[:, 2]
        
        threshold = threshold
        k,num=0,0
        n,m=0,0
        
        for pred in proba_test:
            result = 2 if pred > threshold else 0
            if result==y_train[k] and result==2:
                m+=1
                x_train=np.vstack((x_train,x_train[k]))
                y_train=np.hstack((y_train,result))
            if y_train[k]==2:
                num+=1
            if result==2:
                n+=1
            k+=1
        evaluate(n,m,num)
        print(x_train.shape,y_train.shape)
    return x_train,y_train
   

# 模型测试
def test(threshold,x_train,y_train,x_test,y_test):
    lr = RandomForestClassifier(n_estimators=5)
    lr.fit(x_train, y_train)
    proba_test = lr.predict_proba(x_test)[:, 2]

    k,num=0,0
    n,m=0,0
    threshold=threshold
    
    for pred in proba_test:
        result = 2 if pred > threshold else 0
        if result==y_test[k] and result==2:
            m+=1
        if y_test[k]==2:
            num+=1
        if result==2:
            n+=1
        k+=1
    evaluate(n,m,num)
    

# 主函数
def main():
    # 数据加载
    x_train, x_test, y_train, y_test=load_data()
    print("x_train.shape,x_test.shape",x_train.shape,x_test.shape)
    x_train_1,y_train_1=train(threshold=0.6,x_train=x_train,y_train=y_train)
    test(threshold=0.6,x_train=x_train_1,y_train=y_train_1,x_test=x_test,y_test=y_test)
