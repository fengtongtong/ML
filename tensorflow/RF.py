#!/usr/bin/evn python3
#-*- coding:UTF-8 -*-

# 加载包
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os

####################
#   CSV读取数据
# 只显示错误信息,不显示警告信息
# 数据集目录，数据集名称
# 数据集读取，训练集和测试集
####################

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
#os.chdir("/home/tonny/tensorflow/bin/main/Adware/cross_data")
TRAINING = "train5.csv"
TEST = "test5.csv"

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

####################
#   构建RF结构
####################

hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(num_trees=1000,max_nodes=10000,num_classes=4,num_features=9)
classifier=tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(hparams)


####################
#  训练和测试
####################

for i in range(200):
    classifier.fit(x=training_set.data,y=training_set.target,steps=100,batch_size=10000)
    acc=0.0
    for j in range(6):
        accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target,batch_size=10000)["accuracy"]
        acc+=accuracy_score
    acc=acc/6
    print('Step: %d' % (i*100), 'Accuracy: {0:f}'.format(acc))
