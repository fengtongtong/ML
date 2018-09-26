#!/usr/bin/evn python3
# -*- coding:UTF-8 -*-

import tensorflow as tf

####################
#  CSV读取数据
# 从文件名队列中读取一行数据
# 从一组文件中读取一个batch
# 获取batch数据
####################

def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0]]
    a, b, c, d, e, f, i, g, k, label = tf.decode_csv(value, defaults)
    return tf.stack([a, b, c, d, e, f, i, g, k]), label

def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch([example, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

x_train_batch, y_train_batch = create_pipeline('/home/tonny/tensorflow/bin/main/Adware/cross_data/train7.csv', 571955)
x_test_batch, y_test_batch = create_pipeline('/home/tonny/tensorflow/bin/main/Adware/cross_data/test7.csv', 1)


####################
#   构建KNN结构
# 计算图输入占位符(train 使用全部样本，test 逐个样本进行测试)
# 使用 L1 距离进行最近邻计算, 计算 distance 时 xtest 会进行广播操作
# 预测: 获得最小距离的索引，然后根据此索引的类标和正确的类标进行比较
####################

xtrain = tf.placeholder(tf.float32, shape=[None, 9])
xtest = tf.placeholder(tf.float32, shape=[9])
distance = tf.reduce_sum(tf.abs(tf.subtract(xtrain, xtest)), axis=1)
pred = tf.argmin(distance, axis=0)

####################
#  训练和测试
# 初始化局部和全局变量
# 多线程定义
# 初始化最近邻分类器的准确率
# 获取训练集batch
# 在测试集上进行循环，每次测试一个数据，60000为测试用例数
# 获取当前测试样本的最近邻
# 获得最近邻预测标签，然后与真实的类标签比较
# 计算准确率和误检率
# 计算最终精确率
####################

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

accuracy,acc12,acc13,acc23,acc21,acc31,acc32 = 0.0, 0, 0, 0, 0, 0, 0

Xtrain, Ytrain = sess.run([x_train_batch, y_train_batch])

try:
    for i in range(60000):
        Xtest, Ytest = sess.run([x_test_batch, y_test_batch])
        testx=Xtest[:].reshape((9))
        testy = Ytest[:].reshape((1))
        nn_index = sess.run(pred, feed_dict={xtrain: Xtrain, xtest: testx})
        # print(nn_index)
        pred_class_label = Ytrain[nn_index]
        true_class_label = testy[0]
        # print("Test", i, "Predicted Class Label:", pred_class_label, "True Class Label:", true_class_label)
        if pred_class_label == true_class_label:
            accuracy += 1
        elif [pred_class_label, true_class_label] == [1, 2]:
            acc12+=1
        elif [pred_class_label, true_class_label] == [1, 3]:
            acc13+=1
        elif [pred_class_label, true_class_label] == [2, 1]:
            acc21 += 1
        elif [pred_class_label, true_class_label] == [2, 3]:
            acc23+=1
        elif [pred_class_label, true_class_label] == [3, 1]:
            acc31+=1
        elif [pred_class_label, true_class_label] == [3,2]:
            acc32+=1
        if i % 100 == 0 and i>0:
            acc=accuracy
            acc /= i
            print('Step: %d' % i, 'Accuracy: {0:f}'.format(acc))
            print("acc12,acc21,acc13,acc31,acc23,acc32=", acc12,acc21,acc13,acc31,acc23,acc32)
    print("Done!")
    accuracy /= 60000
    print("The finally accuracy=", accuracy)
    print("acc12,acc21,acc13,acc31,acc23,acc32=",acc12,acc21,acc13,acc31,acc23,acc32)
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
    
'''
for i in range(20000):
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: train_images, y_actual: train_labels, keep_prob: 0.5})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    if i % 400 == 0:
        test_accuracy = accuracy.eval(feed_dict={x: test_images, y_actual: test_labels, keep_prob: 0.5})
        print("step %d, testing accuracy %g" % (i, test_accuracy))
    train_step.run(feed_dict={x: train_images, y_actual: train_labels, keep_prob: 0.5})
'''



