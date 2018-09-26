#!/usr/bin/evn python3
#-*- coding:UTF-8 -*-

import tensorflow as tf
import numpy as np

#从文件名队列中读取一行数据
def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.], [0]]
    a,b,c,d,e,f,i,g,k,label = tf.decode_csv(value, defaults)
    return tf.stack([a,b,c,d,e,f,i,g,k]),label

#从一组文件中读取一个batch
def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


x_train_batch, y_train_batch = create_pipeline('/home/tonny/tensorflow/bin/main/train_data.csv', 1, num_epochs=10)
x_test_batch, y_test_batch = create_pipeline('/home/tonny/tensorflow/bin/main/test_data.csv', 1)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

k=0
try:
    for i in range(10):
        k=k+1
        print("trainning",k)
        x_train, y_train = sess.run([x_train_batch, y_train_batch])
        print("testing")
        x_test, y_test = sess.run([x_test_batch, y_test_batch])
        textx = x_test.reshape([9])
        texty = y_test.reshape([1])
        print(texty[0])
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
