# !/usr/bin/evn python3
# -*- coding:UTF-8 -*-

####################
#      加载项
####################
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

####################
#     数据读取
# Step 1: read in data from the .xls file
####################
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.chdir("D:\\Anaconda3.5\\Scripts")
DATA_FILE = 'fire_theft.xls'
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0) # 指定是第几张表
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)]) #按行来读取
n_samples = sheet.nrows - 1
'''
num_puntos = 100
conjunto_puntos = []
for i in range(num_puntos):
    x1= np.random.normal(0.0, 0.9)
    y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.05)
    conjunto_puntos.append([x1, y1])

x_data = [v[0] for v in conjunto_puntos]
y_data = [v[1] for v in conjunto_puntos]
'''

####################
#     模型构建
# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
# Step 3: create weight and bias, initialized to 0
# Step 4: build model to predict
# Step 5: use the square error as the loss function
# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
####################
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')
Y_predicted = X * w + b
loss = tf.square(Y - Y_predicted, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


####################
#    训练和测试
# Step 7: initialize the necessary variables, in this case, w and b
# Step 8: train the model, Session runs train_op and fetch values of loss
# Step 9: output the values of w and b
####################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/03/linear_reg', sess.graph)
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))
    writer.close()
    w_value, b_value = sess.run([w, b])


####################
#     画图显示
####################
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()
'''
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.show()
'''