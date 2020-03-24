#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 下午4:37
# @Author  : Aries
# @Site    : 
# @File    : main.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.contrib import rnn

import amazon_master.data_process as process
import amazon_master.normailzation as norm
import numpy as np
import gensim
from random import randint
import os

batch_size = 100
max_step = 10
wordvec_size = 192
lstm_units = 64
learning_rate = 0.01

# 获取训练集和测试集
train_x, test_x, train_y, test_y = process.get_train_test_data()
# 加载word2vec model
models = gensim.models.Word2Vec.load('./models/zhiwiki_news.word2vec')


def get_batch(step):
	labels = np.zeros([batch_size, 2], dtype=np.float32)
	seg = train_x.shape[0] // batch_size
	corpus = np.zeros(shape=[batch_size, max_len, wordvec_size], dtype=np.float32)
	for i in range(batch_size):
		#  for sentence in data:
		if i % 2 == 0:
			index = randint(step * seg, step * seg + batch_size)
			if step * seg + batch_size > train_x.shape[0] - 1:
				index = 0
			labels[i] = [1.0]
		else:
			start = train_x.shape[0] - 1
			index = randint(step * seg + start, step * seg + start + batch_size)
			if step * seg + start + batch_size > train_x.shape[0] - 1:
				index = start
			labels[i] = [0.1]
		sent = norm.normalize_corpus([train_x[index]])
		for index, word in enumerate(sent[0]):
			if models.__contains__(word):
				word_vec_all = np.zeros(wordvec_size)
				word_vec_all = word_vec_all + models[word]
				if index >= 800:
					break
				corpus[i][index] = word_vec_all
	
	return corpus, labels


def get_test_batch():
	labels = np.zeros(shape=[batch_size, 2], dtype=np.float32)
	corpus = np.zeros(shape=[batch_size, max_len, wordvec_size], dtype=np.float32)
	for i in range(batch_size):
		#  for sentence in data:
		index = randint(0, test_x.shape[0])
		sent = norm.normalize_corpus([test_x[index]])
		for index, word in enumerate(sent[0]):
			if models.wv.__contains__(word):
				if index >= 800:
					break
				corpus[i, index] = models[word]
		labels[i] = test_y[i]
	return corpus, labels


max_len = 800


def main():
	tf.reset_default_graph()
	labels = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2])
	input_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_len, wordvec_size])
	
	# 定义lstm
	lstmCell = rnn.BasicLSTMCell(lstm_units)
	lstmCell = rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
	val, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)
	
	
	# 定义最后输出的参数
	weights = tf.Variable(tf.truncated_normal(shape=[lstm_units, 2]))
	bias = tf.Variable(tf.truncated_normal(shape=[2]))
	val = tf.transpose(val, [1, 0, 2])
	
	last = tf.gather(val, val.get_shape()[0] - 1)
	prediction = tf.matmul(last, weights) + bias
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=prediction))
	
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	
	# 计算准确率
	prediction_index = tf.argmax(prediction, 1)
	true_index = tf.argmax(labels, 1)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction_index, true_index), dtype=tf.float32))
	
	saver = tf.train.Saver()
	sess = tf.InteractiveSession()
	
	if os.path.exists('model') and os.path.exists('./model/checkpoint'):
		saver.restore(sess, tf.train.latest_checkpoint('./model'))
	else:
		tf.global_variables_initializer().run()
	loss_summary = tf.summary.scalar('loos', loss)
	writer = tf.summary.FileWriter('./logs', sess.graph)
	
	print('-----------------------------------------开始训练----------------------------------------------------')
	
	for step in range(max_step):
		for i in range(train_x.shape[0] // batch_size):
			print("i  and step: %d ::: %d" % (i, step))
			corpus, label = get_batch(i)
			_, l_train = sess.run([optimizer, loss], feed_dict={input_data: corpus, labels: label})
			if i % 50 == 0:
				l = loss_summary.eval(feed_dict={input_data: corpus, labels: label})
				writer.add_summary(l, i + step * train_x.shape[0] // batch_size)
			if i % 100 == 0:
				test_corpus, test_labels = get_test_batch()
				
				l_test, accur, index1, index2 = sess.run([loss, accuracy, prediction_index, true_index],
				                                         feed_dict={input_data: test_corpus, labels: test_labels})
				print('this time loss:  %d and accracy:  %d%% ' % (l_test, accur * 100))
			if i % 1000 == 0:
				if not os.path.exists('model'):
					os.mkdir('model')
				saver_path = saver.save(sess, './model/amazon_model.ckpt')
				print('saver_path:  %s' % saver_path)
	saver_path = saver.save(sess, './model/amazon_model.ckpt')
	print('saver_path:  %s' % saver_path)


if __name__ == '__main__':
	main()
