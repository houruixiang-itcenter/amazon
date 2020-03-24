#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 下午12:51
# @Author  : Aries
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

root = './data/yf_amazon/'

pd.set_option('display.max_columns', None)
# # 产品
# print('-------------------------------------------产品----------------------------------------------------')
# products = pd.read_csv(root + 'products.csv')
# print('产品数目: %d' % products.shape[0])
# print(products.head(5))
#
# # 类别
# print('-------------------------------------------类别----------------------------------------------------')
# categories = pd.read_csv(root + 'categories.csv')
# print('类别数目: %d' % categories.shape[0])
# print(categories.head(5))

# 评分
print('-------------------------------------------评分----------------------------------------------------')
ratings = pd.read_csv(root + 'ratings.csv')
print('用户数目: %d' % ratings.userId.unique().shape[0])
print('评分/评论  数目(总计): %d\n' % ratings.userId.unique().shape[0])
print(ratings.head(5))


# # 产品链接
# print('-------------------------------------------产品链接----------------------------------------------------')
# links = pd.read_csv(root + 'links.csv')
#
# print(links.head(5))


def data_pre_process():
	# print(ratings['rating'].info())
	# print('....')
	# print(ratings.isnull().any())
	cols = [x for i, x in enumerate(ratings.columns) if x != 'rating' and 'content']
	content = ratings['title'].fillna(' ') + ratings['comment'].fillna('  ')
	# features = ratings.drop(['title', 'comment'],axis=1)
	ratings['content'] = content
	features = ratings.drop(cols, axis=1)
	features.dropna(axis=0, how='any')
	print(features.head(5))
	return features


def pos_neg_data():
	feat = data_pre_process()
	pos_feat = feat[feat['rating'] > 3.0]
	neg_feat = feat[feat['rating'] <= 3.0]
	pos_corpus = np.array(pos_feat['content'])
	pos_labels = np.concatenate(
		[np.ones(shape=(pos_corpus.shape[0],1), dtype=np.float32), np.zeros(shape=(pos_corpus.shape[0],1), dtype=np.float32)],
		axis=1)
	neg_corpus = np.array(neg_feat['content'])
	neg_labels = np.concatenate(
		[np.zeros(shape=(neg_corpus.shape[0],1), dtype=np.float32), np.ones(shape=(neg_corpus.shape[0],1), dtype=np.float32)],
		axis=1)
	
	return pos_corpus, pos_labels, neg_corpus, neg_labels


def get_train_test_data():
	pos_corpus, pos_labels, neg_corpus, neg_labels = pos_neg_data()
	corpus = np.concatenate([pos_corpus, neg_corpus], axis=0)
	labels = np.concatenate([pos_labels, neg_labels], axis=0)
	
	train_x, test_x, train_y, test_y = train_test_split(corpus, labels,
	                                                    test_size=0.01, random_state=42)
	return train_x, test_x, train_y, test_y


def main():
	print('-------------------------------------------开始处理数据----------------------------------------------------')
	get_train_test_data()

# pos_neg_data()


if __name__ == '__main__':
	main()
