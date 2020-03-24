#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 下午5:15
# @Author  : Aries
# @Site    :
# @File    : word2vec.py
# @Software: PyCharm
from gensim.models import KeyedVectors
import numpy as np


def main():
	path = '/Users/houruixiang/python/data/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'
	# 加载腾讯AILab词向量
	word2vec_text = KeyedVectors.load_word2vec_format(path, binary=False)
	# np.save('./models/word2vec.npy', word2vec_text)


if __name__ == '__main__':
	main()
