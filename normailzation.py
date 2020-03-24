#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 下午4:41
# @Author  : Aries
# @Site    : 
# @File    : normailzation.py
# @Software: PyCharm

import re
import string
import jieba

# 加载停用词
with open('./dict/stop_words.utf8', encoding='utf8') as f:
	stopword_list = f.readlines()


def tokenize_text(text):
	tokens = jieba.cut(text)
	tokens = [token.strip() for token in tokens]
	
	return tokens


def remove_special_chars(text):
	tokens = tokenize_text(text)
	pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
	filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
	filtered_text = ' '.join(filtered_tokens)
	return filtered_text


def remove_stopwords(text):
	tokens = tokenize_text(text)
	filtered_tokens = [token for token in tokens if token not in stopword_list]
	filtered_text = ''.join(filtered_tokens)
	return filtered_text


def normalize_corpus(corpus, tokenize=True):
	normalized_corpus = []
	for text in corpus:
		text = remove_special_chars(text)
		text = remove_stopwords(text)
		# normalized_corpus.append(text)
		
		if tokenize:
			text = tokenize_text(text)
			normalized_corpus.append(text)
			# if len(text) > max:
			# 	max = len(text)
	
	return normalized_corpus


if __name__ == '__main__':
	remove_special_chars('我====是&&&中国人....')
