#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:18:22 2017

@author: salma
"""

from utils import read_data_info
from sklearn.feature_extraction.text import CountVectorizer

#%%

def build_corpus(dic):
    corpus = list()
    mids = list()
    for i in dic.items():
        mids.append(i[0])
        corpus.append(' '.join(i[1]['body']))
    return corpus, mids


def get_bag_words(corpus, mids):
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(corpus)
    d = dict()
    for i in range(len(mids)):
       d[mids[i]] = X[i]
    return d
   
    
if __name__ == "__main__":
    data_info = read_data_info()
    corpus, mids = build_corpus(data_info)
    bag_of_words = get_bag_words(corpus, mids)