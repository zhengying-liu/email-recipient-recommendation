#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:18:22 2017

@author: salma
"""

import numpy
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from utils import read_data_info, clean_text_simple


class Word2VecFeatureExtractor:
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format('text.model.bin', binary=True)
        self.dim = 200

    def fit(self, df):
        pass

    def predict(self, df):
        ret = numpy.zeros((df.shape[0], self.dim))
        for index, row in df.iterrows():
            tokens = clean_text_simple(row.body, stemming=False, keep_dash=False)
            cont = 0

            for word in tokens:
                if word in self.model.vocab:
                    ret[index, :] += self.model[word]
                    cont += 1

            if cont > 0:
                ret[index, :] /= cont
        return ret


def get_network_features(training_info, received_mails_of_each_recipient):
    s = training_info.list_of_recipients.apply(lambda x: x[:10] if len(x)>10 else x)
    dummies = pd.get_dummies(s.apply(pd.Series), prefix='', prefix_sep='').sum(level=0, axis=1)
    df_asint = dummies.astype(int)
    co_occ = df_asint.T.dot(df_asint)
    return co_occ


def build_corpus(dic):
    corpus = list()
    mids = list()
    recipients = list()
    for i in dic.items():
        mids.append(i[0])
        corpus.append(' '.join(i[1]['body']))
        if 'recipients' in i[1]:
            recipients.append(i[1]['recipients'])
    return corpus, mids, recipients


def get_bag_words(corpus, mids, recipients):
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(corpus)
    d = dict()
    for i in range(len(mids)):
       d[mids[i]] = {'vector': X[i], 'recipients': recipients[i]}
    return d, count_vect


def get_word2vec(info):
    model = KeyedVectors.load_word2vec_format('text.model.bin', binary=True)
    d = dict()

    for k,v in tqdm(info.items()):
        body = v['body']
        aux = numpy.zeros((200,))
        cont = 0

        for word in body:
            if word in model.vocab:
                aux += model[word]
                cont += 1

        d[k] = aux / cont

    return d


def group_by_recipient(dic):
    d = dict()
    for k, v in dic.items():
        for rec in v['recipients']:
            d[rec] = []
    for k, v in dic.items():
        for rec in v['recipients']:
            d[rec].append((k, v['vector']))
    return d


if __name__ == "__main__":
    data_info = read_data_info()
    corpus, mids, recipients = build_corpus(data_info)
    bag_of_words, _ = get_bag_words(corpus, mids, recipients)
    recipient_dict = group_by_recipient(bag_of_words)
    #get_word2vec = get_word2vec(data_info)
