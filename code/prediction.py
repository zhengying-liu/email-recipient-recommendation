#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:28:09 2017

@author: Evariste
"""

import numpy as np
import pandas as pd

from code.feature_extraction import get_bag_words, read_data_info, group_by_recipient, build_corpus


def vector_average(recipient_dict):
    d = dict()
    for k, list_of_tuples in recipient_dict.items():
        n = len(list_of_tuples)
        vec_sum = 0
        for vec in list_of_tuples:
            vec_sum = vec_sum + vec[1]
        avg_vec = vec_sum/n
        d[k] = [avg_vec]
    return d

def dict_to_dataframe(d):
    res = pd.DataFrame({"recipient":[], "vector":[]})
    i = 0
    for k,v in d.items():
        res.loc[i] = [k, v]
        i += 1
    return res

def cos_similarity(vec_csr1, vec_csr2):
    scalar_product = vec_csr1.dot(vec_csr2.T)[0,0]
    vec_csr1 = vec_csr1.copy()
    vec_csr1.data **= 2
    vec_csr2 = vec_csr2.copy()
    vec_csr2.data **= 2
    norm1 = np.sqrt(vec_csr1.sum())
    norm2 = np.sqrt(vec_csr2.sum())
    res = scalar_product/ norm1/ norm2
    return res

def predict_for_each_message(recipient_vector_df, message_vector):
    k=10
    func = lambda row: cos_similarity(row.vector, message_vector)
    recipient_vector_df["cosine_similarity"] =\
                       recipient_vector_df.apply(func, axis=1)
    first_10 = list(recipient_vector_df.sort_values(by="cosine_similarity")[:k]\
                                              .recipient.values)
    return ' '.join(first_10)

def predict_for_test_info(test_corpus, test_info, recipient_vector_df, count_vect, path="pred.txt"):
    test_info = test_info.copy()
    matrix = count_vect.transform(test_corpus)
    list_vectors = list()
    for i in range(matrix.shape[0]):
        list_vectors.append(matrix[i])
    test_info["vector"] = list_vectors
    func2 = lambda row: predict_for_each_message(recipient_vector_df, row.vector)
    test_info["recipients"] = test_info.apply(func2,axis=1)
    pred_df = test_info[["mid","recipients"]]
    pred_df.to_csv(path, index=False)
    
    

if __name__ == "__main__":
    data_info = read_data_info()
    corpus, mids, recipients = build_corpus(data_info)
    bag_of_words, count_vect = get_bag_words(corpus, mids, recipients)
    recipient_dict = group_by_recipient(bag_of_words)
    d = vector_average(recipient_dict)
    print("Converting dict to DataFrame")
    recipient_vector_df = dict_to_dataframe(d)
    
    test_info_df = pd.read_csv('input/test_info.csv', sep=',', header=0)
    test_info = read_data_info(filename="input/test_info.csv")
    test_corpus, _, _ = build_corpus(test_info)
    print("Begin prediction...")
    predict_for_test_info(test_corpus, test_info_df, recipient_vector_df, count_vect, path="pred.txt")
    