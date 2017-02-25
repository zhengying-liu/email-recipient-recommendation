#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:28:09 2017

@author: Evariste
"""

import numpy as np

def cos_similarity(vec_csr1, vec_csr2):
    scalar_product = vec_csr1.dot(vec_csr2.T)[0,0]
    vec_csr1 = vec_csr1.copy()
    vec_csr1.data **= 2
    vec_csr2 = vec_csr2.copy()
    vec_csr2.data **= 2
    norm1 = np.sqrt(vec_csr1.sum())
    norm2 = np.sqrt(vec_csr2.sum())
    return scalar_product/ norm1/ norm2

def predict_for_each_message(recipient_vector_df, message_vector):
    k=10
    func = lambda row: cos_similarity(row.vector, message_vector)
    recipient_vector_df["cosinus_similarity"] =\
                       recipient_vector_df.apply(func, axis=1)
    first_10 = list(recipient_vector_df.sort_values(by="cosinus_similarity")[:k]\
                                              .recipient.values)
    return ' '.join(first_10)

def predict_for_test_info(test_info, recipient_vector_df):
    pass
    
    

if __name__ == "__main__":
    print("haha")