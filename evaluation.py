#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:44:23 2017

@author: salma
"""

import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    
def split_train_test(training_info, pr_train=0.6):
    n_train = int(len(training_info) * pr_train)
    indices = list(training_info.mid)
    train_indices = np.random.choice(indices, size=n_train, replace=False)
    test_indices = [x for x in indices if x not in train_indices]
    training_info_t = training_info[training_info['mid'].isin(train_indices)].reset_index()
    training_info_v = training_info[training_info['mid'].isin(test_indices)].reset_index()
    return training_info_t, training_info_v
    

def get_validation_score(training_info_v, prediction_df):
    training_info_v = training_info_v.sort_values(by='mid', axis=0)
    prediction_df = prediction_df.sort_values(by='mid', axis=0)
    prediction_df['list_of_recipients'] = prediction_df.apply(lambda row: [rec for rec in row.recipients.split(' ')], axis=1)
    return mapk(training_info_v.list_of_recipients.values, prediction_df['list_of_recipients'].values)
        
    