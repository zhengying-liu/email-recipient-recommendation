#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 21:15:56 2017

@author: Evariste
"""
import numpy as np
import pandas as pd

def get_dataframes():
    path_to_data = "input/"
    training = pd.read_csv(path_to_data + 'training_set.csv', sep=',')
    training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',')
    test = pd.read_csv(path_to_data + 'test_set.csv', sep=',')
    test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',')
    
    # list of mids in training, test
    print("Constructing list_of_mids...")
    func = lambda row: list(map(int, row.mids.split(' ')))
    training["list_of_mids"] = training.apply(func, axis=1)
    test["list_of_mids"] = test.apply(func, axis=1)
    
    # list of recipients in training_info
    print("Constructing list_of_recipients...")
    func = lambda row: [rec for rec in row.recipients.split(' ') if '@' in rec]
    training_info["list_of_recipients"] = training_info.apply(func, axis=1)
    
    # create an empty column for sender in training_info, test_info
    func = lambda row: ""
    training_info["sender"] = training_info.apply(func,axis=1)
    test_info["sender"] = test_info.apply(func,axis=1)

    # address book in training, test
    print("Constructing address book...")
    def count_contacted_recipients(row):
        list_of_mids = row["list_of_mids"]
        res = dict()
        for mid in list_of_mids:
            idx = np.where(training_info["mid"] == mid)[0][0]
            recipients = training_info.loc[idx]["list_of_recipients"]
            # add sender to training_info
            training_info.loc[idx, "sender"] = row.sender
            for rec in recipients:
                if rec not in res:
                    res[rec] = 0
                else:
                    res[rec] += 1
        return res
    training["address_book"]= training.apply(count_contacted_recipients,axis=1)
    test["address_book"] = training["address_book"].copy()
    
    # add sender to test_info
    print("Add sender to test_info...")
    for index, row in test.iterrows():
        list_of_mids = row["list_of_mids"]
        for mid in list_of_mids:
            idx = np.where(test_info["mid"] == mid)[0][0]
            test_info.loc[idx, "sender"] = row.sender
    
    return training, training_info, test, test_info


from sklearn.feature_extraction.text import CountVectorizer

def predict_by_nearest_message(training_info, test_info, path="pred.txt"):
    count_vect = CountVectorizer(stop_words='english')
    corpus = training_info.body
    X_train = count_vect.fit_transform(corpus)
    X_test = count_vect.transform(test_info.body)
    similarity = X_test.dot(X_train.T).toarray()
    indexes_of_nearest_message = similarity.argmax(axis=1)
    # for each row of test_info, return recipients of the nearest message 
    # in training_info
    func = lambda row: training_info.loc[indexes_of_nearest_message[row.name]]\
                                        ["recipients"]
    test_info["recipients"] = test_info.apply(func, axis=1)
    pred = test_info[["mid","recipients"]]
    pred.to_csv(path, index=False)
    return pred, similarity, count_vect
    

if __name__ == "__main__":
#    training, training_info, test, test_info = get_dataframes()
    pred, similarity, count_vect = predict_by_nearest_message(training_info, test_info)
    print(pred)