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
    training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', 
                           index_col="sender")
    training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',',
                                index_col="mid")
    test = pd.read_csv(path_to_data + 'test_set.csv', sep=',',
                       index_col="sender")
    test_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',',
                            index_col="mid")
    
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
            recipients = training_info.loc[mid]["list_of_recipients"]
            # add sender to training_info
            training_info["sender"][mid] = row.name
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
            test_info["sender"][mid] = index
    
    return training, training_info, test, test_info
 

if __name__ == "__main__":
    training, training_info, test, test_info = get_dataframes()
    print(test_info)