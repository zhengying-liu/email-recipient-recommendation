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

def predict_by_nearest_message(training_info, test_info, write_file=False,
                               path="pred_nearest_message.txt"):
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
    if write_file:
        pred.to_csv(path, index=False)
    return pred, similarity, count_vect, X_train, X_test

def received_mails_of_each_recipient_by_index(training_info_train):
    print("Constructing received_mids_for_each_recipient...")
    recipient_mids_dict = dict()
    for index, row in training_info_train.iterrows():
        list_of_recipients = row["list_of_recipients"]
        for recipient in list_of_recipients:
            if recipient in recipient_mids_dict:
                recipient_mids_dict[recipient][0].append(index)
            else:
                recipient_mids_dict[recipient] = [[index]]
    mails_of_each_recipient = pd.DataFrame(recipient_mids_dict).T
    mails_of_each_recipient.columns = ['list_of_messages_by_index']
    mails_of_each_recipient['number_of_received_messages'] =\
                           mails_of_each_recipient.apply(lambda row: 
                               len(row['list_of_messages_by_index']),axis=1)
    return mails_of_each_recipient
    

def predict_by_nearest_recipient(training_info, test_info, 
                                 path="pred_nearest_recipient.txt"):
    pass

if __name__ == "__main__":
    
    read_this_please =\
    """
    Involved pd.DataFrame : training, training_info, test, test_info, 
                            mails_of_each_recipient
    columns in training: ['sender', 'mids', 'list_of_mids', 'address_book']
    columns in training_info: ['mid', 'date', 'body', 'recipients', 
                               'list_of_recipients', 'sender']
    columns in test: ['sender', 'mids', 'list_of_mids', 'address_book']
    columns in test_info: ['mid', 'date', 'body', 'sender']
    index of mails_of_each_recipient: recipient
    columns in mails_of_each_recipient: ['list_of_messages_by_index',
                                         'number_of_received_messages']

    Essential columns: 
        training['address_book']
        training_info[date', 'body', 'sender',
                      'list_of_recipients']
        test_info[date', 'body', 'sender']
    column to predict: 
        test_info['list_of_recipients']
    """


#    Run following two lines only once:
#    training, training_info, test, test_info = get_dataframes()
    mails_of_each_recipient = received_mails_of_each_recipient_by_index(training_info)


#    pred, similarity, count_vect, X_train, X_test =\
#        predict_by_nearest_message(training_info, test_info)