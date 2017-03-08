 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 21:15:56 2017

@author: Evariste
"""
import numpy as np
import pandas as pd
from scipy.sparse import linalg
from utils import get_dataframes, received_mails_of_each_recipient_by_index
from evaluation import split_train_test, get_validation_score


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
 
    
def get_bag_words(training_info, test_info):
    count_vect = CountVectorizer(stop_words='english')
    corpus = training_info.body
    X_train = count_vect.fit_transform(corpus)
    X_test = count_vect.transform(test_info.body)
    return X_train, X_test, count_vect

    
def build_char_vector(X_train, mails_of_each_recipient,
                                 write_file=True, path="pred_nearest_recipient.txt"):    
    def sum_up(row): 
        vec = sum([X_train[i] for i in row.list_of_messages_by_index])
        if linalg.norm(vec) != 0:
            vec = vec.astype('float64') / linalg.norm(vec)
        return vec
        
    mails_of_each_recipient['char_vect'] = mails_of_each_recipient.apply(sum_up, axis=1)
    
    return mails_of_each_recipient
    
    
def predict_by_nearest_recipients(mails_of_each_recipient, test_info, count_vect, training,
                                      write_file=True, path="pred_nearest_recipient.txt"):   
    print("Begin prediction...")
    def predict(row):
        msg_vec = count_vect.transform([row.body])
        msg_vec = msg_vec.astype('float64')
        address_book = training.loc[row.sender].address_book
        similarity = []
        for k in address_book:
            if k in mails_of_each_recipient.index:
                sim = mails_of_each_recipient.loc[k].char_vect.dot(msg_vec.T)
                sim = sim[0,0]
                similarity.append((sim,k))
        li_sorted = sorted(similarity, reverse=True)[:10]
        first_10 = [t[1] for t in li_sorted]
        li = [mails_of_each_recipient.loc[idx].name for idx in first_10]
        res = " ".join(li)
        return res
    test_info["recipients"] = test_info.apply(predict, axis=1)
    pred = test_info[["mid","recipients"]]
    if write_file:
        pred.to_csv(path, index=False)
    return pred
    
if __name__ == "__main__":
    
    read_this_please =\
    """
    Involved pd.DataFrame : training, training_info, test, test_info, 
                            mails_of_each_recipient

    Essential columns: 
        training['address_book']
        training_info[date', 'body', 'sender',
                      'list_of_recipients']
        test_info[date', 'body', 'sender']
    column to predict: 
        test_info['list_of_recipients']
    """


#    Run following two lines only once:
    training, training_info, test, test_info = get_dataframes()
    training_info_t, training_info_v = split_train_test(training_info)
    X_train, X_test, count_vect = get_bag_words(training_info_t, training_info_v)
    mails_of_each_recipient = received_mails_of_each_recipient_by_index(training_info_t)
    build_char_vector(X_train, mails_of_each_recipient)
    
#   predict_by_nearest_message(training_info, test_info)
    pred = predict_by_nearest_recipients(mails_of_each_recipient, training_info_v, count_vect, training)  
    score = get_validation_score(training_info_v, pred)
    