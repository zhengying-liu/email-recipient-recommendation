#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:02:50 2017

@author: Evariste
"""

import pandas as pd
import numpy as np
from feature_extraction import Word2VecFeatureExtractor
from utils import get_dataframes
from evaluation import split_train_test, get_validation_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier

class MultilabelClassifier():

    def __init__(self, feature_extractor_name='word2vec', feature_extractor=None):
        self.classifier = None

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            if feature_extractor == 'word2vec':
                self.feature_extractor = Word2VecFeatureExtractor()
            else:
                raise Exception("Unknown feature extractor")

        self.mlb = MultiLabelBinarizer()

        # df_train: pd.DataFrame with columns ['mid', 'date', 'body', 'list_of_recipients']
        # df_test: pd.DdataFrame with columns at least ['mid', 'date', 'body']
        # both of them have continous integer index
        self.df_train = None
        self.df_test = None

    # return a df with cols ['mid', 'list_of_recipients']
    def Y_to_df(self, Y, address_book, threshold=0.5, debug=False):
        df = self.df_test[['mid']].copy()
        inds = Y.argsort(axis=1)

        sum_freq = 0
        for k,v in address_book.items():
            sum_freq += v

        list_of_recipients = []
        for i, index in enumerate(inds):
            if debug:
                print("*" * 50)
                #print(self.df_test['body'][i])
                print(self.df_test['list_of_recipients'][i])

            aux = []
            for ind in index:
                #if Y[i, ind] >= threshold:
                receiver = self.mlb.classes_[ind]
                aux.append((Y[i, ind] + address_book[receiver] / float(sum_freq), ind))
            aux = sorted(aux, reverse=True)[:10]

            filtered_index = []
            for x in aux:
                filtered_index.append(x[1])
            list_of_recipients.append(self.mlb.classes_[filtered_index])

            if debug:
                print(aux)
                print(self.mlb.classes_[filtered_index])
                print("*" * 50)

        df['list_of_recipients'] = list_of_recipients
        return df

    # return a 0-1 encoding matrix of the recipients in df
    def df_to_Y(self, df):
        Y = self.mlb.fit_transform(df['list_of_recipients'])
        return Y

    def feature_extractor_fit_transform(self, df_train):
        self.feature_extractor.fit(df_train)
        X_train = self.feature_extractor.predict(df_train)
        return X_train

    def feature_extractor_transform(self, df_test):
        X_test = self.feature_extractor.predict(df_test)
        return X_test

    def classifier_fit(self, X_train, Y_train):
        #self.classifier = OneVsRestClassifier(DecisionTreeClassifier(max_depth=30, random_state=0))
        #self.classifier = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, max_depth=9))
        self.classifier = OneVsRestClassifier(LogisticRegression(C=1))
        self.classifier.fit(X_train, Y_train)

    def classifier_predict(self, X_test):
        if len(self.mlb.classes_) > 1:
            Y_test = self.classifier.predict_proba(X_test)
        else:
            Y_test = np.ones((X_test.shape[0], 1))
        return Y_test

    def fit(self, df_train):
        self.df_train = df_train
        X_train = self.feature_extractor_fit_transform(df_train)
        Y_train = self.df_to_Y(df_train)
        self.classifier_fit(X_train, Y_train)

    def predict(self, df_test, address_book, debug=False):
        self.df_test = df_test
        X_test = self.feature_extractor_transform(df_test)
        Y_test = self.classifier_predict(X_test)
        pred_df = self.Y_to_df(Y_test, address_book, debug=debug)
        return pred_df


def predict_by_multilabel_for_each_sender(training_info_t, training_info_v, training, validation=False):
    grouped_train = training_info_t.groupby("sender")
    grouped_test = training_info_v.groupby("sender")
    preds = []
    models = []
    feature_extractor = Word2VecFeatureExtractor()
    total = 0
    average_score = 0
    for name, group in tqdm(grouped_train):
        df_train = group.reset_index()
        df_test = grouped_test.get_group(name).reset_index()
        address_book = training.address_book[name]

        model = MultilabelClassifier(feature_extractor=feature_extractor)
        model.fit(df_train)

        # pred_df: pd.DataFrame with columns ['mid', 'list_of_recipients']
        pred_df = model.predict(df_test, address_book)
        preds.append(pred_df)
        models.append(model)

        if validation:
            score = get_validation_score(df_test, pred_df)
            print("Test score for this sender: ", score)
            rows = df_test.shape[0]
            average_score = (total * average_score + rows * score) / (total + rows)
            total += rows
            print("Average score for all senders: ", average_score)

    pred = pd.concat(preds).sort_values('mid')
    pred = pred.reset_index()[['mid', 'list_of_recipients']]

    def write_recipients(row):
        li = row['list_of_recipients']
        ret = " ".join(li)
        return ret
    pred['recipients'] = pred.apply(write_recipients, axis=1)

    return pred, models


if __name__ == "__main__":
    training, training_info, test, test_info = get_dataframes()
    training_info_t, training_info_v = split_train_test(training_info)

    pred, models = predict_by_multilabel_for_each_sender(training_info_t, training_info_v, training, validation=True)
    score = get_validation_score(training_info_v, pred)
    print("Score: ", score)

    #pred_test, models_test = predict_by_multilabel_for_each_sender(training_info, test_info, training)
    #pred_test = pred_test[['mid', 'recipients']]
    #pred_test.to_csv("pred_logistic_regression_2.txt", index=False)
