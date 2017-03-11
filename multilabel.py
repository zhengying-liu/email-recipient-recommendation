#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:02:50 2017

@author: Evariste
"""

import pandas as pd
from feature_extraction import Word2VecFeatureExtractor
from utils import get_dataframes
from evaluation import split_train_test, get_validation_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

class MultilabelClassifier():

    def __init__(self, feature_extractor='word2vec'):
        self.classifier = None
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


    def Y_to_df(self, Y, threshold=0.5):
        "return a df with cols ['mid', 'list_of_recipients'] "
        df = self.df_test[['mid', 'list_of_recipients']]
        Y_bin = np.zeros(Y.shape)
        Y_bin[Y>=threshold] = 1
        recipients_not_ordered = self.mlb.inverse_transform(Y_bin)
        inds = Y[Y>=threshold].argsort(axis=1)
        df['list_of_recipients'] = recipients_not_ordered[inds]
        return df

    def df_to_Y(self, df):
        " return a 0-1 encoding matrix of the recipients in df"
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
        # TODO
        self.classifier = "the classifier of the model"

    def classifier_predict(self, X_test):
        # TODO
        Y_test = "0-1 encoding matrix of the predicted recipients"
        return Y_test

    def fit(self, df_train):
        self.df_train = df_train
        X_train = self.feature_extractor_fit_transform(df_train)
        Y_train = self.df_to_Y(df_train)
        self.classifier_fit(X_train, Y_train)

    def predict(self, df_test):
        self.df_test = df_test
        X_test = self.feature_extractor_transform(df_test)
        Y_test = self.classifier_predict(X_test)
        pred_df = self.Y_to_df(Y_test)
        return pred_df


def predict_by_multilabel_for_each_sender(training_info_t, training_info_v):
    grouped_train = training_info_t.groupby("sender")
    grouped_test = training_info_v.groupby("sender")
    preds = []
    models = []
    for name, group in grouped_train:
        df_train = group.reset_index()
        df_test = grouped_test.get_group(name).reset_index()

        model = MultilabelClassifier()
        model.fit(df_train)

        # pred_df: pd.DataFrame with columns ['mid', 'recipients']
        pred_df = model.predict(df_test)
        preds.append(pred_df)
        models.append(model)
    pred = pd.concat(preds).sort_values('mid')
    pred = pred.reset_index()[['mid', 'recipients']]
    return pred, models


if __name__ == "__main__":

    training, training_info, test, test_info = get_dataframes()
    training_info_t, training_info_v = split_train_test(training_info)

    #pred, models = predict_by_multilabel_for_each_sender(training_info_t, training_info_v)
    #score = get_validation_score(training_info_v, pred)
    print("Score: ", score)
