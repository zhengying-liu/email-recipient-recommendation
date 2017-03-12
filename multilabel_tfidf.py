#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 10:09:50 2017

@author: Evariste
"""

import pandas as pd
import numpy as np
from utils import get_dataframes, clean_raw_text
from evaluation import split_train_test, get_validation_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

class MultilabelClassifier():

    def __init__(self):
        self.classifier = LogisticRegression(multi_class='multinomial',solver='lbfgs')
        self.feature_extractor = TfidfVectorizer(stop_words='english', preprocessor=clean_raw_text)
        self.mlb = MultiLabelBinarizer()

        # df_train: pd.DataFrame with columns ['mid', 'date', 'body', 'list_of_recipients']
        # df_test: pd.DdataFrame with columns at least ['mid', 'date', 'body']
        # both of them have continous integer index
        self.df_train = None
        self.df_test = None

    # return a df with cols ['mid', 'list_of_recipients']
    def Y_to_df(self, Y, threshold=0.5):
        print("Y_to_df...")
        df = self.df_test[['mid']].copy()
        inds = (Y * (Y>=threshold)).argsort(axis=1)

        list_of_recipients = []
        for i, index in enumerate(inds):
            index = index[-10:][::-1]
            #print(Y[i, index])
            #print(self.mlb.classes_[index])
            list_of_recipients.append(self.mlb.classes_[index])

        df['list_of_recipients'] = list_of_recipients
        return df

    # return a 0-1 encoding matrix of the recipients in df
    def df_to_Y(self, df):
        print("df_to_Y...")
        Y = self.mlb.fit_transform(df['list_of_recipients'])
        return Y

    def feature_extractor_fit_transform(self, df_train):
        print("feature_extractor_fit_transform...")
        X_train = self.feature_extractor.fit_transform(df_train)
        return X_train

    def feature_extractor_transform(self, df_test):
        print("feature_extractor_transform...")
        X_test = self.feature_extractor.transform(df_test)
        return X_test

    def classifier_fit(self, X_train, Y_train):
        print("classifier_fit...")
        self.classifier.fit(X_train, Y_train)

    def classifier_predict(self, X_test):
        print("classifier_predict...")
        if len(self.mlb.classes_) > 1:
            Y_test = self.classifier.predict_proba(X_test)
        else:
            Y_test = np.ones((X_test.shape[0], 1))
        return Y_test

    def fit(self, df_train):
        self.df_train = df_train
        X_train = self.feature_extractor_fit_transform(df_train)
        print("shape of X_train: ", X_train.shape)
        Y_train = self.df_to_Y(df_train)
        print("shape of Y_train: ", Y_train.shape)
        self.classifier_fit(X_train, Y_train)

    def predict(self, df_test):
        self.df_test = df_test
        X_test = self.feature_extractor_transform(df_test)
        print("shape of X_test", X_test.shape)
        Y_pred = self.classifier_predict(X_test)
        print("shape of Y_test", Y_pred.shape)
        pred_df = self.Y_to_df(Y_pred)
        return pred_df


def predict_by_multilabel_for_each_sender(training_info_t, training_info_v, validation=False):
    grouped_train = training_info_t.groupby("sender")
    grouped_test = training_info_v.groupby("sender")
    preds = []
    models = []
    for name, group in tqdm(grouped_train):
        df_train = group.reset_index()
        df_test = grouped_test.get_group(name).reset_index()

        model = MultilabelClassifier()
        model.fit(df_train)

        # pred_df: pd.DataFrame with columns ['mid', 'list_of_recipients']
        pred_df = model.predict(df_test)
        preds.append(pred_df)
        models.append(model)

        if validation:
            print("Test score for this sender: ", get_validation_score(df_test, pred_df))
    pred = pd.concat(preds).sort_values('mid')
    pred = pred.reset_index()[['mid', 'list_of_recipients']]
    return pred, models


if __name__ == "__main__":
#    training, training_info, test, test_info = get_dataframes()
#    training_info_t, training_info_v = split_train_test(training_info)

    pred, models = predict_by_multilabel_for_each_sender(training_info_t, training_info_v, validation=True)
    score = get_validation_score(training_info_v, pred)
    print("Score: ", score)

#    pred_test, models_test = predict_by_multilabel_for_each_sender(training_info, test_info)
#    pred_test.to_csv("pred_decision_tree.txt", index=False)