#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:46:05 2018

@author: dtvo
"""
from __future__ import print_function
from __future__ import division
import argparse
import sys
import time
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.svm import LinearSVC 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

def buildcls(ml_cls="NB", tfidf=False, use_hash=False, scaler=False):
    print("- Construct the baseline...")
    start = time.time()
    if ml_cls=="NB":
        classifier = MultinomialNB()
    elif ml_cls=="SVM":
        classifier = LinearSVC(verbose=5)
    elif ml_cls=="AB":
        classifier = AdaBoostClassifier()
    elif ml_cls=="GB":
        classifier = GradientBoostingClassifier(verbose=5)
    elif ml_cls=="RF":
        classifier = RandomForestClassifier(n_estimators=100, verbose=5)
    else:
#        DEFAULT: Logistic Regression
        classifier = SGDClassifier(verbose=5, loss='log', n_iter=100)
#        classifier = LogisticRegression(verbose=5)
    
    settings = []    
    if use_hash:
        settings = [('vectorizer', HashingVectorizer())]    
    elif tfidf:
        settings = [('vectorizer', TfidfVectorizer())]
    else:
#       DEFAULT: BOW counting
        settings = [('vectorizer', CountVectorizer())]
    # scaller cannot use with NB (MultinomialNB)
    if scaler and ml_cls!="NB":
        settings += [('scaler', StandardScaler())]
        
    settings += [('classifier', classifier)]
    model = Pipeline(settings)
    
    parameters = {'vectorizer__analyzer': ['char'],
                  'vectorizer__ngram_range': [(1,3)],
                  'vectorizer__min_df': [5],
                  'vectorizer__binary': (True, False)
                  }
    end = time.time()
    print("\t+ Done: %.4f(s)"%(end-start))
    return model, parameters

def train(args):
    data_train = pd.read_csv(args.train_file, header=None, names=["uid","twid","tweet","langid"]).sample(frac=1).reset_index(drop=True)
    data_dev = pd.read_csv(args.dev_file, header=None, names=["uid","twid","tweet","langid"]).sample(frac=1).reset_index(drop=True)
    
    data_merge = pd.concat([data_train, data_dev])
    dev_fold = [-1]*len(data_train) + [0]*len(data_dev)
    
    x_traindev, y_traindev = data_merge.tweet.as_matrix(), data_merge.langid.as_matrix()
    
    pipeline, parameters = buildcls(args.ml_cls, args.tfidf, args.use_hash, args.scaler)
    
    print("- Train the baseline...")
    start = time.time()
    model = GridSearchCV(pipeline, parameters, cv=PredefinedSplit(test_fold=dev_fold), verbose=5)
    model.fit(x_traindev, y_traindev)
    end = time.time()
    print("\t+ Done: %.4f(s)"%(end-start))
    best_model = model.best_estimator_
    save(best_model, args.model_name)
    return best_model

def evaluate(data, model):
    print("- Evaluate the baseline...")
    start = time.time()
    X_dev, y_true = data
    y_pred = model.predict(X_dev)
    acc, f1_ma, f1_mi, f1_we, f1_no = class_metrics(y_true, y_pred)
    confmtx(y_true, y_pred)
    end = time.time()
    print("\t+ Done: %.4f(s)"%(end-start))
    return acc, f1_ma, f1_mi, f1_we, f1_no

def save(model, mfile):
    print("- Save the model...")
    joblib.dump(model, mfile)
    print("\t+ Done.")
    
def load(mfile):
    print("- Load the model...")
    model = joblib.load(mfile)
    print("\t+ Done.")
    return model

def test(args, model):
    data_test = pd.read_csv(args.test_file, header=None, names=["uid","twid","tweet","langid"]).sample(frac=1).reset_index(drop=True)
    x_test, y_test = data_test.tweet.as_matrix(), data_test.langid.as_matrix()
    mtrcs = evaluate([x_test, y_test], model)
    return mtrcs
    
def class_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)  
    f1_ma = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')    
    f1_mi = metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')    
    f1_we = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted') 
    f1_no = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)  
    print("\t+ Accuracy: %.4f(%%)"%(acc*100))
    return acc, f1_ma, f1_mi, f1_we, f1_no

def confmtx(y_true, y_pred):
    from pandas_ml import ConfusionMatrix
    confusion_matrix = ConfusionMatrix(list(y_true), list(y_pred))
    classification_report = confusion_matrix.classification_report
    print('-' * 75 + '\nConfusion Matrix\n')
    print(confusion_matrix)
    print('-' * 75 + '\nClassification Report\n')
    print(classification_report)

if __name__ == '__main__':
    """
    python baselines.py --train_file /media/data/langID/small_scale/train.csv --dev_file /media/data/langID/small_scale/dev.csv --test_file /media/data/langID/small_scale/test.csv --model_name ./results/small.NB.m --ml_cls NB
    """
    argparser = argparse.ArgumentParser(sys.argv[0])
    
    argparser.add_argument('--train_file', help='Trained file', default="/media/data/langID/small_scale/train.csv", type=str)
    
    argparser.add_argument('--dev_file', help='Developed file', default="/media/data/langID/small_scale/dev.csv", type=str)
    
    argparser.add_argument('--test_file', help='Tested file', default="/media/data/langID/small_scale/test.csv", type=str)
    
    argparser.add_argument("--tfidf", action='store_true', default = False, help = "tfidf flag")
    
    argparser.add_argument("--use_hash", action='store_true', default = False, help = "hashing flag")
    
    argparser.add_argument("--scaler", action='store_true', default = False, help = "scale flag")
    
    argparser.add_argument('--ml_cls', help='Machine learning classifier', default="NB", type=str)
    
    argparser.add_argument('--model_name', help='Model dir', default="./results/small.NB.m", type=str)
        
    args = argparser.parse_args()
    
    model = train(args)
    
    acc, f1_ma, f1_mi, f1_we, f1_no = test(args, model)
    
    