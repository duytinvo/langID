#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:12:36 2018

@author: dtvo
"""
from __future__ import print_function
from __future__ import division

import torch
import argparse
import sys
import torch.nn.functional as F
from model import Classifier
from utils.other_utils import SaveloadHP, Encoder
from utils.data_utils import seqPAD, Data2tensor
from pycountry import languages
use_cuda = torch.cuda.is_available()
import os
from bs4 import BeautifulSoup
import pandas as pd

args_file='./results/small.bilstm.args.pklz'
k=1
args = SaveloadHP.load(args_file)
i2l={}
for k,v in args.vocab.l2i.iteritems():
    i2l[v]=k
support_lang = args.vocab.l2i.keys()

print("Load Model from file: %s"%(args.model_name))
classifier = Classifier(args)
## load model need consider if the model trained in GPU and load in CPU, or vice versa
if not use_cuda:
    classifier.model.load_state_dict(torch.load(args.model_name), map_location=lambda storage, loc: storage)
    # classifier.model = torch.load(args.model_dir, map_location=lambda storage, loc: storage)
else:
    classifier.model.load_state_dict(torch.load(args.model_name))
    # classifier.model = torch.load(args.model_dir)
    

def readtext(filedir, textfile, dtype):
    with open(os.path.join(filedir,textfile),'r') as f:
        text=f.read()
        soup = BeautifulSoup(text, "html.parser")
        text = Encoder.str2uni(soup.get_text(), dtype)
        text = text.strip().splitlines()
    text = [t for t in text if len(t)>0]
    return text
    
def twmetrics(tweetdata, classifier, i2l):
    y_true = list(tweetdata["lang"])
    y_pred = []
    for i in xrange(len(tweetdata)):
        sent = tweetdata.iloc[i,0]
        label_prob, label_pred = classifier.predict(sent,1)
        pl = label_pred[0][0]
        y_pred.append(i2l[pl])
    return y_true, y_pred   

from collections import Counter
def txtmetrics(txtdir, txtdata, classifier, i2l):
    y_true = list(txtdata["lang"]) 
    y_pred = []
    for i in xrange(len(txtdata)):
        textfile = txtdata.iloc[i,0]
        dtype = txtdata.iloc[i,1]
        sents = readtext(txtdir, textfile, dtype)
        pl = Counter()
        for sent in sents:
            label_prob, label_pred = classifier.predict(sent,1)
            pl.update([label_pred[0][0]])
        value, count = pl.most_common()[0]
        for value, count in pl.most_common():
            p='en'
            if i2l[value] in set(y_true):
                p = i2l[value]
                break
        y_pred.append(p)
    return y_true, y_pred
  
from sklearn import metrics
def class_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)  
    f1_ma = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')    
    f1_mi = metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')    
    f1_we = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted') 
    f1_no = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)  
    return acc, f1_ma, f1_mi, f1_we, f1_no

tweet = '/media/data/langID/evaluate/twituser-v1/twituser'
ftweetdata = pd.read_json(tweet, lines=True)
tweetdata = ftweetdata[ftweetdata[u'lang'].isin(support_lang)][["text", "lang"]]

euro = "/media/data/langID/evaluate/naacl2010-langid/EuroGOV"
tcl = "/media/data/langID/evaluate/naacl2010-langid/TCL"
wiki = "/media/data/langID/evaluate/naacl2010-langid/Wikipedia"

feurodata = pd.read_csv(euro+".meta", sep="\t", names=["filenames","dtype","lang","nfold"])
eurodata = feurodata[feurodata[u'lang'].isin(support_lang)][["filenames","dtype","lang"]]
eurotextfile = eurodata.iloc[0,0]
eurodtype = eurodata.iloc[0,1]
eurotext = readtext(euro, eurotextfile, eurodtype)


ftcldata = pd.read_csv(tcl+".meta", sep="\t", names=["filenames","dtype","lang","nfold"])
tcldata = ftcldata[ftcldata[u'lang'].isin(support_lang)][["filenames","dtype","lang"]]
tcltextfile = tcldata.iloc[0,0]
tcldtype = tcldata.iloc[0,1]
tcltext = readtext(tcl, tcltextfile, tcldtype)


fwikidata = pd.read_csv(wiki+".meta", sep="\t", names=["filenames","dtype","lang","nfold"])
wikidata = fwikidata[fwikidata[u'lang'].isin(support_lang)][["filenames","dtype","lang"]]
wikitextfile = wikidata.iloc[0,0]
wikidtype = wikidata.iloc[0,1]
wikitext = readtext(wiki, wikitextfile, wikidtype)