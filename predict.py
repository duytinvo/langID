#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:38:32 2018

@author: duytinvo
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


def interactive_shell(args_file):
    """Creates interactive shell to play with model

    Args:
        model: instance of Classification

    """
    args = SaveloadHP.load(args_file)
    i2l={}
    for k,v in args.vocab.l2i.iteritems():
        i2l[v]=k
        
    print("Load Model from file: %s"%(args.model_name))
    classifier = Classifier(args)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    if not use_cuda:
        classifier.model.load_state_dict(torch.load(args.model_name), map_location=lambda storage, loc: storage)
        # classifier.model = torch.load(args.model_dir, map_location=lambda storage, loc: storage)
    else:
        classifier.model.load_state_dict(torch.load(args.model_name))
        # classifier.model = torch.load(args.model_dir)
        
    print("""
To exit, enter 'EXIT'.
Enter a sentence like 
input> wth is it????""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip()

        if words_raw == "EXIT":
            break
        
        words_raw = Encoder.str2uni(words_raw)
        label_prob, label_pred = classifier.predict(words_raw,5)
        for i in xrange(5):
            print(languages.lookup(i2l[label_pred[0][i]]).name)
            print(label_prob[0][i])
        

def scoring(sent, args, classifier):
    cl = args.vocab.cl            
     ## set model in eval model
    classifier.model.eval()
    
    fake_label = [0]        
    words = classifier.word2idx(sent)
    word_ids, sequence_lengths = seqPAD.pad_sequences([words], pad_tok=0, wthres=cl)

    data_tensors = Data2tensor.sort_tensors(fake_label, word_ids,sequence_lengths, volatile_flag=True)    
    fake_label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors
    label_score = classifier.model(word_tensor, sequence_lengths)
#    label_prob, label_pred = classifier.model.inference(label_score)
    return label_score

def predict(sent, args_file):
    args = SaveloadHP.load(args_file)
    i2l={}
    for k,v in args.vocab.l2i.iteritems():
        i2l[v]=k
        
    print("Load Model from file: %s"%(args.model_name))
    classifier = Classifier(args)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    if not use_cuda:
        classifier.model.load_state_dict(torch.load(args.model_name), map_location=lambda storage, loc: storage)
        # classifier.model = torch.load(args.model_dir, map_location=lambda storage, loc: storage)
    else:
        classifier.model.load_state_dict(torch.load(args.model_name))
        # classifier.model = torch.load(args.model_dir)
    
    label_prob, label_pred = classifier.predict(sent,5)
    for i in xrange(5):
        print(languages.lookup(i2l[label_pred[0][i]]).name)
        print(label_prob[0][i])
    return label_prob, label_pred

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
        
    argparser.add_argument('--model_args', help='Args file', default="./results/small.bilstm.args.pklz", type=str)
    
    args = argparser.parse_args()
        
    interactive_shell(args.model_args)

    
    


    
    





