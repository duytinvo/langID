#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:38:32 2018

@author: duytinvo
"""
from __future__ import print_function
from __future__ import division

import os
import sys
import random
import argparse
import numpy as np
from utils.other_utils import SaveloadHP
from utils.data_utils import Vocab, Embeddings

# Use to initialze word embeddings
seed_num = 12345
random.seed(seed_num)
np.random.seed(seed_num)

def build_data(args):    
    print("Building dataset...")
    if not os.path.exists(args.model_dir): 
        os.mkdir(args.model_dir)
    vocab = Vocab(wl_th=args.word_thres,cl_th=args.char_thres)
    vocab.build([args.train_file,args.dev_file,args.test_file],cutoff=args.cutoff, firstline=True)
    args.vocab = vocab    
    if args.pre_trained:
        args.wd_embeddings = Embeddings.get_W(args.emb_file, args.word_dim,vocab.w2i, 0.25)
    else:
        args.wd_embeddings = None   
    SaveloadHP.save(args)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    
    argparser.add_argument('--train_file', help='Trained file', default="/media/data/subversive/newdata/train_split.csv", type=str)
    
    argparser.add_argument('--dev_file', help='Developed file', default="/media/data/subversive/newdata/dev_split.csv", type=str)
    
    argparser.add_argument('--test_file', help='Tested file', default="/media/data/subversive/newdata/GoodSubversionDataTest.csv", type=str)
                        
    argparser.add_argument("--cutoff", type = int, default = 3, help = "prune words ocurring <= cutoff")  
    
    argparser.add_argument("--char_thres", type = int, default = None, help = "char threshold")
    
    argparser.add_argument("--word_thres", type = int, default = None, help = "word threshold")

    argparser.add_argument("--emb_file", type = str, default = "/media/data/embeddings/glove/glove.6B.50d.txt", help = "embedding file")
    
    argparser.add_argument("--pre_trained", type = int, default = 1, help = "Use pre-trained embedding or not")
    
    argparser.add_argument("--word_dim", type = int, default = 50, help = "word embedding size")
        
    argparser.add_argument("--word_hidden_dim", type = int, default = 200, help = "LSTM layers")
                
    argparser.add_argument("--dropout", type = float, default = 0.5, help = "dropout probability")
    
    argparser.add_argument("--patience", type = int, default = 2, help = "early stopping")
            
    argparser.add_argument("--optimizer", type = str, default = "SGD", help = "learning method (adagrad, sgd, ...)")
    
    argparser.add_argument("--lr", type = float, default = 0.15, help = "learning rate") 
    
    argparser.add_argument("--decay_rate", type = float, default = 0.95, help = "decay learning rate")
        
    argparser.add_argument("--max_epochs", type = int, default = 100, help = "maximum # of epochs")
    
    argparser.add_argument("--batch_size", type = int, default = 16, help = "mini-batch size")  
    
    argparser.add_argument('--clip', help='Clipping value', default=-1, type=int)
    
    argparser.add_argument('--model_dir', help='Model dir', default="./results/", type=str)
    
    argparser.add_argument('--model_name', help='Model dir', default="bilstm.m", type=str)

    args = argparser.parse_args()
    
    build_data(args)
        

    
    





