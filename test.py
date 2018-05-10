#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:38:32 2018

@author: duytinvo
"""
from __future__ import print_function
from __future__ import division

import sys
import random
import argparse
import numpy as np

import torch
from model import Classifier
from utils.other_utils import SaveloadHP
from utils.data_utils import Csvfile

use_cuda = torch.cuda.is_available()
seed_num = 12345
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def test(test_file, args_file):
    args = SaveloadHP.load(args_file)
    print("Load Model from file: %s"%(args.model_name))
    classifier = Classifier(args)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    if not use_cuda:
        classifier.model.load_state_dict(torch.load(args.model_name), map_location=lambda storage, loc: storage)
        # classifier.model = torch.load(args.model_dir, map_location=lambda storage, loc: storage)
    else:
        classifier.model.load_state_dict(torch.load(args.model_name))
        # classifier.model = torch.load(args.model_dir)

    test_data = Csvfile(test_file, firstline=False, word2idx=classifier.word2idx, tag2idx=classifier.tag2idx)
    test_metric = classifier.evaluate_batch(test_data)
    print("         - Test acc: %2f"%(100*test_metric))
    return test_metric


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
        
    argparser.add_argument('--test_file', help='Tested file', default="/media/data/langID/small_scale/test.csv", type=str)

    argparser.add_argument('--args_file', help='Args file', default="./results/s.bilstm_args.pklz", type=str)
    
    args = argparser.parse_args()
    
    test(args.test_file, args.args_file)
    

    

    
    





