#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:03:17 2018

@author: dtvo
"""
from __future__ import print_function
from utils.other_utils import Csvfile, Encoder, Writefile
from utils.preprocess import Tokenizer
from utils.confusables import is_dangeous, alter_word
import argparse
import sys
import csv
def pre_process(inpfile,outfile):
    inpdata=Csvfile(inpfile,textpos=None,firstline=False, split=True)
    splitter = Tokenizer(lowercase=True, allcapskeep=False, normalize=3, 
                         usernames='<user>', urls='<url>', phonenumbers=None, 
                         times=None, numbers=None, hashtags=None, 
                         ignorequotes=False, ignorestopwords=False, emoji=False)
    
    with open(outfile,'wb') as f:
        writer = csv.writer(f)
        for line in inpdata:
            uid, tid, text, label = line
            line = Encoder.str2uni(text)
            sent = splitter.tokenize(line)
    #        sent = [alter_word(word) if is_dangeous(word) else word for word in sent]
            newline = Encoder.uni2str(u' '.join(sent))
            newrow = [uid, tid, newline, label]
            writer.writerow(newrow)
    
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(sys.argv[0])
    
    argparser.add_argument('--inp_file', help='Input file', default="/media/data/langID/small_scale/train/vi.csv.aa", type=str)
    
    argparser.add_argument('--out_file', help='Processed file', default="/media/data/langID/small_scale/train/vi.csv.aap", type=str)
    
    args = argparser.parse_args()
        
    pre_process(args.inp_file,args.out_file)
    
