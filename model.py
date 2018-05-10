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
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.core_nns import BiLSTM as fNN
from utils.other_utils import Progbar, Timer, SaveloadHP
from utils.data_utils import Vocab, Data2tensor, Embeddings, Csvfile, seqPAD

use_cuda = torch.cuda.is_available()
seed_num = 12345
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
    
class Classifier(object):
    def __init__(self, args=None):
        self.args = args  
        word_layers = 1
        word_bidirect = True
        char_HPs = [len(self.args.vocab.c2i), self.args.char_dim, self.args.char_pred_embs, self.args.char_hidden_dim, self.args.dropout, word_layers, word_bidirect, self.args.c_zeros]
        
        self.model = fNN(word_HPs=char_HPs, num_labels=len(self.args.vocab.l2i))

        if args.optimizer.lower() == "adamax":
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        
        self.word2idx = self.args.vocab.wd2idx(vocab_chars=self.args.vocab.c2i, allow_unk=True, start_end=self.args.start_end)
        self.tag2idx = self.args.vocab.tag2idx(vocab_tags=self.args.vocab.l2i)
        
    def evaluate_batch(self, eva_data):
        cl = self.args.vocab.cl    
        
        batch_size = self.args.batch_size  
         ## set model in eval model
        self.model.eval()
        num_label = 0
        num_correct = 0
        for i,(words, label_ids) in enumerate(self.args.vocab.minibatches(eva_data, batch_size=batch_size)):
            word_ids, sequence_lengths = seqPAD.pad_sequences(words, pad_tok=0, wthres=cl)
            data_tensors = Data2tensor.sort_tensors(label_ids, word_ids,sequence_lengths, volatile_flag=True)
            label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors 
            
            label_score = self.model(word_tensor, sequence_lengths)
            label_prob, label_pred = self.model.inference(label_score, k=1)
                
            assert len(label_pred)==len(label_tensor)
            correct_pred = (label_pred.squeeze()==label_tensor.data).sum()
            assert correct_pred <=batch_size  
            num_label += len(label_tensor)
            num_correct += correct_pred
        acc = num_correct/num_label  
        return acc 

    def train_batch(self,train_data):
        cl = self.args.vocab.cl 
        clip_rate = self.args.clip
        
        batch_size = self.args.batch_size
        num_train = len(train_data)
        total_batch = num_train//batch_size+1
        prog = Progbar(target=total_batch)
        ## set model in train model
        self.model.train()
        train_loss = []
        for i,(words, label_ids) in enumerate(self.args.vocab.minibatches(train_data, batch_size=batch_size)):
            word_ids, sequence_lengths = seqPAD.pad_sequences(words, pad_tok=0, wthres=cl)
            data_tensors = Data2tensor.sort_tensors(label_ids, word_ids,sequence_lengths)
            label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors 

            label_score = self.model(word_tensor, sequence_lengths)
            batch_loss = self.model.NLL_loss(label_score, label_tensor)

            train_loss.append(batch_loss.data.tolist()[0])
            self.model.zero_grad()
            batch_loss.backward()
            
            if clip_rate>0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), clip_rate)
                
            self.optimizer.step()
            
            prog.update(i + 1, [("Train loss", batch_loss.data.tolist()[0])])
        return np.mean(train_loss)

    def lr_decay(self, epoch):
        lr = self.args.lr/(1+self.args.decay_rate*epoch)
        print("INFO: - Learning rate is setted as: %f"%lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):            
        train_data = Csvfile(self.args.train_file, firstline=False, word2idx=self.word2idx, tag2idx=self.tag2idx)
        dev_data = Csvfile(self.args.dev_file, firstline=False, word2idx=self.word2idx, tag2idx=self.tag2idx)
        test_data = Csvfile(self.args.test_file, firstline=False, word2idx=self.word2idx, tag2idx=self.tag2idx)
    
        max_epochs = self.args.max_epochs
        best_dev = -1
        nepoch_no_imprv = 0
        epoch_start = time.time()
        for epoch in xrange(max_epochs):
            self.lr_decay(epoch)
            print("Epoch: %s/%s" %(epoch,max_epochs))
            train_loss = self.train_batch(train_data)
            # evaluate on developing data
            dev_metric = self.evaluate_batch(dev_data)
            if dev_metric > best_dev:
                nepoch_no_imprv = 0
                torch.save(self.model.state_dict(), self.args.model_name)
                best_dev = dev_metric 
                print("\nUPDATES: - New improvement")
#                test_metric = self.evaluate_batch(test_data)
                print("         - Train loss: %4f"%train_loss)
                print("         - Dev acc: %2f"%(100*best_dev))
#                print("         - Test acc: %2f"%(100*test_metric))                
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.args.patience:
                    print("\nSUMMARY: - Early stopping after %d epochs without improvements"%(nepoch_no_imprv))
                    test_metric = self.evaluate_batch(test_data)
                    print("         - Train loss: %4f"%train_loss)
                    print("         - Dev acc: %2f"%(100*best_dev))
                    print("         - Test acc: %2f"%(100*test_metric))
                    return

            epoch_finish = Timer.timeEst(epoch_start,(epoch+1)/max_epochs)
            print("\nINFO: - Trained time(Remained time): %s; - Dev acc: %.2f"%(epoch_finish,100*dev_metric))

        print("\nSUMMARY: - Completed %d epoches"%(max_epochs))
        test_metric = self.evaluate_batch(test_data)
        print("         - Train loss: %4f"%train_loss)
        print("         - Dev acc: %2f"%(100*best_dev))
        print("         - Test acc: %2f"%(100*test_metric))
        
        return 

    def predict(self, sent, k=1):
        cl = self.args.vocab.cl            
         ## set model in eval model
        self.model.eval()
        
        fake_label = [0]        
        words = self.word2idx(sent)
        word_ids, sequence_lengths = seqPAD.pad_sequences([words], pad_tok=0, wthres=cl)
    
        data_tensors = Data2tensor.sort_tensors(fake_label, word_ids,sequence_lengths, volatile_flag=True)    
        fake_label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors
        label_score = self.model(word_tensor, sequence_lengths)
        label_prob, label_pred = self.model.inference(label_score, k)
        return label_prob, label_pred 


def build_data(args):    
    print("Building dataset...")
    model_dir, _ = os.path.split(args.model_args)
    if not os.path.exists(model_dir): 
        os.mkdir(model_dir)

    vocab = Vocab(cl_th=args.char_thres, cutoff=args.cutoff, c_lower=args.c_lower, c_norm=args.c_norm)
    vocab.build([args.train_file,args.dev_file,args.test_file], firstline=False)
    args.vocab = vocab    
    if args.c_pre_trained:
        scale = np.sqrt(3.0 / args.char_dim)
        args.char_pred_embs = Embeddings.get_W(args.c_emb_file, args.char_dim,vocab.c2i, scale)
    else:
        args.char_pred_embs = None  
    SaveloadHP.save(args, args.model_args)
    return args

if __name__ == '__main__':
    """
    python model.py --char_thres 64 --emb_file /media/data/embeddings/glove/glove.6B.300d.txt --word_dim 300 --word_hidden_dim 300 --char_dim 100 --char_hidden_dim 200 --patience 16
    """
    argparser = argparse.ArgumentParser(sys.argv[0])
    
    argparser.add_argument('--train_file', help='Trained file', default="/media/data/langID/small_scale/train.csv", type=str)
    
    argparser.add_argument('--dev_file', help='Developed file', default="/media/data/langID/small_scale/dev.csv", type=str)
    
    argparser.add_argument('--test_file', help='Tested file', default="/media/data/langID/small_scale/test.csv", type=str)
                        
    argparser.add_argument("--cutoff", type = int, default = 5, help = "prune words ocurring <= cutoff")  
    
    argparser.add_argument("--char_thres", type = int, default = None, help = "char threshold")
    
    argparser.add_argument("--c_lower", action='store_true', default = False, help = "lowercase characters")
    
    argparser.add_argument("--c_norm", action='store_true', default = False, help = "number-norm characters")
    
    argparser.add_argument("--c_zeros", action='store_true', default = False, help = "set zeros to padc, unkc, soc, eoc")
    
    argparser.add_argument("--start_end", action='store_true', default = False, help = "start-end paddings")
    
    argparser.add_argument("--c_emb_file", type = str, default = "/media/data/embeddings/glove/glove.6B.100d.txt", help = "embedding file")
    
    argparser.add_argument("--c_pre_trained", type = int, default = 0, help = "Use pre-trained embedding or not")
    
    argparser.add_argument("--char_dim", type = int, default = 50, help = "char embedding size")
        
    argparser.add_argument("--char_hidden_dim", type = int, default = 100, help = "char LSTM layers")
                
    argparser.add_argument("--dropout", type = float, default = 0.5, help = "dropout probability")
    
    argparser.add_argument("--patience", type = int, default = 32, help = "early stopping")
            
    argparser.add_argument("--optimizer", type = str, default = "SGD", help = "learning method (adagrad, sgd, ...)")
    
    argparser.add_argument("--lr", type = float, default = 0.02, help = "learning rate") 
    
    argparser.add_argument("--decay_rate", type = float, default = 0.05, help = "decay learning rate")
        
    argparser.add_argument("--max_epochs", type = int, default = 512, help = "maximum # of epochs")
    
    argparser.add_argument("--batch_size", type = int, default = 32, help = "mini-batch size")  
    
    argparser.add_argument('--clip', help='Clipping value', default=5, type=int)
        
    argparser.add_argument('--model_name', help='Model dir', default="./results/small.bilstm.m", type=str)
    
    argparser.add_argument('--model_args', help='Model dir', default="./results/small.bilstm.args.pklz", type=str)
    
    args = argparser.parse_args()
    
    args = build_data(args)
    
    classifier = Classifier(args)    

    classifier.train() 
    
    

    
    





