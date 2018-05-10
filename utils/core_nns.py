#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:41:43 2018

@author: dtvo
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
use_cuda = torch.cuda.is_available()

class Embs(nn.Module):
    """
    This module is used to build an embeddings layer with BiLSTM model
    """
    def __init__(self, HPs):
        super(Embs, self).__init__()
        [size, dim, pre_embs, hidden_dim, dropout, layers, bidirect, zeros] = HPs
        self.layers = layers
        self.bidirect = bidirect
        self.zeros = zeros
        self.hidden_dim = hidden_dim // 2 if bidirect else hidden_dim
            
        self.embeddings = nn.Embedding(size, dim)
        if pre_embs is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pre_embs))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(size, dim)))

        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(dim, self.hidden_dim, num_layers=layers, batch_first=True, bidirectional=bidirect)
        
        self.att_layer = nn.Linear(hidden_dim,1, bias=False)
        self.softmax = nn.Softmax(-1)
        if use_cuda:
            self.embeddings = self.embeddings.cuda()
            self.drop = self.drop.cuda()
            self.lstm = self.lstm.cuda()
            self.att_layer = self.att_layer.cuda()
            self.softmax = self.softmax.cuda()
            
    def forward(self, inputs, input_lengths):
        return self.get_all_hiddens(inputs, input_lengths)

    def get_last_hiddens(self, inputs, input_lengths):
        """
            input:  
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        # set zero vector for padding, unk, eot, sot
        if self.zeros:
            self.set_zeros([0,1,2,3])
        batch_size = inputs.size(0)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        hc_0 = self.initHidden(batch_size)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)
        return  h_n

    def get_all_hiddens(self, inputs, input_lengths=None):
        """
            input:  
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        # set zero vector for padding, unk, eot, sot
        if self.zeros:    
            self.set_zeros([0,1,2,3])
        batch_size = inputs.size(0)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        hc_0 = self.initHidden(batch_size)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        return  rnn_out

    def get_last_atthiddens(self, inputs, input_lengths=None):
        """
            input:  
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        # set zero vector for padding, unk, eot, sot
        if self.zeros:
            self.set_zeros([0,1,2,3])
        batch_size = inputs.size(0)
        word_length = inputs.size(1)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        hc_0 = self.initHidden(batch_size)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)
                
        #(batch_size, word_length, 1)
        att_features = F.relu(self.att_layer(rnn_out))
         #(batch_size, word_length)
        att_features.squeeze_()
        alpha = self.softmax(att_features)
        att_embs = embs_drop*alpha.view(batch_size,word_length,1)
        att_h = att_embs.sum(1)
        features = torch.cat([h_n, att_h], -1)
        return  features
    
    def random_embedding(self, size, dim):
        pre_embs = np.empty([size, dim])
        scale = np.sqrt(3.0 / dim)
        for index in range(size):
            pre_embs[index,:] = np.random.uniform(-scale, scale, [1, dim])
        return pre_embs

    def initHidden(self, batch_size):
        d = 2 if self.bidirect else 1
        h = Variable(torch.zeros(self.layers*d, batch_size, self.hidden_dim))
        c = Variable(torch.zeros(self.layers*d, batch_size, self.hidden_dim))
        if use_cuda:
            return h.cuda(), c.cuda()
        else:
            return h,c
        
    def set_zeros(self,idx):
        for i in idx:
            self.embeddings.weight.data[i].fill_(0)

class BiLSTM(nn.Module):
    def __init__(self, word_HPs=None, num_labels = None):
        super(BiLSTM, self).__init__()
        [word_size, word_dim, wd_embeddings, word_hidden_dim, word_dropout, word_layers, word_bidirect, zeros] = word_HPs
        self.num_labels = num_labels
        self.lstm = Embs(word_HPs)
        self.dropfinal = nn.Dropout(word_dropout)
        if num_labels > 2:
            self.hidden2tag = nn.Linear(word_hidden_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag = nn.Linear(word_hidden_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()            

        if use_cuda:
            self.dropfinal = self.dropfinal.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.lossF = self.lossF.cuda()
    
    def forward(self, word_tensor, word_lengths):  
        word_h_n = self.lstm.get_last_hiddens(word_tensor, word_lengths)
        label_score = self.hidden2tag(word_h_n)
        label_score = self.dropfinal(label_score)
        return label_score

    def repesentation(self, word_tensor, word_lengths):  
        word_h_n = self.lstm.get_last_hiddens(word_tensor, word_lengths)
        return word_h_n
    
    def NLL_loss(self, label_score, label_tensor):  
        if self.num_labels > 2:
            batch_loss = self.lossF(label_score, label_tensor)
        else:
            batch_loss = self.lossF(label_score, label_tensor.float().view(-1,1))
        return batch_loss  

    def inference(self, label_score, k=1):
        if self.num_labels > 2:
            label_prob = F.softmax(label_score, dim=-1)
            label_prob, label_pred = label_prob.data.topk(k)
        else:
            label_prob = F.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred
        
if __name__ == "__main__":
    from data_utils import Data2tensor, Vocab, seqPAD, Csvfile
    filename = "/media/data/langID/small_scale/train.csv"
    vocab = Vocab(cl_th=None, cutoff=1, c_lower=False, c_norm=False)
    vocab.build([filename], firstline=False)
    word2idx = vocab.wd2idx(vocab.c2i)
    tag2idx = vocab.tag2idx(vocab.l2i)
    train_data = Csvfile(filename, firstline=False, word2idx=word2idx, tag2idx=tag2idx)
        
    train_iters = Vocab.minibatches(train_data, batch_size=10)
    data=[]
    label_ids = []
    for words, labels in train_iters:
        data.append(words)
        label_ids.append(labels)
        word_ids, sequence_lengths = seqPAD.pad_sequences(words, pad_tok=0, wthres=1024, cthres=32)
    
    w_tensor=Data2tensor.idx2tensor(word_ids)
    y_tensor=Data2tensor.idx2tensor(labels)
    
    data_tensors = Data2tensor.sort_tensors(labels, word_ids, sequence_lengths)
    label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors  