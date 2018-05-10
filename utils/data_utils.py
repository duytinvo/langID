#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:31:21 2018

@author: dtvo
"""
from __future__ import print_function
from __future__ import division

import csv
import sys
import torch
import itertools
import numpy as np
from torch.autograd import Variable
from collections import Counter
from other_utils import Encoder
    
PADc = u"<PADc>"
UNKc = u"<UNKc>"
SOc = u"<sc>"
EOc = u"</sc>"
PADw = u"<PADw>"
UNKw = u"<UNKw>"
SOw = u"<sw>"
EOw = u"</sw>"

class Vocab(object):
    def __init__(self, cl_th=None, cutoff=1, c_lower=False, c_norm=False):
        self.c2i = {}
        self.l2i = {}
        self.cl = cl_th
        self.c_lower = c_lower
        self.c_norm = c_norm
        self.cutoff = cutoff
                        
    def build(self, files, firstline=False, limit=-1):
        lcnt = Counter()
        ccnt = Counter()
        print("Extracting vocabulary:")
        cl=0
        for fname in files:
            raw=Csvfile(fname, firstline=firstline, limit=limit)  
            for sent,label in raw:
                ccnt.update(sent)
                cl=max(cl,len(sent))
                
                lcnt.update([label])
                
        print("\t%d total characters, %d total labels" % (sum(ccnt.values()),sum(lcnt.values())))
        
        clst=[x for x, y in ccnt.iteritems() if y >= self.cutoff]
        clst = [PADc, UNKc, SOc, EOc] + clst
        cvocab = dict([ (y,x) for x,y in enumerate(clst) ])
        
        lvocab = dict([ (y,x) for x,y in enumerate(lcnt.keys()) ])
        print("\t%d unique characters, %d unique labels" % (len(ccnt), len(lcnt)))
        print("\t%d unique characters appearing at least %d times" % (len(cvocab)-4, self.cutoff))
        self.c2i = cvocab
        self.l2i = lvocab 
        if self.cl is None:
            self.cl = cl
        else:
            self.cl = min(cl, self.cl)

    def wd2idx(self, vocab_chars=None, allow_unk=True, start_end=False):
        '''
        Return a function to convert tag2idx or word/char2idx
        '''
        def f(sent):                 
            if vocab_chars is not None:
                # SOc,EOc charcters for  SOW
                char_ids = []
                for char in sent:
                    # ignore chars out of vocabulary
                    if char in vocab_chars:
                        char_ids += [vocab_chars[char]]
                    else:
                        if allow_unk:
                             char_ids += [vocab_chars[UNKc]]
                        else:
                            raise Exception("Unknow key is not allowed. Check that "\
                                            "your vocab (tags?) is correct")  
                if start_end:
                    # SOc,EOc charcters for  EOW
                    char_ids += [vocab_chars[SOc]] + char_ids + [vocab_chars[EOc]]                
            return char_ids
        return f 

    @staticmethod
    def tag2idx(vocab_tags=None):
        def f(tags): 
            if tags in vocab_tags:
                tag_ids = vocab_tags[tags]
            else:
                raise Exception("Unknow key is not allowed. Check that "\
                                                "your vocab (tags?) is correct")  
            return tag_ids
        return f
        
    @staticmethod
    def minibatches(data, batch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)
    
        Yields:
            list of tuples
    
        """
        x_batch, y_batch = [], []
        for (x, y) in data:
            if len(x_batch) == batch_size:
                # yield a tuple of list ([wd_ch_i], [label_i])
                yield x_batch, y_batch
                x_batch, y_batch = [], []
            # if use char, decompose x into wd_ch_i=[([char_ids],...[char_ids]),(word_ids)]
            if type(x[0]) == tuple:
                x = zip(*x)
            x_batch += [x]
            y_batch += [y]
    
        if len(x_batch) != 0:
            yield x_batch, y_batch

class Csvfile(object):
    """
    Read cvs file
    """
    def __init__(self, fname, word2idx=None, tag2idx=None, firstline=True, limit=-1):
        self.fname = fname
        self.firstline = firstline
        if limit <0:
            self.limit = None
        else:
            self.limit = limit
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.length = None
        
    def __iter__(self):
        maxInt = sys.maxsize
        decrement = True
        while decrement:
            # decrease the maxInt value by factor 10 
            # as long as the OverflowError occurs.
            decrement = False
            try:
                csv.field_size_limit(maxInt)
            except OverflowError:
                maxInt = int(maxInt/10)
                decrement = True
        with open(self.fname,'rb') as f:
            f.seek(0)
            csvreader = csv.reader(f)
            if self.firstline:
                # Skip the header
                next(csvreader)
            for line in itertools.islice(csvreader, self.limit):
                _,_,sent,tag= line
                sent = Encoder.str2uni(sent)
                tag = Encoder.str2uni(tag)
                if self.word2idx is not None:
                    # return a list [word_id1, ..., word_idm] if don't use char
                    # else return a list of tuple [([char_id1, ..., char_idn],wd_id1), ..., ([char_id1, ..., char_idn],wd_idm)]
                    sent = self.word2idx(sent)
                if self.tag2idx is not None:
                    tag = self.tag2idx(tag)
                # yield a tuple (words, tag)
                yield sent, tag
                
    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length
                          
class seqPAD:
    @staticmethod
    def _pad_sequences(sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
    
        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []
    
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded +=  [seq_]
            sequence_length += [min(len(seq), max_length)]
    
        return sequence_padded, sequence_length

    @staticmethod
    def pad_sequences(sequences, pad_tok, nlevels=1, wthres=1024, cthres=32):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
            nlevels: "depth" of padding, for the case where we have characters ids
    
        Returns:
            a list of list where each sublist has same length
    
        """
        if nlevels == 1:
            max_length = max(map(lambda x : len(x), sequences))
            max_length = min(wthres,max_length)
            sequence_padded, sequence_length = seqPAD._pad_sequences(sequences, pad_tok, max_length)
    
        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            max_length_word = min (cthres,max_length_word)
            sequence_padded, sequence_length = [], []
            for seq in sequences:
                # pad the character-level first to make the word length being the same
                sp, sl = seqPAD._pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [sl]
            # pad the word-level to make the sequence length being the same
            max_length_sentence = max(map(lambda x : len(x), sequences))
            max_length_sentence = min(wthres,max_length_sentence)
            sequence_padded, _ = seqPAD._pad_sequences(sequence_padded, [pad_tok]*max_length_word, max_length_sentence)
            # set sequence length to 1 by inserting padding 
            sequence_length, _ = seqPAD._pad_sequences(sequence_length, 1, max_length_sentence)
    
        return sequence_padded, sequence_length
    
    @staticmethod
    def pad_labels(y, nb_classes=None):
        '''Convert class vector (integers from 0 to nb_classes)
        to binary class matrix, for use with categorical_crossentropy.
        '''
        if not nb_classes:
            nb_classes = max(y)+1
        Y=[[0]*nb_classes for i in xrange(len(y))]
        for i in range(len(y)):
            Y[i][y[i]] = 1
        return Y 
    
class Embeddings:
    @staticmethod
    def load_embs(fname):
        embs=dict()
        s=0
        V=0
        with open(fname,'rb') as f:
            for line in f: 
                p=line.strip().split()
                if len(p)==2:
                    V=int(p[0]) ## Vocabulary
                    s=int(p[1]) ## embeddings size
                else:
#                    assert len(p)== s+1
                    w=p[0]
                    e=[float(i) for i in p[1:]]
                    embs[w]=np.array(e,dtype="float32")
#        assert len(embs)==V
        return embs 
    
    @staticmethod
    def get_W(emb_file, wsize, vocabx, scale=0.25):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        print("Extracting pretrained embeddings:")
        word_vecs =Embeddings.load_embs(emb_file)
        print(('\t%d pre-trained word embeddings')%(len(word_vecs)))
        print('Mapping to vocabulary:')
        unk=0
        part = 0
        W = np.zeros(shape=(len(vocabx), wsize),dtype="float32")            
        for word,idx in vocabx.iteritems():
            if idx ==0:
                continue
            if word_vecs.get(word) is not None:
                W[idx]=word_vecs.get(word)
            else:
                if word_vecs.get(word.lower()) is not None:
                    W[idx]=word_vecs.get(word.lower())
                    part += 1
                else:
                    unk+=1
                    rvector=np.asarray(np.random.uniform(-scale,scale,(1,wsize)),dtype="float32")
                    W[idx]=rvector
        print('\t%d randomly word vectors;'%unk)
        print('\t%d partially word vectors;'%part)
        print('\t%d pre-trained embeddings.'%(len(vocabx)-unk-part))
        return W

    @staticmethod
    def init_W(wsize, vocabx, scale=0.25):
        """
        Randomly initial word vectors between [-scale, scale]
        """
        W = np.zeros(shape=(len(vocabx), wsize),dtype="float32")            
        for word,idx in vocabx.iteritems():
            if idx ==0:
                continue
            rvector=np.asarray(np.random.uniform(-scale,scale,(1,wsize)),dtype="float32")
            W[idx]=rvector
        return W

class Data2tensor:
    @staticmethod
    def idx2tensor(indexes, volatile_flag=False):
        result = Variable(torch.LongTensor(indexes), volatile=volatile_flag)
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result

    @staticmethod
    def sort_tensors(label_ids, word_ids, sequence_lengths, volatile_flag=False):        
        label_tensor=Data2tensor.idx2tensor(label_ids, volatile_flag)
        word_tensor=Data2tensor.idx2tensor(word_ids, volatile_flag)
        sequence_lengths = Data2tensor.idx2tensor(sequence_lengths, volatile_flag)
        
        sequence_lengths, word_perm_idx = sequence_lengths.sort(0, descending=True)
        
        word_tensor = word_tensor[word_perm_idx]
        label_tensor = label_tensor[word_perm_idx]
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)
        return label_tensor, word_tensor, sequence_lengths, word_seq_recover

          
if __name__ == "__main__":
    filename = "/media/data/langID/large_scale/p.train.csv"
    vocab = Vocab(cl_th=None, cutoff=5, c_lower=False, c_norm=False)
    vocab.build([filename], firstline=False)
    word2idx = vocab.wd2idx(vocab.c2i)
    tag2idx = vocab.tag2idx(vocab.l2i)
    train_data = Csvfile(filename, firstline=False, word2idx=word2idx, tag2idx=tag2idx)
        
    train_iters = Vocab.minibatches(train_data, batch_size=10)
    data=[]
    label_ids = []
    i=0
    flag=[]
    for words, labels in train_data:
        i+=1
        if len(words)<=0:
            data.append(words)
            label_ids.append(labels)
            flag.append(i)
#        data.append(words)
#        label_ids.append(labels)
#        word_ids, sequence_lengths = seqPAD.pad_sequences(words, pad_tok=0, wthres=1024, cthres=32)
    
#    w_tensor=Data2tensor.idx2tensor(word_ids)
#    y_tensor=Data2tensor.idx2tensor(labels)
    
#    data_tensors = Data2tensor.sort_tensors(labels, word_ids, sequence_lengths)
#    label_tensor, word_tensor, sequence_lengths, word_seq_recover = data_tensors  
    
