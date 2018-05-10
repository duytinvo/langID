#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 13:41:03 2018

@author: dtvo
"""
from __future__ import print_function
import sys
import csv
import itertools
import json
import gzip
import cPickle as pickle

class Encoder:
    @staticmethod
    def str2uni(text, encoding='utf8', errors='strict'):
        """Convert `text` to unicode.
    
        Parameters
        ----------
        text : str
            Input text.
        errors : str, optional
            Error handling behaviour, used as parameter for `unicode` function (python2 only).
        encoding : str, optional
            Encoding of `text` for `unicode` function (python2 only).
    
        Returns
        -------
        str
            Unicode version of `text`.
    
        """
        if isinstance(text, unicode):
            return text
        return unicode(text, encoding, errors=errors)

    @staticmethod
    def uni2str(text, errors='strict', encoding='utf8'):
        """Convert utf8 `text` to bytestring.
    
        Parameters
        ----------
        text : str
            Input text.
        errors : str, optional
            Error handling behaviour, used as parameter for `unicode` function (python2 only).
        encoding : str, optional
            Encoding of `text` for `unicode` function (python2 only).
    
        Returns
        -------
        str
            Bytestring in utf8.
    
        """
    
        if isinstance(text, unicode):
            return text.encode('utf8')
        # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
        return unicode(text, encoding, errors=errors).encode('utf8')

class Jfile:
    @staticmethod    
    def load(filename):
        """Loads a JSON data file.
    
        :return: A dict.
        :rtype: dict
        """
        with open(filename, 'r') as file:
            return json.load(file)

    @staticmethod
    def dump(filename, data):
        with open(filename, 'w+') as file:
            return json.dump(data, file)

# Save and load hyper-parameters
class Pfile:
    @staticmethod            
    def dump(args,argfile):

        with gzip.open(argfile, "wb") as fout:
            pickle.dump(args,fout,protocol = pickle.HIGHEST_PROTOCOL)
    @staticmethod
    def load(argfile):
        with gzip.open(argfile, "rb") as fin:
            args = pickle.load(fin)
        return args
    
          
class Txtfile(object):
    def __init__(self, fname, split=False, firstline=False, limit=None, textpos=None):
        self.fname = fname
        self.firstline = firstline
        self.limit = limit
        self.textpos = textpos
        self.split = split
        self.length = None
        
    def __iter__(self):
        with open(self.fname,'rb') as f:
            f.seek(0)
            if self.firstline:
                self.firstline = next(f)
            for line in itertools.islice(f, self.limit):
                line = line.strip()
                if self.textpos is not None:
                    if self.textpos<0:
                        line = line[:self.textpos]
                    else: 
                        line = line[self.textpos:]
                if self.split:
                    yield line.split()
                else:
                    yield line
    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length
            
class Csvfile(object):
    def __init__(self, fname, split=False, firstline=True, limit=None, textpos=None):
        self.fname = fname
        self.firstline = firstline
        self.limit = limit
        self.textpos = textpos
        self.split = split
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
                self.firstline = next(csvreader)
            for line in itertools.islice(csvreader, self.limit):
                if self.textpos is not None:
                    line = line[self.textpos]
                    if self.split:
                        yield line.split()
                    else:
                        yield line 
                else:
                    if self.split:
                        yield line
                    else:
                        yield ' '.join(line)                     
    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length
    
class Writefile:
    @staticmethod                            
    def txtfile(data, fname):
        with open(fname,'wb') as f:
            for i,line in enumerate(data):
                if i!=len(data)-1:
                    f.write(line + '\n')
                else:
                    f.write(line)
    @staticmethod
    def csvfile(data, fname):
        with open(fname,'wb') as f:
            writer = csv.writer(f)
            for i,line in enumerate(data):
                if not isinstance(line,list):
                    line = line.split()
                writer.writerow(line)

