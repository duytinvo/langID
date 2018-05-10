#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:17:43 2018

@author: dtvo
"""
from __future__ import print_function
import os
from other_utils import Jfile

_lexicons = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lexicons/{}')
blocks_data = Jfile.load( _lexicons.format('blocks.json'))
confusables_data = Jfile.load( _lexicons.format('confusables.json'))

def aliases_blocks(chr):
    """Retrieves the script block alias and unicode category for a unicode character.

    >>> categories.aliases_categories('A')
    'LATIN'
    >>> categories.aliases_categories('τ')
    'GREEK', 'L'
    >>> categories.aliases_categories('-')
    'COMMON', 'Pd'

    :param chr: A unicode character
    :type chr: str
    :return: The script block alias and unicode category for a unicode character.
    :rtype: (str, str)
    """
    l = 0
    r = len(blocks_data['code_points_ranges']) - 1
    assert isinstance(chr, unicode) and len(chr) == 1, repr(chr)
    c = ord(chr)

    # binary search
    while r >= l:
        m = (l + r) // 2
        if c < blocks_data['code_points_ranges'][m][0]:
            r = m - 1
        elif c > blocks_data['code_points_ranges'][m][1]:
            l = m + 1
        else:
            return blocks_data['blocks'][blocks_data['code_points_ranges'][m][2]]
    return 'No_Block'
        
def grouping_blocks(string):
    cats = [aliases_blocks(c) for c in string]
    return cats

def is_dangeous(string):
    if len(set(grouping_blocks(string)))>=2:
        return True
    return False

def alter_word(word):
    newword=u""
    for ch in word:
        if ch in confusables_data:
            newword += confusables_data[ch][0][u'c']
            print(ch,confusables_data[ch][0][u'c'])
        else:
            newword += ch
    return newword

                
if __name__ == '__main__':
    tests = [[u'ρæch',"peach"],
             [u"Jꜹ", "jav"],
             [u"𝕔unt", "cunt"],
             [u"ℬitch","bitch"],
             [u"𝒻uck","fuck"],]
    
    words, labels = zip(*tests)
    for word in words:
        if is_dangeous:
            print(alter_word(word))
