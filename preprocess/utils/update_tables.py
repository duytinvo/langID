#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:45:58 2018

@author: dtvo
"""
from __future__ import print_function
import re
import os
from urllib2 import urlopen
from collections import defaultdict
from other_utils import Jfile, Encoder

_lexicons = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lexicons/{}')

def download_unidata(filename,urladd):
    if not os.path.exists(filename):
        print('Unicode data file %s does not exist. Downloading…' % filename)
        file = urlopen(urladd, timeout=None).read().decode('utf-8').split('\n')
        with open(filename, 'wb') as f:
            for line in file:
                f.write(line)
                
def update():
    """
    Update the homoglyph data files from https://www.unicode.org
    """
    block_file = _lexicons.format('UNIDATA_Blocks.txt')
    #    "../metadata/UNIDATA_Blocks.txt"
    script_file = _lexicons.format('UNIDATA_Scripts.txt')
    #    "../metadata/UNIDATA_Scripts.txt"
    confus_file = _lexicons.format('UNIDATA_confusables.txt')
    #    "../metadata/UNIDATA_confusables.txt"

    download_unidata(block_file, 'http://unicode.org/Public/UNIDATA/Blocks.txt')
    download_unidata(script_file, 'http://www.unicode.org/Public/UNIDATA/Scripts.txt')
    download_unidata(confus_file, 'http://www.unicode.org/Public/security/latest/confusables.txt')


   
    generate_blocks(block_file)    
    generate_categories(script_file)
    generate_confusables(confus_file)



def generate_blocks(block_file):
    """Generates the categories JSON data file from the unicode specification.

    :return: True for success, raises otherwise.
    :rtype: bool
    """
    # inspired by https://gist.github.com/anonymous/2204527
    code_points_ranges = []
    blocks = []

    match = re.compile(r'([0-9A-F]+)\.\.([0-9A-F]+);\ (\S.*\S)', re.UNICODE)

    with open(block_file,"rb") as f:
        for line in f:
            line = Encoder.str2uni(line)
            p = re.findall(match, line)
            if p:
                code_point_range_from, code_point_range_to, block_name = p[0]
                if block_name == 'No_Block':
                    continue
                block_name = block_name.upper()
                if block_name not in blocks:
                    blocks.append(block_name)
                code_points_ranges.append((
                    int(code_point_range_from, 16),
                    int(code_point_range_to, 16),
                    blocks.index(block_name)))
    code_points_ranges.sort()

    blocks_data = {
        'blocks': blocks,
        'code_points_ranges': code_points_ranges,
    }
    Jfile.dump(_lexicons.format('blocks.json'), blocks_data)
    
def generate_categories(script_file):
    """Generates the categories JSON data file from the unicode specification.

    :return: True for success, raises otherwise.
    :rtype: bool
    """
    # inspired by https://gist.github.com/anonymous/2204527
    code_points_ranges = []
    iso_15924_aliases = []
    categories = []

    match = re.compile(r'([0-9A-F]+)(?:\.\.([0-9A-F]+))?\W+(\w+)\s*#\s*(\w+)',
                       re.UNICODE)

    with open(script_file,'rb') as f:
        for line in f:
            line = Encoder.str2uni(line)
            p = re.findall(match, line)
            if p:
                code_point_range_from, code_point_range_to, alias, category = p[0]
                alias = alias.upper()
                if alias not in iso_15924_aliases:
                    iso_15924_aliases.append(alias)
                if category not in categories:
                    categories.append(category)
                code_points_ranges.append((
                    int(code_point_range_from, 16),
                    int(code_point_range_to or code_point_range_from, 16),
                    iso_15924_aliases.index(alias), categories.index(category))
                )
    code_points_ranges.sort()

    categories_data = {
        'iso_15924_aliases': iso_15924_aliases,
        'categories': categories,
        'code_points_ranges': code_points_ranges,
    }

    Jfile.dump(_lexicons.format('categories.json'), categories_data)



def generate_confusables(confus_file):
    """Generates the confusables JSON data file from the unicode specification.

    :return: True for success, raises otherwise.
    :rtype: bool
    """
    confusables_matrix = defaultdict(list)
    match = re.compile(ur'[0-9A-F ]+\s+;\s*[0-9A-F ]+\s+;\s*\w+\s*#'
                       ur'\*?\s*\( (.+) → (.+) \) (.+) → (.+)\t#',
                       re.UNICODE)
    with open(confus_file,'rb') as f:
        for line in f:
            line = Encoder.str2uni(line)
            p = re.findall(match, line)
            if p:
                char1, char2, name1, name2 = p[0]
                confusables_matrix[char1].append({
                    'c': char2,
                    'n': name2,
                })
#                confusables_matrix[char2].append({
#                    'c': char1,
#                    'n': name1,
#                })

    Jfile.dump(_lexicons.format('confusables.json'), dict(confusables_matrix))
    
if __name__ == '__main__':
    update()
