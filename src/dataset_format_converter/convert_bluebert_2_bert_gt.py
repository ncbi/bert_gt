#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:36:41 2020

@author: Poting
"""

import os
import random
import spacy
import scispacy
import re

class DataInstance:
    
    def __init__(self, index, text, label):
        self.index = index
        self.text = text
        self.label = label
        self.tokens = []
        self.in_neighbors = []

def load_bluebert_dataset_2_instances(
        in_bluebert_dataset_file):
    
    data_instances = []
    
    with open(in_bluebert_dataset_file, 'r', encoding='utf8') as tsv_reader:
        tsv_reader.readline() # skip header
        for line in tsv_reader:
            tks = line.rstrip().split('\t')
            index = tks[0]
            text = tks[1]
            label = tks[-1]
            
            data_instances.append(DataInstance(index,
                                            text,
                                            label))
    return data_instances

def add_neighbors_info_2_instances_by_spacy(
        data_instances,
        tag_2_word_dict,
        spacy_model):
    
    nlp = spacy.load(spacy_model)
    
    word_2_tag_dict = {}
    
    for tag, word in tag_2_word_dict.items():
        word_2_tag_dict[word] = tag
    
    for data_instance in data_instances:
        
        
        tokens = data_instance.text.split()
        for i, token in enumerate(tokens):
            if token in tag_2_word_dict:
                tokens[i] = tag_2_word_dict[token]
        sent = ' '.join(tokens)
        
        doc = nlp(re.sub(r'\s+', ' ', sent))
                
        for i, token in enumerate(doc):
            
            if token.text in word_2_tag_dict:
                data_instance.tokens.append(word_2_tag_dict[token.text])
            else:
                data_instance.tokens.append(token.text)
            data_instance.in_neighbors.append(token.head.i)
            
            
def dump_instances_2_bert_gt_format(
        data_instances, 
        out_train_file, 
        is_test_set):
    
    
    with open(out_train_file, 'w', encoding='utf8') as bert_writer:
        
        if is_test_set:
            bert_writer.write('index\tsentence\tin_neighbors\tlabel\n')
            
        for data_instance in data_instances:
            _str_index = str(data_instance.index)
            _str_tokens = ' '.join(data_instance.tokens)
            _str_in_neighbors = ' '.join(str(x) for x in data_instance.in_neighbors)
            _str_label = str(data_instance.label)
            
            _str_out = _str_index
            _str_out += '\t' + _str_tokens
            _str_out += '\t' + _str_in_neighbors
            _str_out += '\t' + _str_label + '\n'
            
            bert_writer.write(_str_out)
        

def generate_train_dev_test_data(
        in_bluebert_dataset_dir,
        out_bert_dir,
        spacy_model,
        tag_2_word_dict):
    
    if not os.path.exists(out_bert_dir):
        os.makedirs(out_bert_dir)
    
    in_train_file = in_bluebert_dataset_dir + 'train.tsv'
    in_dev_file = in_bluebert_dataset_dir + 'dev.tsv'
    in_test_file = in_bluebert_dataset_dir + 'test.tsv'
    
    out_train_file = out_bert_dir + 'train.tsv'
    out_dev_file = out_bert_dir + 'dev.tsv'
    out_test_file = out_bert_dir + 'test.tsv'
    
    
    train_instances = load_bluebert_dataset_2_instances(in_train_file)
    train_instances.extend(load_bluebert_dataset_2_instances(in_dev_file))
    test_instances = load_bluebert_dataset_2_instances(in_test_file)
    
    add_neighbors_info_2_instances_by_spacy(train_instances, tag_2_word_dict, spacy_model)
    add_neighbors_info_2_instances_by_spacy(test_instances, tag_2_word_dict, spacy_model)
    
    random.seed(1234)
    random.shuffle(train_instances)
    dev_instances = train_instances[:200]
    train_instances = train_instances[200:]
    
    dump_instances_2_bert_gt_format(
        train_instances, 
        out_train_file, 
        is_test_set=False)
    
    dump_instances_2_bert_gt_format(
        dev_instances, 
        out_dev_file, 
        is_test_set=False)
    
    dump_instances_2_bert_gt_format(
        test_instances, 
        out_test_file, 
        is_test_set=True)

if __name__ == '__main__':
    
    spacy_model     = 'en_core_sci_md'
    
    in_bluebert_dataset_dir = 'datasets/ddi2013-type/'
    out_bert_dir    = 'datasets/ddi2013-type/processed/'
    tag_2_word_dict = {'@DRUG$': 'DRUG12AB'}
    
    generate_train_dev_test_data(
        in_bluebert_dataset_dir   = in_bluebert_dataset_dir,
        out_bert_dir              = out_bert_dir,
        spacy_model               = spacy_model,
        tag_2_word_dict           = tag_2_word_dict)
    
    in_bluebert_dataset_dir = 'datasets/ChemProt/'
    out_bert_dir    = 'datasets/ChemProt/processed/'
    tag_2_word_dict = {'@GENE$': 'GENE12AB', '@CHEMICAL$': 'CHEM12AB'}
    
    generate_train_dev_test_data(
        in_bluebert_dataset_dir   = in_bluebert_dataset_dir,
        out_bert_dir              = out_bert_dir,
        spacy_model               = spacy_model,
        tag_2_word_dict           = tag_2_word_dict)