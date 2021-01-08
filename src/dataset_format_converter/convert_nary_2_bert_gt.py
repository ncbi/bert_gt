# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 14:31:03 2019

@author: laip2
"""
    

import json
import re
import numpy as np
import os
import random

from enum import Enum


class NEType(str, Enum):
    NONE = 'NONE'
    DRUG = 'DRUG'
    DISEASE = 'DISEASE'
    CHEMICAL = 'CHEMICAL'
    GENE = 'GENE'
    VARIANT = 'VARIANT'
    

class RelationType(str, Enum):
    NO = 'NO'
    YES = 'YES'

class AnnotationInstance:

    def __init__(self, start, end, ne_type):
        self.start = start
        self.end = end
        self.type = ne_type
        self.text = ''

class NaryInstance:
    
    def __init__(self):
        self.docid = ''
        self.id = ''
        
        self.ne_list = []
        
        self.relation = RelationType.NO
        
        self.token = [] # [string, …, string]
        self.pos = [] # [string, …, string]
        self.deprel = [] # either [string, …, string] or [[string,..,string], …, []]
        self.head = [] # either [int, …, int]  or [[int,..,int], …, []]
        self.ner = [] # [string, …, string]


def convert_iob2_to_tagged_sent(
        tokens, 
        labels, 
        in_neighbors):
    
    tagged_sent = ''
    
    in_neighbors_list = []
    
    num_orig_tokens = len(tokens)
    
    previous_label = 'O'
    
    orig_token_index_2_new_token_index_mapping = []
    
    current_idx = -1
    ne_text = ''
    ne_list = []
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == 'O':
            if previous_label != 'O':
                tagged_sent += '$ ' + token
                ne_list.append(ne_text)
                ne_text = ''
            else:
                tagged_sent += ' ' + token
            current_idx += 1
        elif label.startswith('B-'):
            if previous_label[2:] == label[2:]:
                current_idx = current_idx # do nothing
                ne_text = ' ' + token
            elif previous_label != 'O':
                tagged_sent += '$ @' + label[2:]
                current_idx += 1
                ne_list.append(ne_text)
                ne_text = token
            else:
                tagged_sent += ' @' + label[2:]
                ne_text = token
                current_idx += 1
        elif label.startswith('I-'):
            ne_text = ' ' + token
        previous_label = label
        orig_token_index_2_new_token_index_mapping.append(current_idx)
    
    if ne_text != '':
        ne_list.append(ne_text)
        ne_text = ''
    
    previous_idx = 0
    _token_in_neighbors_list = []
    
    _tokens = tagged_sent.split(' ')
    
    shift_point_indices = [] 
    for i in range(len(_tokens)):
        token = _tokens[i]
        if token.startswith('@') and token.endswith('$'):
            shift_point_indices.append(i)
    
    for i in range(num_orig_tokens):
        if previous_idx == orig_token_index_2_new_token_index_mapping[i]:
            for neighbor_idx in in_neighbors[i]:
                _token_in_neighbors_list.append(orig_token_index_2_new_token_index_mapping[neighbor_idx])
        else:
            in_neighbors_list.append(list(set(_token_in_neighbors_list)))
            _token_in_neighbors_list = []
            for neighbor_idx in in_neighbors[i]:
                _token_in_neighbors_list.append(orig_token_index_2_new_token_index_mapping[neighbor_idx])
        previous_idx = orig_token_index_2_new_token_index_mapping[i]
    in_neighbors_list.append(list(set(_token_in_neighbors_list)))
    
    if previous_label != 'O':
        tagged_sent += '$'
        
        
    tagged_sent = shift_neighbor_indices(tagged_sent,
                           shift_point_indices,
                           ne_list,
                           in_neighbors_list)
            
    in_neighbors_list = ['|'.join([str(i) for i in set(neighbors)]) for neighbors in in_neighbors_list]
        
    return tagged_sent.strip(),\
           ' '.join(in_neighbors_list)

def shift_neighbor_indices(tagged_sent,
                           shift_point_indices,
                           ne_list, 
                           neighbor_indices):
    
    new_tagged_sent = tagged_sent.split(' ')
    
    for _neighbor_indices in neighbor_indices:
        
        for i, _indice in enumerate(_neighbor_indices):
            
            _shift_num = 0
            for j, shift_point_indice in enumerate(shift_point_indices):
                if _indice > shift_point_indice:
                    _shift_num += len(ne_list[j].split(' '))
            _neighbor_indices[i] += _shift_num
            
    shift_point_indices.reverse()
    ne_list.reverse()
    
    for shift_point_indice, ne_text in zip(shift_point_indices, ne_list):
        _neighbor_indices = neighbor_indices[shift_point_indice]
        neighbor_indices.insert(shift_point_indice, [shift_point_indice])
        new_tagged_sent.insert(shift_point_indice + 1, ne_text)
        neighbor_indices[shift_point_indice] += _neighbor_indices
    
    return ' '.join(new_tagged_sent)

def dump_nary_instances_2_biobert_format(
        nary_instances, 
        out_nary_biobert_file, 
        only_single_sent,
        is_output_multiple_classes,
        is_drug_gene_var):
    
    with open(out_nary_biobert_file, 'w', encoding='utf8') as bert_writer:
        
        for index, nary_instance in enumerate(nary_instances):
            
            rel_label = nary_instance.relation
            
            if not is_output_multiple_classes:
                if rel_label == 'None':
                    rel_label = RelationType.NO
                else:
                    rel_label = RelationType.YES
            
            drug = ''
            gene = ''
            variant = ''
            
            for ne in nary_instance.ne_list:
                if ne.type == NEType.GENE:
                    gene = ne
                elif ne.type == NEType.DRUG:
                    drug = ne
                elif ne.type == NEType.VARIANT:
                    variant = ne
                else:
                    print('check me')
                
            tokens = nary_instance.token # [string, …, string]
            #deprels = nary_instance.deprel # either [string, …, string] or [[string,..,string], …, []]
            heads = nary_instance.head # either [int, …, int]  or [[int,..,int], …, []]
            ners = nary_instance.ner # [string, …, string]
            
            iob2_labels = []
            
            previous_label = ''
            for label in ners:
                if label != 'O':
                    if label == previous_label:
                        iob2_labels.append('I-' + label)
                    else:
                        iob2_labels.append('B-' + label)
                else:
                    iob2_labels.append(label)
                    
                label = previous_label
            
            tagged_sent, all_neighbors = convert_iob2_to_tagged_sent(
                tokens,
                iob2_labels,
                heads)
            
            orig_tokenized_text = ' '.join(tokens)
            
            if is_drug_gene_var:
                instance = drug.text + '\t' +\
                           gene.text + '\t' +\
                           variant.text + '\t' +\
                           tagged_sent + '\t' +\
                           all_neighbors + '\t' +\
                           rel_label + '\t' +\
                           orig_tokenized_text
            else:
                instance = drug.text + '\t' +\
                           variant.text + '\t' +\
                           tagged_sent + '\t' +\
                           all_neighbors + '\t' +\
                           rel_label + '\t' +\
                           orig_tokenized_text
                           
            bert_writer.write(instance + '\n')
   


def __merge_spacy_doc_tokens(std_tokens, doc):
    
    to_merge_tuples = []
    __sent_len = len(std_tokens)
    
    i = 0
    k = 0
    while i < __sent_len:
        std_tokens[i] = re.sub(r'[\s\t]+', ' ', std_tokens[i])
        if std_tokens[i] != doc[i].text:
            j = k + 2
            while j <= len(doc):
                #if std_tokens[i] == re.sub(r'[\s\t]+', ' ', doc[k:j].text):
                if std_tokens[i] == doc[k:j].text:
                    to_merge_tuples.append((k,j))
                    k = j - 1
                    break
                j += 1
        i += 1
        k += 1
        
    with doc.retokenize() as retokenizer:
        for to_merge_tuple in to_merge_tuples:
            #print(to_merge_tuple[0], 'Berfore merging', doc[to_merge_tuple[0]])
            retokenizer.merge(doc[to_merge_tuple[0]:to_merge_tuple[1]])
            #print(to_merge_tuple[0], 'After merging', doc[to_merge_tuple[0]])
         

def __convert_nary_json_2_object_with_remaining_all_neighbors(
        in_json_str, 
        only_single_sent, 
        is_output_multiple_classes, 
        nlp=None):

    nary_instances = []

    parsed = json.loads(in_json_str)
    
    for article in parsed:
        #try:
                
        nary_instance = NaryInstance()
        
        nary_instance.docid = article['article']
        nary_instance.id = article['article']
        
        
        for entity in article['entities']:
            ne_type = NEType.NONE
            if entity['type'] == 'gene':
                ne_type = NEType.GENE
            elif entity['type'] == 'drug':
                ne_type = NEType.DRUG
            elif entity['type'] == 'variant':
                ne_type = NEType.VARIANT
            else:
                print('check me')
            
            ne = None
            
            if len(entity['indices']) > 0:
                ne = AnnotationInstance(
                    entity['indices'][0],
                    entity['indices'][-1],
                    ne_type)
            else:
                # missing value in input data
                ne = AnnotationInstance(
                    0,
                    0,
                    ne_type)
            
            ne.text = entity['mention']
            
            if '\t' in ne.text:
                ne.text = ne.text.replace('\t', '_')
            
            nary_instance.ne_list.append(ne)
        
        if is_output_multiple_classes:
            nary_instance.relation = article['relationLabel'].strip().replace('  ', ' ')
        else:
            if article['relationLabel'].lower() != 'none':
                nary_instance.relation = RelationType.YES
            else:
                nary_instance.relation = RelationType.NO
            
        tokenized_text = ''
        if nlp == None:
            
            if only_single_sent and len(article['sentences']) > 1:
                continue
            
            for sent in article['sentences']:
                for node in sent['nodes']:
                    
                    token = node['label']
                    # it seems that scispacy's dep is more accurate than that of the original data
                
                    pos = node['postag']
                    all_deprels = []
                    all_heads = []
                    
                    for arc in node['arcs']:
                        deprel = arc['label'].split(':')[1]
                        head = arc['toIndex']
                        all_deprels.append(deprel)
                        all_heads.append(head)
                        
                    if '\t' in token:
                        token = token.replace('\t', '_')                    
                        
                    nary_instance.token.append(token)
                    tokenized_text += ' ' + token
                     # original data contains noise dep which either end with '(' or 'dep_XXX'
                    deprel = deprel.split('(')[0].split('_')[0]
                    nary_instance.pos.append(pos)
                    nary_instance.deprel.append(all_deprels)
                    nary_instance.head.append(all_heads)
                    nary_instance.ner.append('O')
            
            tokenized_text = tokenized_text.strip()
        else:
            # add scispacy postag and dep
            
            if only_single_sent and len(article['sentences']) > 1:
                continue
            
            for sent in article['sentences']:
                for node in sent['nodes']:
                    
                    token = node['label']
                    # it seems that scispacy's dep is more accurate than that of the original data
                    
                    nary_instance.token.append(token)
                    tokenized_text += ' ' + token
                    
            tokenized_text = tokenized_text.strip()
            doc = nlp(tokenized_text)
            
            __merge_spacy_doc_tokens(nary_instance.token, doc)
            
            _max_head = 0
                    
            for i, token in enumerate(doc):
                nary_instance.pos.append(token.pos_)
                nary_instance.deprel.append(token.dep_)
                if token.head.i != i:
                    nary_instance.head.append(token.head.i + 1)
                else:
                    # is root
                    nary_instance.head.append(0)
                    
                if nary_instance.head[i] > _max_head:
                    _max_head = nary_instance.head[i]
                    
                nary_instance.ner.append('O')
                #print(i, token, token.pos_, token.dep_, token.head.text, token.head.i)
            #print(nary_instance.head)
            
        
        for ne in nary_instance.ne_list:
            
            nary_instance.ner[ne.end] = ne.type
            nary_instance.ner[ne.start] = ne.type
            
            for i in range(ne.start+1,ne.end):
                nary_instance.ner[i] = ne.type
        
        
        if len(nary_instance.head) != len(nary_instance.token):
            print('len(nary_instance.head)', len(nary_instance.head))
            print('len(nary_instance.token)', len(nary_instance.token))
            '''for i, token in enumerate(doc):
                print(i, token, token.pos_, token.dep_, token.head.text, token.head.i, nary_instance.head[i])'''
            for i, (_token, token, ne) in enumerate(zip(nary_instance.token, doc, nary_instance.ner)):
                print(i, _token, token, ne)
            print(tokenized_text)
            print(token)
            print('Failed in ', nary_instance.docid)
            input('wait') 
           
        nary_instances.append(nary_instance)
        
    return nary_instances

def __convert_nary_json_2_tsv(
    in_nary_json_files,
    out_nary_biobert_file,
    only_single_sent,
    is_output_multiple_classes,
    is_drug_gene_var):
    
    all_nary_instances = []
    
    for in_nary_json_file in in_nary_json_files:
        with open(in_nary_json_file, 'r', encoding='utf8') as reader:
            for in_json_str in reader.readlines():
                json_instances = __convert_nary_json_2_object_with_remaining_all_neighbors(
                        in_json_str, 
                        only_single_sent, 
                        is_output_multiple_classes, 
                        None)
                all_nary_instances.extend(json_instances)
            
    dump_nary_instances_2_biobert_format(
        all_nary_instances, 
        out_nary_biobert_file, 
        only_single_sent,
        is_output_multiple_classes,
        is_drug_gene_var)
    
def generate_train_dev_test_for_bert_song_split(
        in_nary_dir,
        out_nary_dir,
        only_single_sent,
        is_output_multiple_classes,
        is_drug_gene_var):
        
    # generate original five folds' json
    for i in range(0,5):
        in_nary_2nd_files = [in_nary_dir + str(i) + '/data_graph_1', in_nary_dir + str(i) + '/data_graph_2']
        out_nary_biobert_file = out_nary_dir + str(i) + '.tsv'
        print('Converting', in_nary_2nd_files, 'to', out_nary_biobert_file)
        __convert_nary_json_2_tsv(
                in_nary_2nd_files, 
                out_nary_biobert_file, 
                only_single_sent,
                is_output_multiple_classes,
                is_drug_gene_var)
        print('Finished!')
    
    all_folds = np.arange(5)
    for i in range(0,5):
                
        out_dir = out_nary_dir + 'cv' + str(i)
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        out_train_tsv_file = out_dir + '/train.tsv'
        out_dev_tsv_file = out_dir + '/dev.tsv'
        out_test_tsv_file = out_dir + '/test.tsv'
        
        all_train_lines = []
        test_lines = []
        all_tsv_files = [out_nary_dir + str(i) + '.tsv' for i in range(0,5)]
        
        for j, tsv_file in enumerate(all_tsv_files):
            if j == i:
                for line in open(tsv_file, 'r', encoding='utf8'):
                    test_lines.append(line)
                continue
            for line in open(tsv_file, 'r', encoding='utf8'):
                all_train_lines.append(line)
                
        random.seed(1234)
        random.shuffle(all_train_lines)
        devset = all_train_lines[:200]
        trainset = all_train_lines[200:]
        
        with open(out_train_tsv_file, 'w', encoding='utf8') as tsv_writer:
            for i, line in enumerate(trainset):
                tsv_writer.write(str(i) + '\t' + line)
                
        with open(out_dev_tsv_file, 'w', encoding='utf8') as tsv_writer:
            for i, line in enumerate(devset):
                tsv_writer.write(str(i) + '\t' + line)
        
        with open(out_test_tsv_file, 'w', encoding='utf8') as test_tsv_writer:
            
            if is_drug_gene_var:
                test_tsv_writer.write('index\tdrug\tgene\tvariant\tsentence\tneighbors\tlabel\torig_text\n')
            else:
                test_tsv_writer.write('index\tdrug\tvariant\tsentence\tneighbors\tlabel\torig_text\n')
            for i, line in enumerate(test_lines):
                test_tsv_writer.write(str(i) + '\t' + line)

        all_folds = np.roll(all_folds,-1)
    

def use_songs_splits(
        in_nary_dir,
        out_all_dir,
        out_single_sent_dir):

    # refers to https://github.com/freesunshine0316/nary-grn/blob/master/gs_lstm/G2S_trainer.py
    # line 102~104:
    # random.shuffle(trainset)
    # devset = trainset[:200]
    # trainset = trainset[200:]
    
    if not os.path.exists(out_single_sent_dir):
        os.makedirs(out_single_sent_dir)
    if not os.path.exists(out_all_dir):
        os.makedirs(out_all_dir)
    
    data_folders = ['drug_gene_var/', 'drug_var/']
    are_drug_gene_var = [True, False]

    for data_folder, is_drug_gene_var in zip(data_folders, are_drug_gene_var):
        
        
        if not os.path.exists(out_single_sent_dir + data_folder):
            os.makedirs(out_single_sent_dir + data_folder)
        generate_train_dev_test_for_bert_song_split(
            in_nary_dir = in_nary_dir + data_folder,
            out_nary_dir = out_single_sent_dir + data_folder,
            only_single_sent = True,
            is_output_multiple_classes = True,
            is_drug_gene_var = is_drug_gene_var)
        
        if not os.path.exists(out_all_dir + data_folder):
            os.makedirs(out_all_dir + data_folder)
        generate_train_dev_test_for_bert_song_split(
            in_nary_dir = in_nary_dir + data_folder,
            out_nary_dir = out_all_dir + data_folder,
            only_single_sent = False,
            is_output_multiple_classes = True,
            is_drug_gene_var = is_drug_gene_var)

if __name__ == '__main__':
    
    in_nary_dir = 'datasets/nary/origin/'
    out_all_dir = 'datasets/nary/processed/all/'
    out_single_sent_dir = 'datasets/nary/processed/only_single_sent/'
    
    use_songs_splits(
        in_nary_dir = in_nary_dir,
        out_all_dir = out_all_dir,
        out_single_sent_dir = out_single_sent_dir)
    