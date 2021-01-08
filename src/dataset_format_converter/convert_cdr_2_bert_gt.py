# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:58:51 2019

@author: laip2
"""

from document import PubtatorDocument, TextInstance
from annotation import AnnotationInfo
import os
import random

import re
import spacy
import scispacy

def _spacy_split_sentence(text, nlp):
    offset = 0
    offsets = []
    doc = nlp(text)
    
    for sent in doc.sents:
        start = offset
        end = offset + len(sent.text)
        offsets.append((start, end))
        
        offset = end
        for c in text[end:]:
            if c == ' ':
                offset += 1
            else:
                break
    return offsets

def split_sentence(document, nlp):
    new_text_instances = []
    for text_instance in document.text_instances:
        offsets = [o for o in _spacy_split_sentence(text_instance.text, nlp)]
        #offsets = [o for o in _nltk_split_sentence(text_instance.text)]
        _tmp_text_instances = []
        for start, end in offsets:
            new_text_instance = TextInstance(text_instance.text[start:end])
            new_text_instance.offset = start
            _tmp_text_instances.append(new_text_instance)
        for annotation in text_instance.annotations:
            is_entity_splited = True
            for _tmp_text_instance in _tmp_text_instances:
                if _tmp_text_instance.offset <= annotation.position and \
                    (annotation.position + annotation.length) - _tmp_text_instance.offset <= len(_tmp_text_instance.text):
                    annotation.position = annotation.position - _tmp_text_instance.offset
                    _tmp_text_instance.annotations.append(annotation)
                    is_entity_splited = False
                    break
            if is_entity_splited:
                print(annotation.position, annotation.length, annotation.text)
                print (' splited by Spacy\' sentence spliter is failed to be loaded into TextInstance\n')
                for _tmp_text_instance in _tmp_text_instances:
                    print (_tmp_text_instance.offset, len(_tmp_text_instance.text), _tmp_text_instance.text)
        new_text_instances.extend(_tmp_text_instances)
    
    document.text_instances = new_text_instances

def tokenize_document_by_spacy(document, nlp):
    for text_instance in document.text_instances: 
  
        doc = nlp(re.sub(r'\s+', ' ', text_instance.text))
                
        tokens = []
        for i, token in enumerate(doc):
            tokens.append(token.text)
            text_instance.pos_tags.append(token.pos_)
            text_instance.head.append(token.dep_)
            text_instance.head_indexes.append(token.head.i)
            text_instance.stems.append(token.lemma_)
            
        text_instance.tokenized_text = ' '.join(tokens)

def tokenize_documents_by_spacy(documents, spacy_model):
    
    nlp = spacy.load(spacy_model)
    
    for document in documents:
        split_sentence(document, nlp)
        tokenize_document_by_spacy(document, nlp)
        

def add_annotations_2_text_instances(text_instances, annotations):
    offset = 0
    for text_instance in text_instances:
        text_instance.offset = offset
        offset += len(text_instance.text) + 1
        
    for annotation in annotations:
        can_be_mapped_to_text_instance = False
                
        for i, text_instance in enumerate(text_instances):
            if text_instance.offset <= annotation.position and annotation.position + annotation.length <= text_instance.offset + len(text_instance.text):
                
                annotation.position = annotation.position - text_instance.offset
                text_instance.annotations.append(annotation)
                can_be_mapped_to_text_instance = True
                break
        if not can_be_mapped_to_text_instance:
            print(annotation.text)
            print(annotation.position)
            print(annotation.length)
            print(annotation, 'cannot be mapped to original text')
            raise

def load_pubtator_into_documents(in_pubtator_file):
    
    documents = []
    
    with open(in_pubtator_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        
        document = None
        
        annotations = []
        text_instances = []
        relation_pairs = []
        
        for line in pub_reader:
            line = line.rstrip()
            
            if line == '':
                
                document = PubtatorDocument(pmid)
                #print(pmid)
                add_annotations_2_text_instances(text_instances, annotations)
                document.text_instances = text_instances
                document.relation_pairs = relation_pairs
                documents.append(document)
                
                annotations = []
                text_instances = []
                relation_pairs = []
                continue
            
            tks = line.split('|')
            
            
            if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                pmid = tks[0]
                x = TextInstance(tks[2])
                text_instances.append(x)
            else:
                _tks = line.split('\t')
                if _tks[1] != 'CID':
                    start = int(_tks[1])
                    end = int(_tks[2])
                    text = _tks[3]
                    ne_type = _tks[4]
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    ids = _tks[5]
                    _anno = AnnotationInfo(start, end-start, text, ne_type)
                    _anno.ids = set(ids.split('|'))
                    annotations.append(_anno)
                else:
                    id1 = _tks[2]
                    id2 = _tks[3]
                    relation_pairs.append((id1, id2))
                    
        if len(text_instances) != 0:
            document = PubtatorDocument(pmid)
            add_annotations_2_text_instances(text_instances, annotations)
            document.text_instances = text_instances
            document.relation_pairs = relation_pairs
            documents.append(document)
                
    return documents
    
def convert_text_instance_2_iob2(text_instance, id1, id2, do_mask_other_nes):
    tokens = []
    labels = []
    
    for token in text_instance.tokenized_text.split(' '):
        tokens.append(token)
        labels.append('O')
        
    annotation_indexes_wo_count_space = []
    for annotation in text_instance.annotations:
        start = len(text_instance.text[:annotation.position].replace(' ', ''))
        end = start + len(annotation.text.replace(' ', ''))
        annotation_indexes_wo_count_space.append((start, end))
    
    for (start, end), annotation in zip(annotation_indexes_wo_count_space, text_instance.annotations):
        offset = 0
        for i, token in enumerate(tokens):
            if offset == start:
                if id1 in annotation.ids:
                    labels[i] = "B-" + annotation.ne_type + 'Src'
                elif id2 in annotation.ids:
                    labels[i] = "B-" + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = "B-" + annotation.ne_type
            elif start < offset and offset < end:
                if id1 in annotation.ids:
                    labels[i] = "I-" + annotation.ne_type + 'Src'
                elif id2 in annotation.ids:
                    labels[i] = "I-" + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = "I-" + annotation.ne_type
            elif offset < start and start < offset + len(token): #ex: renin-@angiotensin$
                if id1 in annotation.ids:
                    labels[i] = "B-" + annotation.ne_type + 'Src'
                elif id2 in annotation.ids:
                    labels[i] = "B-" + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = "B-" + annotation.ne_type                
                    
            offset += len(token)
        
    return tokens, labels

def convert_iob2_to_tagged_sent(
        tokens, 
        labels, 
        in_neighbors, 
        out_neighbors,
        token_offset):
    
    tagged_sent = ''
    
    in_neighbors_list = []
    out_neighbors_list = []
    
    num_orig_tokens = len(tokens)
    
    previous_label = 'O'
    
    orig_token_index_2_new_token_index_mapping = []
    new_token_index_2_orig_token_index_mapping = {}
    
    new_idx = -1
    for i, (token, label) in enumerate(zip(tokens, labels)):
            
        if label == 'O':
            if previous_label != 'O':
                tagged_sent += '$ ' + token
            else:
                tagged_sent += ' ' + token
            new_idx += 1
        elif label.startswith('B-'):
            if previous_label != 'O':
                tagged_sent += '$ @' + label[2:]
            else:
                tagged_sent += ' @' + label[2:]
            new_idx += 1

        previous_label = label
        orig_token_index_2_new_token_index_mapping.append(new_idx)
        if new_idx not in new_token_index_2_orig_token_index_mapping:
            new_token_index_2_orig_token_index_mapping[new_idx] = []
        new_token_index_2_orig_token_index_mapping[new_idx].append(i)
    
    
    for new_idx in range(len(new_token_index_2_orig_token_index_mapping)):
        new_neighbors = []
        for old_index in new_token_index_2_orig_token_index_mapping[new_idx]:
            for old_neighbor in in_neighbors[old_index]:
                new_neighbor = orig_token_index_2_new_token_index_mapping[old_neighbor]
                if new_neighbor not in new_neighbors:
                    new_neighbors.append(new_neighbor)       
        in_neighbors_list.append(new_neighbors)
        
    
    
    for new_idx in range(len(new_token_index_2_orig_token_index_mapping)):
        new_neighbors = []
        for old_index in new_token_index_2_orig_token_index_mapping[new_idx]:
            for old_neighbor in out_neighbors[old_index]:
                new_neighbor = orig_token_index_2_new_token_index_mapping[old_neighbor]
                if new_neighbor not in new_neighbors:
                    new_neighbors.append(new_neighbor)        
        out_neighbors_list.append(new_neighbors)
        
        
    if previous_label != 'O':
        tagged_sent += '$'
    
    new_in_neighbors_list = ['|'.join([str(i + token_offset) for i in set(neighbors)]) for neighbors in in_neighbors_list]
    new_out_neighbors_list = ['|'.join([str(i + token_offset) for i in set(neighbors)]) for neighbors in out_neighbors_list]
    
    return tagged_sent.strip(),\
           ' '.join(new_in_neighbors_list),\
           ' '.join(new_out_neighbors_list),\
           token_offset + len(new_in_neighbors_list)
    
def enumerate_all_cdr_pairs(document):
    
    all_cdr_pairs = set()
    
    all_chemical_ids = set()
    all_disease_ids = set()
    
    for text_instance in document.text_instances:
        for annotation in text_instance.annotations:
            if annotation.ne_type == 'Chemical':
                for id in annotation.ids:
                    all_chemical_ids.add(id)
            elif annotation.ne_type == 'Disease':
                for id in annotation.ids:
                    all_disease_ids.add(id)
    
    for id1 in all_chemical_ids:
        for id2 in all_disease_ids:
            all_cdr_pairs.add((id1, id2))
    
    return all_cdr_pairs
    
def get_in_neighbors_list(text_instance):
    in_neighbors_list = []
    in_neighbors_head_list = []
    for current_idx, (head, head_idx) in enumerate(zip(
                                         text_instance.head,
                                         text_instance.head_indexes)):
        neighbors = []
        neighbors_head = []
        
        neighbors.append(head_idx)
        neighbors_head.append(head)
        
        in_neighbors_list.append(neighbors)
        in_neighbors_head_list.append(neighbors_head)
        
    return in_neighbors_list, in_neighbors_head_list

def get_out_neighbors_list(text_instance):
    
    out_neighbors_list = []
    out_neighbors_head_list = []
    
    invert_heads = {}
        
    
    for current_idx, (head, head_idx) in enumerate(zip(text_instance.head,
                                                       text_instance.head_indexes)):
        if head_idx not in invert_heads:
            invert_heads[head_idx] = []
        
        _edge = {}
        _edge['label'] = head
        _edge['toIndex'] = current_idx
        
        invert_heads[head_idx].append(_edge)
    
    for current_idx, (head, head_idx) in enumerate(zip(
                                         text_instance.head,
                                         text_instance.head_indexes)):
        neighbors = []
        neighbors_head = []
                    
        if current_idx in invert_heads:
            for edge in invert_heads[current_idx]:
                neighbors.append(edge['toIndex'])
                neighbors_head.append(edge['label'])
        
        out_neighbors_list.append(neighbors)
        out_neighbors_head_list.append(neighbors_head)
        
    return out_neighbors_list, out_neighbors_head_list

def dump_documents_2_bert_gt_format(
    all_documents, 
    out_bert_file, 
    is_test_set, 
    do_mask_other_nes):
    
    num_seq_lens = []
    
    with open(out_bert_file, 'w', encoding='utf8') as bert_writer:
        
        if is_test_set:
            bert_writer.write('index\tid1\tid2\tis_in_same_sent\tmin_sents_window\tsentence\tin_neighbors\tlabel\n')
        
        number_unique_YES_instances = 0
        
        for document in all_documents:
            
            all_pairs = enumerate_all_cdr_pairs(document)
            
            unique_YES_instances = set()
            
            for cdr_relation_pair in all_pairs:
        
                relation_label = '0'
                
                if cdr_relation_pair in document.relation_pairs:
                    relation_label = '1'
                
                id1 = cdr_relation_pair[0]
                id2 = cdr_relation_pair[1]
                
                tagged_sents = []
                all_sents_in_neighbors = []
                all_sents_out_neighbors = []
                
                is_in_same_sent = False
                
                src_sent_ids = []
                tgt_sent_ids = []
                
                token_offset = 0
                
                for sent_id, text_instance in enumerate(document.text_instances):
                    
                    tokens, labels = convert_text_instance_2_iob2(text_instance, id1, id2, do_mask_other_nes)
                    
                    in_neighbors_list, _ = get_in_neighbors_list(text_instance)
                    out_neighbors_list, _ = get_out_neighbors_list(text_instance)
                    
                    if not is_in_same_sent:
                        is_Src_in = False
                        is_Tgt_in = False
                        for _label in labels:
                            if 'Src' in _label:
                                is_Src_in = True
                                src_sent_ids.append(sent_id)
                                break
                        for _label in labels:
                            if 'Tgt' in _label:
                                is_Tgt_in = True
                                tgt_sent_ids.append(sent_id)
                                break
                        if is_Src_in and is_Tgt_in:
                            is_in_same_sent = True
                    
                    tagged_sent, in_neighbors_str, out_neighbors_str, token_offset =\
                        convert_iob2_to_tagged_sent(
                            tokens,
                            labels,
                            in_neighbors_list,
                            out_neighbors_list,
                            token_offset)
                
                    tagged_sents.append(tagged_sent)
                    all_sents_in_neighbors.append(in_neighbors_str)
                    all_sents_out_neighbors.append(out_neighbors_str)
                
                
                min_sents_window = 100
                for src_sent_id in src_sent_ids:
                    for tgt_sent_id in tgt_sent_ids:
                        _min_sents_window = abs(src_sent_id - tgt_sent_id)
                        if _min_sents_window < min_sents_window:
                            min_sents_window = _min_sents_window
                            
                num_seq_lens.append(float(len(tagged_sent.split(' '))))
                
                instance = document.id + '\t' +\
                           id1 + '\t' +\
                           id2 + '\t' +\
                           str(is_in_same_sent) + '\t' +\
                           str(min_sents_window) + '\t' +\
                           ' '.join(tagged_sents) + '\t' +\
                           ' '.join(all_sents_in_neighbors)
                           #' '.join(all_sents_in_neighbors) + '\t' +\
                           #' '.join(all_sents_out_neighbors)
                           
                if relation_label == '1':
                    unique_YES_instances.add(instance)
                
                if is_test_set or (id1 != '-1' and id2 != '-1'):
                    bert_writer.write(instance + '\t' + 
                                      relation_label + '\n')
                
            number_unique_YES_instances += len(unique_YES_instances)
                    
            bert_writer.flush()
            
        print('number_unique_YES_instances', number_unique_YES_instances)
        
    return sum(num_seq_lens) / len(num_seq_lens)
    
def generate_train_dev_test_data(
        in_pubtator_dir,
        out_bert_dir,
        spacy_model,
        do_mask_other_nes = False):
    
    if not os.path.exists(out_bert_dir):
        os.makedirs(out_bert_dir)
    
    in_train_file = in_pubtator_dir + 'CDR_TrainingSet.PubTator.txt'
    in_dev_file = in_pubtator_dir + 'CDR_DevelopmentSet.PubTator.txt'
    in_test_file = in_pubtator_dir + 'CDR_TestSet.PubTator.txt'
    
    out_train_file = out_bert_dir + 'train.tsv'
    out_dev_file = out_bert_dir + 'dev.tsv'
    out_test_file = out_bert_dir + 'test.tsv'
    
    
    train_documents = load_pubtator_into_documents(in_train_file)
    train_documents.extend(load_pubtator_into_documents(in_dev_file))
    test_documents = load_pubtator_into_documents(in_test_file)
    
    tokenize_documents_by_spacy(train_documents, spacy_model)
    tokenize_documents_by_spacy(test_documents, spacy_model)
    
    random.seed(1234)
    random.shuffle(train_documents)
    dev_documents = train_documents[:200]
    train_documents = train_documents[200:]
    
    dump_documents_2_bert_gt_format(
        train_documents, 
        out_train_file, 
        is_test_set=False, 
        do_mask_other_nes=do_mask_other_nes)
    
    dump_documents_2_bert_gt_format(
        dev_documents, 
        out_dev_file, 
        is_test_set=False, 
        do_mask_other_nes=do_mask_other_nes)
    
    dump_documents_2_bert_gt_format(
        test_documents, 
        out_test_file, 
        is_test_set=True, 
        do_mask_other_nes=do_mask_other_nes)


if __name__ == '__main__':
    
    in_pubtator_dir = 'datasets/cdr/CDR_Data/CDR.Corpus.v010516/'
    out_bert_dir    = 'datasets/cdr/processed/'
    spacy_model     = 'en_core_sci_md'
    
    generate_train_dev_test_data(
        in_pubtator_dir   = in_pubtator_dir,
        out_bert_dir      = out_bert_dir,
        spacy_model       = spacy_model)
    