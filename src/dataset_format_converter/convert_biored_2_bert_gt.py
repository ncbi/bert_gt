# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:27:17 2021

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
    
    do_not_split = False
    start = 0
    end = 0
    for sent in doc.sents:
        if re.search(r'\b[a-z]\.$|[A-Z] ?\>$|[^a-z]del\.$| viz\.$', sent.text):
            if not do_not_split:
                start = offset
            end = offset + len(sent.text)
            offset = end
            for c in text[end:]:
                if c == ' ':
                    offset += 1
                else:
                    break
            do_not_split = True
        else:
            if do_not_split:                
                do_not_split = False
                end = offset + len(sent.text)
                offset = end
                for c in text[end:]:
                    if c == ' ':
                        offset += 1
                    else:
                        break
                offsets.append((start, end))
            else:
                start = offset
                end = offset + len(sent.text)
                offsets.append((start, end))
                
                offset = end
                for c in text[end:]:
                    if c == ' ':
                        offset += 1
                    else:
                        break
        
    if do_not_split:
        offsets.append((start, end))
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
        
def shift_neighbor_indices_and_add_end_tag(tagged_sent,
                                           ne_positions,
                                           ne_list, 
                                           neighbor_indices,
                                           has_end_tag):
                    
    new_tagged_sent = tagged_sent.split(' ')
    
    # if parsing sentence fail => len(neighbor_indices) == 0
    if len(neighbor_indices) > 0:
        # update indices by using ne_positions if indices > ne_positions then shift NE's length
        for _neighbor_indices in neighbor_indices:
            
            for i, _indice in enumerate(_neighbor_indices):
                
                if not has_end_tag:
                    _shift_num = 0
                else:
                    # we consider "end tag" as part of ne text
                    _shift_num = 1
                for j, shift_point_indice in enumerate(ne_positions):
                    if _indice > shift_point_indice:
                        _shift_num += len(ne_list[j].split(' '))
                _neighbor_indices[i] += _shift_num
            
    ne_positions.reverse()
    ne_list.reverse()
        
    for ne_position, ne_text in zip(ne_positions, ne_list):
        
        if len(neighbor_indices) > 0:
            ne_tag_neighbor_indices = neighbor_indices[ne_position]
        
        # add ne into neighbor and tagged sent
        for i, _ne_token in enumerate(ne_text.split(' ')):
            
            if len(neighbor_indices) > 0:
                # ne text point to ne tag
                neighbor_indices.insert(ne_position + i, [ne_position])
                neighbor_indices[ne_position + i] += ne_tag_neighbor_indices
            
            # insert ne text
            new_tagged_sent.insert(ne_position + 1 + i, _ne_token)
        
        
        if has_end_tag:
            # ne text point to ne tag
            end_tag_index = ne_position + len(ne_text.split(' ')) + 1
            if len(neighbor_indices) > 0:
                neighbor_indices.insert(end_tag_index, [ne_position])
                neighbor_indices[end_tag_index] += ne_tag_neighbor_indices
            new_tagged_sent.insert(end_tag_index, new_tagged_sent[ne_position].replace('@', '@/'))
                        
    return ' '.join(new_tagged_sent)
        
def convert_iob2_to_tagged_sent(
        tokens, 
        labels, 
        in_neighbors_list,
        token_offset,
        to_mask_src_and_tgt = False,
        has_end_tag = False):
    
        
    num_orig_tokens = len(tokens)
    
    previous_label = 'O'
    
    orig_token_index_2_new_token_index_mapping = []
    
    current_idx = -1
    
    tagged_sent = ''
    ne_type = ''
    ne_text = ''
    ne_list = []
    # convert IOB2 to bert format
    # NEs are replaced by tags
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == 'O':
            if previous_label != 'O':
                tagged_sent += '$ ' + token
                #print('1 ne_list.append(ne_text)', ne_text)
                ne_list.append(ne_text)
                ne_text = ''
            else:
                tagged_sent += ' ' + token    
            current_idx += 1
                
        elif label.startswith('B-'):
            if previous_label != 'O':
                tagged_sent += '$ @' + label[2:]
                #print('2 ne_list.append(ne_text)', ne_text)
                ne_list.append(ne_text)
                ne_text = token
                ne_type = label[2:]
            else:
                tagged_sent += ' @' + label[2:]
                ne_text = token
                ne_type = label[2:]
            current_idx += 1
                
        elif label.startswith('I-'):
            ne_text += ' ' + token
        #print('=================>')
        #print(i, token, label)
        #print(tagged_sent)
        previous_label = label
        orig_token_index_2_new_token_index_mapping.append(current_idx)
    if ne_text != '':
        ne_list.append(ne_text)
        ne_text = ''
    tagged_sent = tagged_sent.strip()
    if previous_label != 'O':
        tagged_sent += '$'
            
    #    
    
    # update neighbor index
    previous_idx = 0
    _new_neighbors = [] # 
    
        
    _tokens = tagged_sent.split(' ')
    
    ne_positions = []
    for i in range(len(_tokens)):
        token = _tokens[i]
        if token.startswith('@') and token.endswith('$'):
            ne_positions.append(i)
            
    new_in_neighbors_list = []
    if len(in_neighbors_list) != 0:
        # update in_neighbors_list to new_in_neighbors_list by orig_token_index_2_new_token_index_mapping
        for i in range(num_orig_tokens):
            if previous_idx == orig_token_index_2_new_token_index_mapping[i]:
                for neighbor_idx in in_neighbors_list[i]:
                    _new_neighbors.append(orig_token_index_2_new_token_index_mapping[neighbor_idx])
            else:
                new_in_neighbors_list.append(list(set(_new_neighbors)))
                _new_neighbors = []
                for neighbor_idx in in_neighbors_list[i]:
                    _new_neighbors.append(orig_token_index_2_new_token_index_mapping[neighbor_idx])
            previous_idx = orig_token_index_2_new_token_index_mapping[i]
        new_in_neighbors_list.append(list(set(_new_neighbors)))
    #
    
    # insert ne text and update neighbor index again
    if to_mask_src_and_tgt == False:
        tagged_sent = shift_neighbor_indices_and_add_end_tag(
                               tagged_sent,
                               ne_positions,
                               ne_list,
                               new_in_neighbors_list,
                               has_end_tag)
    #

    # add token_offset to neighbor index
    new_in_neighbors_list = ['|'.join([str(i + token_offset) for i in set(neighbors)]) for neighbors in new_in_neighbors_list]
    
    
    return tagged_sent.strip(),\
           ' '.join(new_in_neighbors_list),\
           token_offset + len(new_in_neighbors_list)
    
def enumerate_all_id_pairs_by_specified(document,
                                        src_tgt_pairs,
                                        only_pair_in_same_sent):
    all_pairs = set()
    
    if only_pair_in_same_sent:
        for text_instance in document.text_instances:
                    
            all_id_infos_list = list()
            _all_id_infos_set = set()
            
            text_instance.annotations = sorted(text_instance.annotations, key=lambda x: x.position, reverse=False)
            for annotation in text_instance.annotations:
                for id in annotation.ids:
                    if (id, annotation.ne_type) not in _all_id_infos_set:
                        all_id_infos_list.append((id, annotation.ne_type))
                        _all_id_infos_set.add((id, annotation.ne_type))
            
            #print('====>len(all_id_infos_list)', len(all_id_infos_list))
            for i in range(0, len(all_id_infos_list) - 1):
                id1_info = all_id_infos_list[i]
                for j in range(i + 1, len(all_id_infos_list)):
                    id2_info = all_id_infos_list[j]
                    #print(id1_info[0], id2_info[1], id1_info[1], id2_info[1])
                    for src_ne_type, tgt_ne_type in src_tgt_pairs:
                        if id1_info[1] == src_ne_type and id2_info[1] == tgt_ne_type:
                            all_pairs.add((id1_info[0], id2_info[0], id1_info[1], id2_info[1]))
                            break
                            #print('OK')
                        elif id2_info[1] == src_ne_type and id1_info[1] == tgt_ne_type:
                            all_pairs.add((id2_info[0], id1_info[0], id2_info[1], id1_info[1]))
                            break
                        #print('OK')
    else:    
        all_id_infos_list = list()
        _all_id_infos_set = set()
        
        for text_instance in document.text_instances:
            text_instance.annotations = sorted(text_instance.annotations, key=lambda x: x.position, reverse=False)
            for annotation in text_instance.annotations:
                for id in annotation.ids:
                    if (id, annotation.ne_type) not in _all_id_infos_set:
                        all_id_infos_list.append((id, annotation.ne_type))
                        _all_id_infos_set.add((id, annotation.ne_type))
        
        #print('====>len(all_id_infos_list)', len(all_id_infos_list))
        for i in range(0, len(all_id_infos_list) - 1):
            id1_info = all_id_infos_list[i]
            for j in range(i + 1, len(all_id_infos_list)):
                id2_info = all_id_infos_list[j]
                #print(id1_info[0], id2_info[1], id1_info[1], id2_info[1])
                for src_ne_type, tgt_ne_type in src_tgt_pairs:
                    if id1_info[1] == src_ne_type and id2_info[1] == tgt_ne_type:
                        all_pairs.add((id1_info[0], id2_info[0], id1_info[1], id2_info[1]))
                        break
                        #print('OK')
                    elif id2_info[1] == src_ne_type and id1_info[1] == tgt_ne_type:
                        all_pairs.add((id2_info[0], id1_info[0], id2_info[1], id1_info[1]))
                        break
                        #print('OK')
    return all_pairs

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

def get_ne_id_2_ne_text_dict(document):
    ne_id_2_ne_text_dict = {}
    for text_instance in document.text_instances:
        for ann in text_instance.annotations:
            for id in ann.ids:
                ne_id_2_ne_text_dict[id] = ann.text
    return ne_id_2_ne_text_dict

def dump_documents_2_bert_gt_format(
    all_documents, 
    out_bert_file, 
    src_ne_type = '',
    tgt_ne_type = '',
    src_tgt_pairs = set(),
    is_test_set = False, 
    do_mask_other_nes = False,
    only_pair_in_same_sent = False,
    to_mask_src_and_tgt = False,
    to_insert_src_and_tgt_at_left = False,
    has_novelty = False,
    has_end_tag = False,
    task_tag = None,
    neg_label = 'None',
    pos_label = '',
    has_dgv = False,
    use_corresponding_gene_id = False,
    has_ne_type = False,
    only_co_occurrence_sent = False):
    
    num_seq_lens = []
    
    _index = 0
    with open(out_bert_file, 'w', encoding='utf8') as bert_writer:
        
        if is_test_set:
            if not has_novelty:
                bert_writer.write('pmid\tid1\tid2\tis_in_same_sent\tmin_sents_window\tsentence\tin_neighbors\tlabel\n')
            else:
                bert_writer.write('pmid\tid1\tid2\tis_in_same_sent\tmin_sents_window\tsentence\tin_neighbors\tlabel\tnovelty\n')
        
        number_unique_YES_instances = 0
        
        #print('================>')
        #print(len(all_documents))
        
        for document in all_documents:
            pmid = document.id
            all_pairs = enumerate_all_id_pairs_by_specified(document,
                                                   src_tgt_pairs,
                                                   only_pair_in_same_sent)              
            
            unique_YES_instances = set()
            
            disease_ids = set()
            variant_ids = set()
            for text_instance in document.text_instances:
                for annotation in text_instance.annotations:
                    if annotation.ne_type == 'DiseaseOrPhenotypicFeature':
                        for id in annotation.ids:
                            disease_ids.add(id)
                    elif annotation.orig_ne_type == 'SequenceVariant':
                        for id in annotation.ids:
                            variant_ids.add(id)

                        
            #print('===============>document.relation_pairs', document.relation_pairs)
            #print('===============>all_pairs', all_pairs)
            
            # for pairs have two entities
            for relation_pair in all_pairs:
        
                if not has_novelty:
                    relation_label = neg_label
                else:
                    relation_label = neg_label + '|None' # rel_type|novelty novelty => ['None', 'No', 'Novel']
                
                #print('=================>relation_pair', relation_pair)
                if not document.relation_pairs:
                    #print('=================>no relation_pair', document.id)
                    document.relation_pairs = {}
                
                if (relation_pair[0], relation_pair[1]) in document.relation_pairs:
                    relation_label = document.relation_pairs[(relation_pair[0], relation_pair[1])]
                    if pos_label != '' and (not has_novelty):
                        relation_label = pos_label
                elif (relation_pair[1], relation_pair[0]) in document.relation_pairs:
                    relation_label = document.relation_pairs[(relation_pair[1], relation_pair[0])]
                    if pos_label != '' and (not has_novelty):
                        relation_label = pos_label
                id1 = relation_pair[0]
                id2 = relation_pair[1]    
                id1type = relation_pair[2]
                id2type = relation_pair[3]
                
                tagged_sents = []
                all_sents_in_neighbors = []
                #all_sents_out_neighbors = []
                
                is_in_same_sent = False
                
                src_sent_ids = []
                tgt_sent_ids = []
                                
                token_offset = 0
                
                                
                #print('ggggggggggggggggg')
                for sent_id, text_instance in enumerate(document.text_instances):
                    
                    tokens, labels = convert_text_instance_2_iob2(text_instance, id1, id2, do_mask_other_nes)
                    #print(' '.join(tokens))
                    
                    in_neighbors_list, _ = get_in_neighbors_list(text_instance)
                    #out_neighbors_list, _ = get_out_neighbors_list(text_instance)
                    
                    # raise if neighbor is wrong
                    if len(tokens) != len(in_neighbors_list):
                        print('==================>')
                        print('len(tokens)', len(tokens))
                        print('len(in_neighbors_list)', len(in_neighbors_list))
                        print('tokens', tokens)
                        print(document.id, sent_id)
                        '''
                        for _sent_id, _text_instance in enumerate(document.text_instances):
                            
                            print(_sent_id, _text_instance.tokenized_text)
                            
                        for current_idx, (head, head_idx) in enumerate(zip(
                                         text_instance.head,
                                         text_instance.head_indexes)):
                            print(tokens[current_idx], head_idx, head, in_neighbors_list[current_idx])
                        print('==================>')
                        for i in range(len(tokens)):
                            print(tokens[i], in_neighbors_list[i])'''
                        in_neighbors_list = []
                    #
                    
                    # check if Source and Target are in the same sentence
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
                    #
                        
                    
                    tagged_sent, in_neighbors_str, token_offset =\
                        convert_iob2_to_tagged_sent(
                            tokens,
                            labels,
                            in_neighbors_list,
                            #out_neighbors_list,
                            token_offset,
                            to_mask_src_and_tgt,
                            has_end_tag)
                      
                    if only_co_occurrence_sent:
                        if is_in_same_sent:
                            tagged_sents.append(tagged_sent)
                            all_sents_in_neighbors.append(in_neighbors_str)
                    else:
                        tagged_sents.append(tagged_sent)
                        all_sents_in_neighbors.append(in_neighbors_str)
                    #all_sents_out_neighbors.append(out_neighbors_str)
                
                
                min_sents_window = 100
                for src_sent_id in src_sent_ids:
                    for tgt_sent_id in tgt_sent_ids:
                        _min_sents_window = abs(src_sent_id - tgt_sent_id)
                        if _min_sents_window < min_sents_window:
                            min_sents_window = _min_sents_window
                            
                num_seq_lens.append(float(len(tagged_sent.split(' '))))
                
                #print('================>id1', id1)
                #print('================>all_sents_in_neighbors', all_sents_in_neighbors)
                
                out_sent = ' '.join(tagged_sents)
                
                if id1 == '-1' or id2 == '-1':
                    continue
                if ' '.join(tagged_sents) == '':
                    continue
                if has_ne_type:
                    instance = document.id + '\t' +\
                               id1type + '\t' +\
                               id2type + '\t' +\
                               id1 + '\t' +\
                               id2 + '\t' +\
                               str(is_in_same_sent) + '\t' +\
                               str(min_sents_window) + '\t' +\
                               out_sent + '\t' +\
                               ' '.join(all_sents_in_neighbors)
                           #' '.join(all_sents_in_neighbors) + '\t' +\
                           #' '.join(all_sents_out_neighbors)
                else:
                    instance = document.id + '\t' +\
                               id1 + '\t' +\
                               id2 + '\t' +\
                               str(is_in_same_sent) + '\t' +\
                               str(min_sents_window) + '\t' +\
                               out_sent + '\t' +\
                               ' '.join(all_sents_in_neighbors)
                    
                if relation_label != neg_label:
                    unique_YES_instances.add(instance)
                
                if is_test_set or (id1 != '-' and id2 != '-'):
                    if has_novelty:
                        relation_label = relation_label.replace('|', '\t')
                    bert_writer.write(instance + '\t' + 
                                      relation_label + '\n')
                                
            number_unique_YES_instances += len(unique_YES_instances)
                    
            bert_writer.flush()
            
        print('number_unique_YES_instances', number_unique_YES_instances)
    #raise('GG')
    if len(num_seq_lens) == 0:
        return 0
    
    return sum(num_seq_lens) / len(num_seq_lens)

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
            
def load_pubtator_into_documents(in_pubtator_file, 
                                 normalized_type_dict = {},
                                 re_id_spliter_str = r'\,',
                                 pmid_2_index_2_groupID_dict = None):
    
    documents = []
    
    with open(in_pubtator_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        
        document = None
        
        annotations = []
        text_instances = []
        relation_pairs = {}
        index2normalized_id = {}
        id2index = {}
        
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
                relation_pairs = {}
                id2index = {}
                index2normalized_id = {}
                continue
            
            tks = line.split('|')
            
            
            if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                pmid = tks[0]
                x = TextInstance(tks[2])
                text_instances.append(x)
            else:
                _tks = line.split('\t')
                if len(_tks) == 6:
                    start = int(_tks[1])
                    end = int(_tks[2])
                    index = _tks[1] + '|' + _tks[2]
                    text = _tks[3]
                    ne_type = _tks[4]
                    ne_type = re.sub('\s*\(.*?\)\s*$', '', ne_type)
                    orig_ne_type = ne_type
                    if ne_type in normalized_type_dict:
                        ne_type = normalized_type_dict[ne_type]
                    
                    _anno = AnnotationInfo(start, end-start, text, ne_type)
                    
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    ids = [x.strip('*') for x in re.split(re_id_spliter_str, _tks[5])]
                    
                    # if annotation has groupID then update its id
                    if orig_ne_type == 'SequenceVariant':
                        if pmid_2_index_2_groupID_dict != None and index in pmid_2_index_2_groupID_dict[pmid]:
                            index2normalized_id[index] = pmid_2_index_2_groupID_dict[pmid][index][0] # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                            _anno.corresponding_gene_id = pmid_2_index_2_groupID_dict[pmid][index][1]
                    for i, _id in enumerate(ids):
                        if pmid_2_index_2_groupID_dict != None and index in pmid_2_index_2_groupID_dict[pmid]:
                            id2index[ids[i]] = index
                            ids[i] = pmid_2_index_2_groupID_dict[pmid][index][0] # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                            _anno.corresponding_gene_id = pmid_2_index_2_groupID_dict[pmid][index][1]
                        else:
                            #ids[i] = re.sub('\s*\(.*?\)\s*$', '', _id)
                            ids[i] = _id
                        
                    
                    _anno.orig_ne_type = orig_ne_type
                    _anno.ids = set(ids)
                    annotations.append(_anno)
                elif len(_tks) == 4 or len(_tks) == 5:
                    
                    id1 = _tks[2]
                    id2 = _tks[3]
                    
                    if pmid_2_index_2_groupID_dict != None and (id1 in id2index) and (id2index[id1] in index2normalized_id):
                        id1 = index2normalized_id[id2index[id1]] # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                    if pmid_2_index_2_groupID_dict != None and (id2 in id2index) and (id2index[id2] in index2normalized_id):
                        id2 = index2normalized_id[id2index[id2]] # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                    rel_type = _tks[1]
                    if len(_tks) == 5:
                        rel_type += '|' + _tks[-1]
                    relation_pairs[(id1, id2)] = rel_type
                    
        if len(text_instances) != 0:
            document = PubtatorDocument(pmid)
            add_annotations_2_text_instances(text_instances, annotations)
            document.text_instances = text_instances
            document.relation_pairs = relation_pairs
            documents.append(document)
    
    return documents
        
def convert_pubtator_to_tsv_file(
        in_pubtator_file,
        out_tsv_file,
        spacy_model,
        re_id_spliter_str,
        src_tgt_pairs,
        normalized_type_dict = {},
        has_novelty = False):
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
        
    all_documents = load_pubtator_into_documents(in_pubtator_file, 
                                                 normalized_type_dict = normalized_type_dict,
                                                 re_id_spliter_str = re_id_spliter_str)
    tokenize_documents_by_spacy(all_documents, spacy_model)
        
    print('=======>len(all_documents)', len(all_documents))
        
    dump_documents_2_bert_gt_format(
        all_documents = all_documents, 
        out_bert_file = out_tsv_file, 
        src_tgt_pairs = src_tgt_pairs,
        has_novelty = has_novelty)

def gen_biored_dataset(
        in_data_dir,
        out_data_dir,
        spacy_model,
        re_id_spliter_str,
        normalized_type_dict,
        has_novelty = False,
        neg_label = 'None'):
    
    in_train_pubtator_file = in_data_dir + 'Train.PubTator'
    in_dev_pubtator_file   = in_data_dir + 'Dev.PubTator'
    in_test_pubtator_file  = in_data_dir + 'Test.PubTator'
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)        
    
    out_train_tsv_file = out_data_dir + 'train.tsv'
    out_dev_tsv_file   = out_data_dir + 'dev.tsv'
    out_test_tsv_file  = out_data_dir + 'test.tsv'
    
    src_tgt_pairs = set(
            [('ChemicalEntity', 'ChemicalEntity'),
             ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
             ('ChemicalEntity', 'GeneOrGeneProduct'),
             ('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'),
             ('GeneOrGeneProduct', 'GeneOrGeneProduct')])
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_train_pubtator_file,
        out_tsv_file     = out_train_tsv_file,
        src_tgt_pairs    = src_tgt_pairs,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        has_novelty      = has_novelty)    
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_dev_pubtator_file,
        out_tsv_file     = out_dev_tsv_file,
        src_tgt_pairs    = src_tgt_pairs,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        has_novelty      = has_novelty)    
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_test_pubtator_file,
        out_tsv_file     = out_test_tsv_file,
        src_tgt_pairs    = src_tgt_pairs,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        has_novelty      = has_novelty)

if __name__ == '__main__':
    
    spacy_model = 'en_core_sci_md'
    normalized_type_dict = {}
    random.seed(1234)
        
    in_data_dir       = 'datasets/biored/BioRED/'
    out_data_dir      = 'datasets/biored/processed/all/'
    re_id_spliter_str= r'\,'  
    normalized_type_dict = {'SequenceVariant':'GeneOrGeneProduct'}
    
    gen_biored_dataset(
        in_data_dir  = in_data_dir,
        out_data_dir = out_data_dir,
        spacy_model  = spacy_model,
        re_id_spliter_str = re_id_spliter_str,
        normalized_type_dict = normalized_type_dict)
    
    out_data_dir      = 'datasets/biored/processed/novelty/'
    re_id_spliter_str= r'\,'  
    normalized_type_dict = {'SequenceVariant':'GeneOrGeneProduct'}
    
    gen_biored_dataset(
        in_data_dir  = in_data_dir,
        out_data_dir = out_data_dir,
        spacy_model  = spacy_model,
        re_id_spliter_str = re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        has_novelty          = True)
    