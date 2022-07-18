# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:41:35 2021

@author: laip2
"""

import pandas as pd
from collections import defaultdict
import numpy as np


def add_relation_pair_2_novelty_dict(
        in_gold_tsv_file, 
        in_pred_tsv_file,
        pmid_2_rel_pair_2_novelty_dict,
        label = 'novelty',
        label_index = -1):
    
    #print('in_gold_tsv_file', in_gold_tsv_file)
    #print('in_pred_tsv_file', in_pred_tsv_file)
    testdf = pd.read_csv(in_gold_tsv_file, sep="\t", index_col=0)
    try:
        preddf = pd.read_csv(in_pred_tsv_file, sep="\t", header=None)
    except:
        return
    
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    
    
    test_labels = None
    test_is_in_same_sents = None
    
    try:
        test_labels           = testdf[label]
        index_list            = testdf["pmid"]
        id1_list              = testdf["id1"]
        id2_list              = testdf["id2"]
    except:
        testdf = pd.read_csv(in_gold_tsv_file, sep="\t", header=None)
        test_labels           = testdf.iloc[:,label_index]
        index_list            = testdf.iloc[:,0]
        id1_list              = testdf.iloc[:,1]
        id2_list              = testdf.iloc[:,2]
    
    labels = ['None', 
              'No', 
              'Novel']
    for _pred, _gold, index, id1, id2 in zip(pred_class, test_labels, index_list, id1_list, id2_list):
        pred_label = labels[_pred]
        #if pred_label != 'None':
            #print(index, pred_label)
        sindex = str(index)
        if sindex not in pmid_2_rel_pair_2_novelty_dict:
            pmid_2_rel_pair_2_novelty_dict[sindex] = dict()
                #print(sindex)
        if (str(id1), str(id2)) not in pmid_2_rel_pair_2_novelty_dict[sindex]:
            pmid_2_rel_pair_2_novelty_dict[sindex][(str(id1), str(id2))] = pred_label

def add_relation_pairs_dict(
        in_gold_tsv_file, 
        in_pred_tsv_file,
        pmid_2_rel_pairs_dict,
        label = 'label',
        label_index = -2):
    
    #print('in_gold_tsv_file', in_gold_tsv_file)
    #print('in_pred_tsv_file', in_pred_tsv_file)
    testdf = pd.read_csv(in_gold_tsv_file, sep="\t", index_col=0)
    try:
        preddf = pd.read_csv(in_pred_tsv_file, sep="\t", header=None)
    except:
        return
    
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    
    
    test_labels = None
    test_is_in_same_sents = None
    
    try:
        test_labels           = testdf[label]
        index_list            = testdf["pmid"]
        id1_list              = testdf["id1"]
        id2_list              = testdf["id2"]
    except:
        testdf = pd.read_csv(in_gold_tsv_file, sep="\t", header=None)
        test_labels           = testdf.iloc[:,label_index]
        index_list            = testdf.iloc[:,0]
        id1_list              = testdf.iloc[:,1]
        id2_list              = testdf.iloc[:,2]
    
    labels = ['None', 
                'Association', 
                'Bind',
                'Comparison',
                'Conversion',
                'Cotreatment',
                'Drug_Interaction',
                'Negative_Correlation',
                'Positive_Correlation']
    for _pred, _gold, index, id1, id2 in zip(pred_class, test_labels, index_list, id1_list, id2_list):
        pred_label = labels[_pred]
        if pred_label != 'None':
            #print(index, pred_label)
            sindex = str(index)
            if sindex not in pmid_2_rel_pairs_dict:
                pmid_2_rel_pairs_dict[sindex] = set()
                #print(sindex)
            pmid_2_rel_pairs_dict[sindex].add((str(id1), str(id2), pred_label))
            
def add_gold_relation_pairs_dict(
        in_gold_tsv_file,
        pmid_2_rel_pairs_dict,
        label = 'label',
        label_index = -2):
    
    #print('in_gold_tsv_file', in_gold_tsv_file)
    #print('in_pred_tsv_file', in_pred_tsv_file)
    testdf = pd.read_csv(in_gold_tsv_file, sep="\t", index_col=0)
    
    try:
        test_labels           = testdf[label]
        index_list            = testdf["pmid"]
        id1_list              = testdf["id1"]
        id2_list              = testdf["id2"]
    except:
        testdf = pd.read_csv(in_gold_tsv_file, sep="\t", header=None)
        test_labels           = testdf.iloc[:,label_index]
        index_list            = testdf.iloc[:,0]
        id1_list              = testdf.iloc[:,1]
        id2_list              = testdf.iloc[:,2]
    
    for _gold, index, id1, id2 in zip(test_labels, index_list, id1_list, id2_list):
        if _gold != 'None':
            #print(index, pred_label)
            sindex = str(index)
            if sindex not in pmid_2_rel_pairs_dict:
                pmid_2_rel_pairs_dict[sindex] = set()
                #print(sindex)
            pmid_2_rel_pairs_dict[sindex].add((str(id1), str(id2), _gold))
        
def retrive_relation_pairs_dict(in_pubtator_file, 
                                to_binary=False,
                                has_novelty=False):
    
    pmid_2_rel_pairs_dict = {}
    pmid_2_id2ne_type_dict = {}
    
    rel_pairs_set = set()
    id2ne_type_dict = {}
    
    with open(in_pubtator_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        
        for line in pub_reader:
            line = line.rstrip()
            
            if line == '':
                #print('len(rel_pairs_set)', pmid, len(rel_pairs_set))
                pmid_2_rel_pairs_dict[pmid] = rel_pairs_set
                pmid_2_id2ne_type_dict[pmid] = id2ne_type_dict
                rel_pairs_set = set()
                id2ne_type_dict = {}
            else:
                tks = line.split('|')
                if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    pmid = tks[0]
                else:
                    tks = line.split('\t')
                    if len(tks) == 5 or len(tks) == 4:
                        rel_type = tks[1]
                        id1 = tks[2]
                        id2 = tks[3]
                        #id1      = re.sub('\s*\(.*?\)\s*$', '', tks[2])
                        #id2      = re.sub('\s*\(.*?\)\s*$', '', tks[3])
                        ne_type1 = id2ne_type_dict[id1]
                        ne_type2 = id2ne_type_dict[id2]
                        
                        if has_novelty and len(tks) != 4:
                            novelty  = tks[4]
                        else:
                            novelty  = 'None'
                        
                        if to_binary:
                            rel_type = 'Association'
                            
                        if has_novelty:
                            rel_type += '|' + novelty
                        #rel_pairs_set.add((id2, id1, rel_type))
                        if ne_type1 == 'DiseaseOrPhenotypicFeature' and ne_type2 == 'DiseaseOrPhenotypicFeature':
                            continue
                        elif ne_type1 == 'GeneOrGeneProduct' and ne_type2 == 'SequenceVariant':
                            continue
                        elif ne_type2 == 'GeneOrGeneProduct' and ne_type1 == 'SequenceVariant':
                            continue
                        elif ne_type2 == 'GeneOrGeneProduct' and ne_type1 == 'OrganismTaxon':
                            continue
                        elif ne_type2 == 'OrganismTaxon' and ne_type1 == 'GeneOrGeneProduct':
                            continue
                        
                        if str(id2) > str(id1):
                            rel_pairs_set.add((id1, id2, rel_type, ne_type1, ne_type2))
                        else:
                            rel_pairs_set.add((id2, id1, rel_type, ne_type2, ne_type1))
                    else:
                        #id = re.sub('\s*\(.*?\)\s*$', '', tks[5])
                        id = tks[5]
                        ne_type = tks[4]
                        '''if ne_type == 'SequenceVariant':
                            ne_type = 'GeneOrGeneProduct'''
                        for _id in id.split(','):
                            id2ne_type_dict[_id] = ne_type
                        
        if len(rel_pairs_set) != 0:
            pmid_2_rel_pairs_dict[pmid] = rel_pairs_set
            #print('len(rel_pairs_set)', pmid, len(rel_pairs_set))
    
    return pmid_2_rel_pairs_dict, pmid_2_id2ne_type_dict
     
    
def eval(in_gold_pubtator_file,
         in_pred_pubtator_file,
         out_error_cases_file,
         in_pmid_file='',
         to_binary=False,
         has_novelty=False,
         eval_tags=set(['Development']),
         eval_pmids=set()):
                    
    gold_relation_pairs_dict, id2type_dict = retrive_relation_pairs_dict(in_gold_pubtator_file, to_binary, has_novelty)
    pred_relation_pairs_dict, _ = retrive_relation_pairs_dict(in_pred_pubtator_file, to_binary, has_novelty)
    
    if in_pmid_file != '':
        with open(in_pmid_file, 'r', encoding='utf8') as pmid_reader:
            for line in pmid_reader:
                tks = line.rstrip().split('\t')
                pmid = tks[0]
                if tks[1] in eval_tags:
                    eval_pmids.add(pmid)
    
    tp_count = 0.
    fp_count = 0.
    fn_count = 0.
    
    typed_tp_count = defaultdict(float)
    typed_fp_count = defaultdict(float)
    typed_fn_count = defaultdict(float)
    
    
    rel_typed_tp_count = defaultdict(float)
    rel_typed_fp_count = defaultdict(float)
    rel_typed_fn_count = defaultdict(float)
    
    outputs = []
    
    has_eval_set = len(eval_pmids) > 0
    
    for pmid in gold_relation_pairs_dict.keys():
        if has_eval_set and pmid not in eval_pmids:
            continue
        if pmid in pred_relation_pairs_dict:
            pred_relation_pairs = pred_relation_pairs_dict[pmid]
            gold_relation_pairs = gold_relation_pairs_dict[pmid]       
            print('==========>>')
            print('len(pred_relation_pairs)', len(pred_relation_pairs))
            print('len(gold_relation_pairs)', len(gold_relation_pairs))
            for pair in pred_relation_pairs:
                print('pred:', pair[0], pair[1], pair[2])
            for pair in gold_relation_pairs:
                print('gold:', pair[0], pair[1], pair[2])
            for gold_pair in gold_relation_pairs:
                if gold_pair in pred_relation_pairs:
                    tp_count += 1.
                    ne1_type = gold_pair[3]
                    ne2_type = gold_pair[4]
                    rel_type = gold_pair[2]

                    rel_typed_tp_count[rel_type] += 1.
                                       
                    if ne1_type <= ne2_type:
                        typed_tp_count[(ne1_type, ne2_type)] += 1.
                        outputs.append(ne1_type + '|' + ne2_type + 'TP:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                    else:
                        typed_tp_count[(ne2_type, ne1_type)] += 1.
                        outputs.append(ne2_type + '|' + ne1_type + 'TP:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                else:
                    fn_count += 1.
                    ne1_type = gold_pair[3]
                    ne2_type = gold_pair[4]
                    rel_type = gold_pair[2]

                    rel_typed_fn_count[rel_type] += 1.
                    
                    if ne1_type <= ne2_type:
                        typed_fn_count[(ne1_type, ne2_type)] += 1.
                        outputs.append(ne1_type + '|' + ne2_type + 'FN:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                    else:
                        typed_fn_count[(ne2_type, ne1_type)] += 1.
                        outputs.append(ne2_type + '|' + ne1_type + 'FN:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
            for pred_pair in pred_relation_pairs:
                if pred_pair not in gold_relation_pairs:
                    fp_count += 1.                    
                    ne1_type = pred_pair[3]
                    ne2_type = pred_pair[4]
                    rel_type = pred_pair[2]

                    rel_typed_fp_count[rel_type] += 1.
                    
                    if ne1_type <= ne2_type:
                        typed_fp_count[(ne1_type, ne2_type)] += 1.
                        outputs.append(ne1_type + '|' + ne2_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
                    else:
                        typed_fp_count[(ne2_type, ne1_type)] += 1.
                        outputs.append(ne2_type + '|' + ne1_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
    for pmid in pred_relation_pairs_dict.keys():
        if has_eval_set and pmid not in eval_pmids:
            continue
        if pmid not in gold_relation_pairs_dict:
            pred_relation_pairs = pred_relation_pairs_dict[pmid]
            for pred_pair in pred_relation_pairs:
                fp_count += 1.
                ne1_type = pred_pair[3]
                ne2_type = pred_pair[4]
                rel_type = pred_pair[2]

                rel_typed_fp_count[rel_type] += 1.
                
                if ne1_type <= ne2_type:
                    typed_fp_count[(ne1_type, ne2_type)] += 1.
                    outputs.append(ne1_type + '|' + ne2_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
                else:
                    typed_fp_count[(ne2_type, ne1_type)] += 1.
                    outputs.append(ne2_type + '|' + ne1_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
    
    all_pair_types = set(typed_fn_count.keys())
    all_pair_types.update(list(typed_tp_count.keys()))
    all_pair_types.update(list(typed_fp_count.keys()))
    all_pair_types = list(all_pair_types)
    all_pair_types.sort()
    
    all_rel_types = set(rel_typed_fn_count.keys())
    all_rel_types.update(list(rel_typed_tp_count.keys()))
    all_rel_types.update(list(rel_typed_fp_count.keys()))
    all_rel_types = list(all_rel_types)
    all_rel_types.sort()
    
    prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) != 0 else 0.
    reca = tp_count / (tp_count + fn_count) if (tp_count + fn_count) != 0 else 0.
    fsco = (2*prec*reca) / (prec + reca) if (prec + reca) != 0 else 0.
    
    with open(out_error_cases_file, 'w', encoding='utf8') as score_writer:
        score_writer.write('Overall')
        score_writer.write('\t' + str(int(tp_count + fn_count)))
        score_writer.write('\t' + str(int(tp_count)))
        score_writer.write('\t' + str(int(fp_count)))
        score_writer.write('\t' + str(int(fn_count)))
        score_writer.write('\t' + str(prec))
        score_writer.write('\t' + str(reca))
        score_writer.write('\t' + str(fsco) + '\n')
        score_writer.write('\n')
        for rel_type in all_rel_types:
            tp = int(rel_typed_tp_count[rel_type])
            fp = int(rel_typed_fp_count[rel_type])
            fn = int(rel_typed_fn_count[rel_type])
            prec = float(tp) / (float(tp) + float(fp)) if (float(tp) + float(fp)) > 0 else 0.
            reca = float(tp) / (float(tp) + float(fn)) if (float(tp) + float(fn)) > 0 else 0.
            fsco = (2*prec*reca) / (prec + reca) if (prec + reca) > 0 else 0.
            score_writer.write(rel_type)
            score_writer.write('\t' + str(tp + fn))
            score_writer.write('\t' + str(tp))
            score_writer.write('\t' + str(fp))
            score_writer.write('\t' + str(fn))
            score_writer.write('\t' + str(prec))
            score_writer.write('\t' + str(reca))
            score_writer.write('\t' + str(fsco) + '\n')
        score_writer.write('\n')
        for pair_types in all_pair_types:
            tp = int(typed_tp_count[pair_types])
            fp = int(typed_fp_count[pair_types])
            fn = int(typed_fn_count[pair_types])
            prec = float(tp) / (float(tp) + float(fp)) if (float(tp) + float(fp)) > 0 else 0.
            reca = float(tp) / (float(tp) + float(fn)) if (float(tp) + float(fn)) > 0 else 0.
            fsco = (2*prec*reca) / (prec + reca) if (prec + reca) > 0 else 0.
            score_writer.write(pair_types[0] + '|' + pair_types[1])
            score_writer.write('\t' + str(tp + fn))
            score_writer.write('\t' + str(tp))
            score_writer.write('\t' + str(fp))
            score_writer.write('\t' + str(fn))
            score_writer.write('\t' + str(prec))
            score_writer.write('\t' + str(reca))
            score_writer.write('\t' + str(fsco) + '\n')
        score_writer.write('\n')
        for output in outputs:
            score_writer.write(output + '\n')
            
    return tp_count, fp_count, fn_count, prec, reca, fsco
    
def dump_pred_2_pubtator_file(in_pubtator_file,                              
                              out_pubtator_pred_file, 
                              task_names, 
                              bin_or_mul_label,
                              set_all_novelty_to_novel,
                              in_pmid_file = '',
                              is_test = False,
                              eval_tag = 'Dev'):
    
    pmid_2_rel_pairs_dict = {}
    pmid_2_rel_pair_2_novelty_dict = {}
    
    eval_pmids = set()
    
    if in_pmid_file != '':
        with open(in_pmid_file, 'r', encoding='utf8') as pmid_reader:
            for line in pmid_reader:
                tks = line.rstrip().split('\t')
                pmid = tks[0]
                if tks[1].startswith(eval_tag):
                    eval_pmids.add(pmid)
    
    in_gold_tsv_file = 'datasets/biored/processed/all/test.tsv'
    in_pred_tsv_file = 'out_biored_all_' + bin_or_mul_label + '_test_results.tsv'
    add_relation_pairs_dict(
            in_gold_tsv_file,
            in_pred_tsv_file,
            pmid_2_rel_pairs_dict)
    
    in_gold_tsv_file = 'datasets/biored/processed/novelty/test.tsv'
    in_pred_tsv_file = 'out_biored_novelty_test_results.tsv'
    add_relation_pair_2_novelty_dict(
            in_gold_tsv_file,
            in_pred_tsv_file,
            pmid_2_rel_pair_2_novelty_dict)
        
    pred_writer = open(out_pubtator_pred_file, 'w', encoding='utf8')
    
    pmids = sorted(list(pmid_2_rel_pairs_dict.keys()), reverse=True)
    for pmid in pmids:
        print(pmid)
    
    id2ne_type_dict = {}
            
    with open(in_pubtator_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        
        for line in pub_reader:
            line = line.rstrip()
            
            if line == '':
                #print('len(pmid_2_rel_pairs_dict)', len(pmid_2_rel_pairs_dict))
                #print('len(pmid_2_rel_pairs_dict[pmid])', len(pmid_2_rel_pairs_dict[pmid]))
                if in_pmid_file != '' and pmid not in eval_pmids:
                    pmid = ''
                    continue
                if pmid in pmid_2_rel_pairs_dict:
                    #print('len(pmid_2_rel_pairs_dict[pmid])', len(pmid_2_rel_pairs_dict[pmid]))
                    for id1, id2, rel_type in pmid_2_rel_pairs_dict[pmid]:
                        
                        if set_all_novelty_to_novel:
                            novelty = 'Novel'
                        else:
                            if pmid in pmid_2_rel_pair_2_novelty_dict and (id1, id2) in pmid_2_rel_pair_2_novelty_dict[pmid]:
                                novelty = pmid_2_rel_pair_2_novelty_dict[pmid][(id1, id2)]
                            else:
                                novelty = 'None'
                            if novelty == 'None':
                                novelty = 'Novel'
                        
                        pred_writer.write(pmid + 
                                  '\t' + rel_type + 
                                  '\t' + id1 + 
                                  '\t' + id2 + 
                                  '\t' + novelty + '\n')
                    pmid = ''        
                pred_writer.write('\n')
                id2ne_type_dict = {}
                        
            else:
                tks = line.split('|')
                if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    pmid = tks[0]
                    if in_pmid_file != '' and pmid not in eval_pmids:
                        continue
                    pred_writer.write(line + '\n')
                else:
                    tks = line.split('\t')
                    pmid = tks[0]
                    if in_pmid_file != '' and pmid not in eval_pmids:
                        continue
                    if len(tks) == 6:
                        #print(line)
                        pred_writer.write(line + '\n')
                        id = tks[5]
                        #id = re.sub('\s*\(.*?\)\s*$', '', tks[5])
                        ne_type = tks[4]
                        '''if ne_type == 'SequenceVariant':
                            ne_type = 'GeneOrGeneProduct'''
                        id2ne_type_dict[id] = ne_type
        if in_pmid_file == '' or pmid in eval_pmids:
            if pmid != '' and pmid in pmid_2_rel_pairs_dict:
                for id1, id2, rel_type in pmid_2_rel_pairs_dict[pmid]:
                    
                    if pmid in pmid_2_rel_pair_2_novelty_dict and (id1, id2) in pmid_2_rel_pair_2_novelty_dict[pmid]:
                        novelty = pmid_2_rel_pair_2_novelty_dict[pmid][(id1, id2)]
                    
                    if novelty == 'None':
                        novelty = 'Novel'
                    pred_writer.write(pmid + 
                                      '\t' + rel_type + 
                                      '\t' + id1 + 
                                      '\t' + id2 + 
                                      '\t' + novelty + '\n')
                    
    pred_writer.close()
    
def run_train_test_eval(in_gold_pubtator_file):
    
    task_names = ['biored_all_mul']        
    bin_or_mul_label       = 'mul'
    out_pred_pubtator_file = 'biored_pred_mul.txt'
    out_error_cases_file   = 'biored_mul_score.txt'
    out_novelty_cases_file = 'biored_mul_novelty_score.txt'
    
    dump_pred_2_pubtator_file(in_pubtator_file       = in_gold_pubtator_file, 
                              out_pubtator_pred_file = out_pred_pubtator_file, 
                              task_names             = task_names, 
                              bin_or_mul_label       = bin_or_mul_label,
                              set_all_novelty_to_novel = False)
    
    eval(in_gold_pubtator_file, 
         out_pred_pubtator_file, 
         out_error_cases_file,
         to_binary = False)
    
    eval(in_gold_pubtator_file, 
         out_pred_pubtator_file, 
         out_novelty_cases_file,
         to_binary = False,
         has_novelty = True)
    
        
    out_error_cases_file   = 'biored_bin_score.txt'
    out_novelty_cases_file = 'biored_bin_novelty_score.txt'
        
    eval(in_gold_pubtator_file, 
         out_pred_pubtator_file, 
         out_error_cases_file,
         to_binary = True)
    
    eval(in_gold_pubtator_file, 
         out_pred_pubtator_file, 
         out_novelty_cases_file,
         to_binary = True,
         has_novelty = True)
    
if __name__ == '__main__':
    
    in_gold_pubtator_file    = 'datasets/biored/BioRED/Test.PubTator'
    
    run_train_test_eval(in_gold_pubtator_file)
    