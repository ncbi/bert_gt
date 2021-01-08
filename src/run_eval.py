# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 19:41:40 2020

@author: potin
"""

import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str,  help='')
parser.add_argument('--answer_path', type=str,  help='')
parser.add_argument('--task', type=str,  default="binary", help='default:binary, possible other options:{chemprot}')
args = parser.parse_args()


testdf = pd.read_csv(args.answer_path, sep="\t", index_col=0)
preddf = pd.read_csv(args.output_path, sep="\t", header=None)


if args.task == "cdr":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [str(np.argmax(v)) for v in pred]
    
    tp_intra_sent = 0.
    fp_intra_sent = 0.
    fn_intra_sent = 0.
    
    tp_inter_sents = 0.
    fp_inter_sents = 0.
    fn_inter_sents = 0.
    
    test_labels = None
    test_is_in_same_sents = None
    try:
        test_labels = testdf["label"]
        test_is_in_same_sents = testdf["is_in_same_sent"]
    except:
        testdf = pd.read_csv(args.answer_path, sep="\t", header=None)
        test_labels = testdf.iloc[:,-1]
        test_is_in_same_sents = testdf.iloc[:,3]
    for _pred, _gold, _is_in_same_sent in zip(pred_class, test_labels, test_is_in_same_sents):
        _gold = str(_gold)
        if _is_in_same_sent:
            if _gold == _pred and _gold != '0':
                tp_intra_sent += 1
            elif _gold != '0' and _pred != '0':
                fn_intra_sent += 1
                fp_intra_sent += 1
            elif _gold != '0' and _pred == '0':
                fn_intra_sent += 1
            elif _gold == '0' and _pred != '0':
                fp_intra_sent += 1
        else:
            if _gold == _pred and _gold != '0':
                tp_inter_sents += 1
            elif _gold != '0' and _pred != '0':
                fn_inter_sents += 1
                fp_inter_sents += 1
            elif _gold != '0' and _pred == '0':
                fn_inter_sents += 1
            elif _gold == '0' and _pred != '0':
                fp_inter_sents += 1
                
    print('Evaluate intra-sent score')
    print('TP      FP      FN      Precision       Recall  F1-measure')
    prec = tp_intra_sent / (tp_intra_sent + fp_intra_sent) if (tp_intra_sent + fp_intra_sent) > 0. else 0.
    reca = tp_intra_sent / (tp_intra_sent + fn_intra_sent) if (tp_intra_sent + fn_intra_sent) > 0. else 0.
    f1 = (2*prec*reca) / (prec+reca) if (prec+reca) > 0. else 0.
    print(tp_intra_sent, fp_intra_sent, fn_intra_sent, prec, reca, f1)

    print('Evaluate inter-sents score')
    print('TP      FP      FN      Precision       Recall  F1-measure')
    prec = tp_inter_sents / (tp_inter_sents + fp_inter_sents) if (tp_inter_sents + fp_inter_sents) > 0. else 0.
    reca = tp_inter_sents / (tp_inter_sents + fn_inter_sents) if (tp_inter_sents + fn_inter_sents) > 0. else 0.
    f1 = (2*prec*reca) / (prec+reca) if (prec+reca) > 0. else 0.
    print(tp_inter_sents, fp_inter_sents, fn_inter_sents, prec, reca, f1)
    
    print('Evaluate overall score')
    print('TP      FP      FN      Precision       Recall  F1-measure')
    
    tp_inter_sents += tp_intra_sent
    fp_inter_sents += fp_intra_sent
    fn_inter_sents += fn_intra_sent
    
    prec = tp_inter_sents / (tp_inter_sents + fp_inter_sents) if (tp_inter_sents + fp_inter_sents) > 0. else 0.
    reca = tp_inter_sents / (tp_inter_sents + fn_inter_sents) if (tp_inter_sents + fn_inter_sents) > 0. else 0.
    f1 = (2*prec*reca) / (prec+reca) if (prec+reca) > 0. else 0.
    print(tp_inter_sents, fp_inter_sents, fn_inter_sents, prec, reca, f1)
elif args.task == "nary_dgv_mul" or args.task == "nary_dv_mul":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    tp_num = 0.
    num_all = 0.
    all_dict = {}
    labels = ["None", "resistance", "response", "resistance or non-response", "sensitivity"]
    for _pred, _gold in zip(pred_class, testdf["label"]):
        #print(_pred)
        #print(_gold)
        if _pred == labels.index(_gold):
            tp_num += 1.
            if _gold not in all_dict:
                all_dict[_gold] = 0.
            all_dict[_gold] += 1.
        num_all += 1.
    print("{:11s} : {:.2%}".format('Overall',tp_num/num_all))
elif args.task == "nary_dgv_bin" or args.task == "nary_dv_bin":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    tp_num = 0.
    num_all = 0.
    all_dict = {}
    labels = ["NO", "YES"]
    for _pred, _gold_label in zip(pred_class, testdf["label"]):
        #print(_pred)
        #print(_gold)
        if _gold_label == 'None':
            _gold = 0
        else:
            _gold = 1
        if _pred == _gold:
            tp_num += 1.
            if _gold not in all_dict:
                all_dict[_gold] = 0.
            all_dict[_gold] += 1.
        num_all += 1.
    print("{:11s} : {:.2%}".format('Overall',tp_num/num_all))
elif args.task == "ddi":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    
    tp_num_dict = {}
    fp_num_dict = {}
    fn_num_dict = {}
    prec_dict = {}
    recall_dict = {}
    f1_dict = {}
    
    labels = ["DDI-false", "DDI-mechanism", "DDI-effect", "DDI-advise", "DDI-int"]
    
    for label in labels:
        tp_num_dict[label] = 0.
        fp_num_dict[label] = 0.
        fn_num_dict[label] = 0.
        prec_dict[label] = 0.
        recall_dict[label] = 0.
        f1_dict[label] = 0.
        
    for _pred, _gold in zip(pred_class, testdf["label"]):
        #print(_pred)
        #print(_gold)
        if _pred == labels.index(_gold):
            tp_num_dict[_gold] += 1.
        else:
            fp_num_dict[labels[_pred]] += 1.
            fn_num_dict[_gold] += 1.
    
    avg_prec = 0.
    avg_recall = 0.
    avg_f1 = 0.
    
    for label in labels:
        if label == "DDI-false":
            continue
        
        tp_num = tp_num_dict[label]
        fp_num = fp_num_dict[label]
        fn_num = fn_num_dict[label]
        
        prec = tp_num / (tp_num + fp_num) if (tp_num + fp_num) > 0 else 0
        recall = tp_num / (tp_num + fn_num) if (tp_num + fn_num) > 0 else 0
        f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
        
        prec_dict[label] = prec
        recall_dict[label] = recall
        f1_dict[label] = f1
        
        avg_prec += prec
        avg_recall += recall
        avg_f1 += f1
        
        print(label + '\t' +
              str(prec) + '\t' +
              str(recall) + '\t' +
              str(f1) + '\n')
        
    print('Overall\t' +
        str(avg_prec / float(len(labels) - 1)) + '\t' +
        str(avg_recall / float(len(labels) - 1)) + '\t' +
        str(avg_f1 / float(len(labels) - 1)) + '\n')
elif args.task == "chemprot":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    
    tp_num_dict = {}
    fp_num_dict = {}
    fn_num_dict = {}
    prec_dict = {}
    recall_dict = {}
    f1_dict = {}
    
    labels = ["false", "CPR:4", "CPR:6", "CPR:5", "CPR:9", "CPR:3"]
    
    for label in labels:
        tp_num_dict[label] = 0.
        fp_num_dict[label] = 0.
        fn_num_dict[label] = 0.
        prec_dict[label] = 0.
        recall_dict[label] = 0.
        f1_dict[label] = 0.
        
    for _pred, _gold in zip(pred_class, testdf["label"]):
        #print(_pred)
        #print(_gold)
        if _pred == labels.index(_gold):
            tp_num_dict[_gold] += 1.
        else:
            fp_num_dict[labels[_pred]] += 1.
            fn_num_dict[_gold] += 1.
    
    all_tp_num = 0.
    all_fp_num = 0.
    all_fn_num = 0.
    
    for label in labels:
        if label == "false":
            continue
        
        tp_num = tp_num_dict[label]
        fp_num = fp_num_dict[label]
        fn_num = fn_num_dict[label]
        
        prec = tp_num / (tp_num + fp_num) if (tp_num + fp_num) > 0 else 0
        recall = tp_num / (tp_num + fn_num) if (tp_num + fn_num) > 0 else 0
        f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
        
        prec_dict[label] = prec
        recall_dict[label] = recall
        f1_dict[label] = f1
        
        all_tp_num += tp_num
        all_fp_num += fp_num
        all_fn_num += fn_num
        
        print(label + '\t' +
              str(prec) + '\t' +
              str(recall) + '\t' +
              str(f1) + '\n')
    
    prec = all_tp_num / (all_tp_num + all_fp_num) if (all_tp_num + all_fp_num) > 0 else 0
    recall = all_tp_num / (all_tp_num + all_fn_num) if (all_tp_num + all_fn_num) > 0 else 0
    f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
    
    
    print('Overall\t' +
        str(prec) + '\t' +
        str(recall) + '\t' +
        str(f1) + '\n')
