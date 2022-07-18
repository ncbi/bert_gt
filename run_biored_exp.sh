#!/bin/bash

cuda_visible_devices=$1

task_names=('biored_all_mul' 'biored_novelty')


for task_name in ${task_names[*]}
do
    in_data_dir='datasets/biored/processed'
    entity_num=2
    no_neg_for_train_dev=false
    
    if [[ $task_name =~ "all" ]]
    then
        in_data_dir+='/all'
    elif [[ $task_name =~ "novelty" ]]
    then
        in_data_dir+='/novelty'
        no_neg_for_train_dev=true
    fi
      
    
    cuda_visible_devices=$cuda_visible_devices python src/run_bert_gt.py \
      --task_name=$task_name \
      --do_train=true \
      --do_eval=false \
      --do_predict=true \
      --data_dir=$in_data_dir \
      --test_file=${in_data_dir}/test.tsv \
      --test_has_header=false \
      --use_balanced_neg=true \
      --no_neg_for_train_dev=$no_neg_for_train_dev \
      --vocab_file=biobert_v1.1_pubmed/vocab.txt \
      --bert_config_file=biobert_v1.1_pubmed/bert_config.json \
      --init_checkpoint=biobert_v1.1_pubmed/model.ckpt-1000000 \
      --max_seq_length=512 \
      --train_batch_size=8 \
      --learning_rate=1e-5 \
      --num_train_epochs=30.0 \
      --save_checkpoints_steps=90000 \
      --do_lower_case=false \
      --shortest_path_neighbors=false \
      --entity_num=$entity_num \
      --max_num_neighbors=20 \
      --max_num_entity_indices=10 \
      --data_dir=$in_data_dir/ \
      --output_dir=out_model_${task_name}/
    cp out_model_${task_name}/test_results.tsv out_${task_name}_test_results.tsv
done

python src/run_eval_biored.py