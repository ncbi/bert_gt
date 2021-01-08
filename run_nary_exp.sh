#!/bin/bash

cuda_visible_devices=$1
task_name=$2
in_data_dir=$3
entity_num=3

if [[ $task_name =~ "dgv" ]]
then
    in_data_dir+='/drug_gene_var'
else
    in_data_dir+='/drug_var'
    entity_num=2
fi
for x in {0..4}
do
  cuda_visible_devices=$cuda_visible_devices python src/run_bert_gt.py \
    --task_name=$task_name \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --vocab_file=biobert_v1.1_pubmed/vocab.txt \
    --bert_config_file=biobert_v1.1_pubmed/bert_config.json \
    --init_checkpoint=biobert_v1.1_pubmed/model.ckpt-1000000 \
    --max_seq_length=512 \
    --max_num_entity_indices=1 \
    --train_batch_size=8 \
    --learning_rate=1e-5 \
    --num_train_epochs=3.0 \
    --do_lower_case=false \
    --shortest_path_neighbors=false \
    --entity_num=$entity_num \
    --max_num_neighbors=20 \
    --max_num_entity_indices=1 \
    --data_dir=$in_data_dir/cv$x/ \
    --output_dir=out_model_${task_name}_cv$x/
done
for x in {0..4}
do
  python src/run_eval.py --task=$task_name --output_path=out_model_${task_name}_cv$x/test_results.tsv --answer_path=$in_data_dir/cv$x/test.tsv
done