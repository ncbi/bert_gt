#!/bin/bash

cuda_visible_devices=$1

cuda_visible_devices=$cuda_visible_devices python src/run_bert_gt.py \
  --do_train=true \
  --do_eval=false \
  --do_predict=true \
  --task_name="cdr" \
  --vocab_file=biobert_v1.1_pubmed/vocab.txt \
  --bert_config_file=biobert_v1.1_pubmed/bert_config.json \
  --init_checkpoint=biobert_v1.1_pubmed/model.ckpt-1000000 \
  --num_train_epochs=20.0 \
  --max_seq_length=512 \
  --train_batch_size=10 \
  --learning_rate=5e-5 \
  --use_balanced_neg=true \
  --surrounding_words_distance=5 \
  --do_lower_case=false \
  --entity_num=2 \
  --max_num_neighbors=5 \
  --max_num_entity_indices=20 \
  --data_dir=datasets/cdr/processed/ \
  --output_dir=out_cdr_model/

python src/run_eval.py --task=cdr --output_path=out_cdr_model/test_results.tsv --answer_path=datasets/cdr/processed/test.tsv