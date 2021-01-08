#!/bin/bash

cuda_visible_devices=$1

cuda_visible_devices=$cuda_visible_devices python src/run_bert_gt.py \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --task_name="ddi" \
  --vocab_file=biobert_v1.1_pubmed/vocab.txt \
  --bert_config_file=biobert_v1.1_pubmed/bert_config.json \
  --init_checkpoint=biobert_v1.1_pubmed/model.ckpt-1000000 \
  --num_train_epochs=15.0 \
  --max_seq_length=256 \
  --train_batch_size=8 \
  --surrounding_words_distance=20 \
  --do_lower_case=false \
  --entity_num=2 \
  --max_num_neighbors=20 \
  --max_num_entity_indices=20 \
  --data_dir=datasets/ddi2013-type/processed/ \
  --output_dir=out_ddi_model/

python src/run_eval.py --task=ddi --output_path=out_ddi_model/test_results.tsv --answer_path=datasets/ddi2013-type/processed/test.tsv