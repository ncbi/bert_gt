#!/bin/bash

echo 'Downloading the dataset'
curl -L -o bert_data.zip https://github.com/ncbi-nlp/BLUE_Benchmark/releases/download/0.1/bert_data.zip
echo 'Unzip the dataset'
unzip bert_data.zip -d datasets
mv datasets/bert_data/ddi2013-type datasets
mv datasets/bert_data/ChemProt datasets
echo 'Installing Sci spacy en_core_sci_md model'
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
echo 'Converting the dataset into BERT-GT input format'
python src/dataset_format_converter/convert_bluebert_2_bert_gt.py