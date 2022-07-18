#!/bin/bash

echo 'Downloading the dataset'
curl -k -o BIORED.zip https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip
echo 'Unzip the dataset'
mkdir -p datasets/biored
unzip BIORED.zip -d datasets/biored
rm BIORED.zip
echo 'Installing Sci spacy en_core_sci_md model'
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
echo 'Converting the dataset into BERT-GT input format'
python src/dataset_format_converter/convert_biored_2_bert_gt.py