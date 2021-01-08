#!/bin/bash

echo 'Downloading the dataset'
curl -k -o CDR_Data.zip https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip
echo 'Unzip the dataset'
mkdir -p datasets/cdr
unzip CDR_Data.zip -d datasets/cdr
rm CDR_Data.zip
echo 'Installing Sci spacy en_core_sci_md model'
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz
echo 'Converting the dataset into BERT-GT input format'
python src/dataset_format_converter/convert_cdr_2_bert_gt.py