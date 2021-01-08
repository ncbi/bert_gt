#!/bin/bash

echo 'Downloading the dataset'
curl -L -o data.tgz https://github.com/freesunshine0316/nary-grn/blob/master/peng_data/data.tgz?raw=true
echo 'Unzip the dataset'
mkdir -p datasets/nary/origin
tar zxvf data.tgz --directory datasets/nary/origin
rm data.tgz
echo 'Converting the dataset into BERT-GT input format'
python src/dataset_format_converter/convert_nary_2_bert_gt.py