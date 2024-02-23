# BERT-GT

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fncbi%2Fbert_gt&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

-----

This repository provides source codes of our paper [BERT-GT: Cross-sentence n-ary relation extraction with BERT and Graph Transformer] and the BERT-GT implementation of our paper [BioRED: A Rich Biomedical Relation Extraction Dataset].

## Environment:

* GPU: NVIDIA Tesla V100 SXM2
* Anaconda: Anaconda3
* Python: python3.6
* Tensorflow: tensorflow-gpu==2

## Set up:

```
pip install -r requirements.txt
```

We assume that you use the Anaconda envrionment, thus the above command will use "pip" to install tensorflow-gpu the packages.

## Step 1:

Because the process of generating the input datasets is trivial (have to install the scispacy), we provide our dataset_format_convert.py, and you can use it to convert the CDR, n-ary, and BioRED datasets into the input datasets of BERT-GT.

### Download and generate the CDR dataset for BERT-GT
```
bash build_cdr_dataset.sh
```

### Download and generate the n-ary dataset for BERT-GT
```
bash build_nary_dataset.sh
```

### Download and generate the BioRED dataset for BERT-GT
```
bash build_biored_dataset.sh
```

## Step 2:

BERT-GT used [Biobert](https://github.com/dmis-lab/biobert)'s [pre-trained model](https://drive.google.com/file/d/1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD/view?usp=sharing) because they support longer text (with 512 sequence length).

After download the model, please use the below command to unzip it.

```
tar -xvzf biobert_v1.1_pubmed.tar.gz
```

## Step 3a: Running on the CDR dataset

```
bash run_cdr_exp.sh <CUDA_VISIBLE_DEVICES>
```

Please replace the above <CUDA_VISIBLE_DEVICES> with your GPUs' IDs. Eg: '0,1' for GPU devices 0 and 1.

For example

```
bash run_cdr_exp.sh 0,1
```

## Step 3b: Running on the n-ary dataset

```
bash run_nary_exp.sh <CUDA_VISIBLE_DEVICES> <TASK_NAME> <INPUT_DATASET_DIR>
```

Please replace the above <CUDA_VISIBLE_DEVICES> with your GPUs' IDs. Eg: '0,1' for GPU devices 0 and 1.

Please replace the above <TASK_NAME> with one of "nary_dgv_bin", "nary_dgv_mul", "nary_dv_bin", or "nary_dv_mul".
The above "dgv" and "dv" mean DRUG-GENE-MUTATION and DRUG-MUTATION, respectively;
Similarly, the above "bin" and "mul" mean two classes and multiple classes, respectively.

Please replace the above <INPUT_DATASET_DIR> with either "datasets/nary/processed/all" or "datasets/nary/processed/only_single_sent".
The "datasets/nary/processed/all" includes intra-sentence and inter-sentence instances.
The "datasets/nary/processed/only_single_sent" only includes intra-sentence instances.

For example

```
bash run_nary_exp.sh 0,1 nary_dgv_bin "datasets/nary/processed/all"
```

## Step 3c: Running on the BioRED dataset

```
bash run_biored_exp.sh <CUDA_VISIBLE_DEVICES>
```

For example

```
bash run_biored_exp.sh 0,1
```

## Citing BERT-GT and BioRED

* Lai P. T. and Lu Z. BERT-GT: Cross-sentence n-ary relation extraction with BERT and Graph Transformer.
Bioinformatics. 2021.

```
@article{lai2021bertgt,
  author    = {Po-Ting Lai and Zhiyong Lu},
  title     = {BERT-GT: Cross-sentence n-ary relation extraction with BERT and Graph Transformer},
  journal   = {Bioinformatics},
  year      = {2021},
  publisher = {Oxford University Press}
}
```


* Luo L., Lai P. T., Wei C. H., Arighi C. N. and Lu Z. BioRED: A Rich Biomedical Relation Extraction Dataset.
Briefing in Bioinformatics. 2022.
```
@article{luo2022biored,
  author    = {Luo, Ling and Lai, Po-Ting and Wei, Chih-Hsuan and Arighi, Cecilia N and Lu, Zhiyong},
  title     = {BioRED: A Rich Biomedical Relation Extraction Dataset},
  journal   = {Briefing in Bioinformatics},
  year      = {2022},
  publisher = {Oxford University Press}
}
```

## Acknowledgments

We are grateful to the authors of AGGCN, BERT, BioBERT, GS LSTM, and NCBI BlueBERT to make the data and codes publicly available. We would like to thank Dr. Zhijiang Guo for helping us to reproduce the results of AGGCN on the n-ary dataset. We thank Dr. Chih-Husan Wei for his assistance on revising the manuscript. 

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI. The information produced
on this website is not intended for direct diagnostic use or medical decision-making without review and oversight
by a clinical professional. Individuals should not change their health behavior solely on the basis of information
produced on this website. NIH does not independently verify the validity or utility of the information produced
by this tool. If you have questions about the information produced on this website, please see a health care
professional. More information about NCBI's disclaimer policy is available.
