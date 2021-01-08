# BERT-GT

-----

This repository provides source codes of our paper [BERT-GT: Cross-sentence n-ary relation extraction with BERT and Graph Transformer].

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

Because the process of generating the input datasets is trivial (have to install the scispacy), we provide our dataset_format_convert.py, and you can use it to convert either the CDR dataset or n-ary dataset into the input datasets of BERT-GT.

### Download and generate the CDR dataset for BERT-GT
```
bash build_cdr_dataset.sh
```

### Download and generate the n-ary dataset for BERT-GT
```
bash build_nary_dataset.sh
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

## Citing BERT-GT

* Lai P and Lu Z. BERT-GT: Cross-sentence n-ary relation extraction with BERT and Graph Transformer.
Bioinformatics.

```
@article{lai2020bertgt,
  author    = {Po-Ting Lai and Zhiyong Lu},
  title     = {BERT-GT: Cross-sentence n-ary relation extraction with BERT and Graph Transformer},
  journal   = {Bioinformatics},
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
