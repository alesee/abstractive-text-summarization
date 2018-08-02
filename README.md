# abstractive-text-summarization

This repository/[notebook](/abstractive-text-summ.ipynb) contains code for in-progress implementation of the [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://arxiv.org/abs/1602.06023) paper.

Requirements
---
1. Create conda environment 

`conda env create -f environment.yml`  --gpu

`conda env create -f environment-cpu.yml`  --cpu

2. Install dependencies (PyTorch, Fastai, etc) via:

`pip install -r requirements.txt`

3. Download `spacy` english module

`python -m spacy download en`

Dataset
--

The dataset used is a subset of the gigaword dataset and can be found [here](https://drive.google.com/file/d/0B6N7tANPyVeBNmlSX19Ld2xDU1E/view?usp=sharing).

It contains 3,803,955 parallel source & target examples for training and 189,649 examples for validation.

After downloading, we created article-title pairs, saved in tabular datset format (.csv) and extracted a sample subset (80,000 for training & 20,000 for validation). This data preparation can be found [here](/data-preparation.ipynb).

An example article-title pair looks like this:

`source: the algerian cabinet chaired by president abdelaziz bouteflika on sunday adopted the #### finance bill predicated on an oil price of ## dollars a barrel and a growth rate of #.# percent , it was announced here .`

`target: algeria adopts #### finance bill with oil put at ## dollars a barrel`


Experimenting on the complete dataset (3M) would take a really long time (also $$$$). So in order to train and experiment faster we use our sample subset of 80,000. 

Current Features
--
* model architecture supports LSTM & GRU (biLSTM-to-uniLSTM or biGRU-to-uniGRU)
* implements batch data processing 
* implements attention mechanism ([Bahdanau et al.](https://arxiv.org/abs/1409.0473) & [Luong et al.(global dot)](https://arxiv.org/abs/1508.04025))
* implements [scheduled sampling (teacher forcing)](https://arxiv.org/abs/1506.03099)
* implements [tied embeddings](https://arxiv.org/pdf/1608.05859.pdf)
* initializes encoder-decoder with pretrained vectors (glove.6B.200d)
* implements custom training callbacks (tensorboard visualization for PyTorch, save best model & log checkpoint)
* implements attention plots


To-Do
---
* Implement additional linguistic features embeddings  
* Implement generator-pointer switch and replace unknown words by selecting source token with the highest attention score.
* Implement large vocabulary trick 
* Implement sentence level attention 
* Implement beam search during inference
* implement rouge evaluation
