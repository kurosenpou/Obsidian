---
language: en
tags:
- transformers
- feature-extraction
- materials
license: other
---

# MaterialsBERT

This model is a fine-tuned version of [PubMedBERT model](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) on a dataset of 2.4 million materials science abstracts.
It was introduced in [this](https://www.nature.com/articles/s41524-023-01003-w) paper. This model is uncased.

## Model description

Domain-specific fine-tuning has been [shown](https://arxiv.org/abs/2007.15779) to improve performance in downstream performance on a variety of NLP tasks. MaterialsBERT fine-tunes PubMedBERT, a pre-trained language model trained using biomedical literature. This model was chosen as the biomedical domain is close to the materials science domain. MaterialsBERT when further fine-tuned on a variety of downstream sequence labeling tasks in materials science, outperformed other baseline language models tested on three out of five datasets.

## Intended uses & limitations

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to
be fine-tuned on materials-science relevant downstream tasks.

Note that this model is primarily aimed at being fine-tuned on tasks that use a sentence or a paragraph (potentially masked)
to make decisions, such as sequence classification, token classification or question answering.


## How to Use

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import BertForMaskedLM, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('pranav-s/MaterialsBERT')
model = BertForMaskedLM.from_pretrained('pranav-s/MaterialsBERT')
text = "Enter any text you like"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

## Training data

A fine-tuning corpus of 2.4 million materials science abstracts was used. The DOI's of the journal articles used are provided in the file training_DOI.txt

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0
- mixed_precision_training: Native AMP


### Framework versions

- Transformers 4.17.0
- Pytorch 1.10.2
- Datasets 1.18.3
- Tokenizers 0.11.0


## Citation

If you find MaterialsBERT useful in your research, please cite the following paper:

```latex
@article{materialsbert,
  title={A general-purpose material property data extraction pipeline from large polymer corpora using natural language processing},
  author={Shetty, Pranav and Rajan, Arunkumar Chitteth and Kuenneth, Chris and Gupta, Sonakshi and Panchumarti, Lakshmi Prerana and Holm, Lauren and Zhang, Chao and Ramprasad, Rampi},
  journal={npj Computational Materials},
  volume={9},
  number={1},
  pages={52},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

<a href="https://huggingface.co/exbert/?model=pranav-s/MaterialsBERT">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>
