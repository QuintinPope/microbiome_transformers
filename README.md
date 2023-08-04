# microbiome_transformers

This repository provides the model architecture and training code for the paper "Learning a deep language model for microbiomes: the power of large scale unlabeled microbiome data".

## Environment Setup
\>=Python 3.5

Required Python packages
* Pytorch
* Scikit-learn
* Huggingface Transformers

## Running Code

The folders pretrain_generator, pretrain_discriminator, and finetune_discriminator each represent independent functionality for pretraining / finetuning models. Each can be used individually, and each has its own README documentation explaining how to do so.
Start by using pretrain_generator to create a series of pretrained generator checkpoints. Then, use pretrain_discriminator to pretrain a discriminator against those checkpoints. Finally, use finetune_discriminator to adapt the pretrained discrimnator to downstream tasks.

### Pretrain Generator

This folder provides code for pretraining a generator on microbiome data (which can then be used as part of the discriminator's pretraining process).

### Pretrain Discriminator

This folder provides code for pretraining a discriminator on specified pretrained generators. More specifically, it trains the discriminator on each specified generator for a set number of epochs.

### Finetune Discriminator

This folder provides code for finetuning a discriminator on a binary classification task
