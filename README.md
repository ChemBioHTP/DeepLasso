# DeepLasso
##About the project
The implementation of the paper "A High-Throughput Screen Reveals the structure-Activity Relationship of the Antimicrobial Lasso Peptide Ubonodin"

##Getting started 
### Prerequistites
Install [Pytorch =1.8.1], [scikit-learn=1.2.], [numpy], [pandas]

## Dataset

The dataset for training process is under the folder input_31th which curated from the large enrichment datasets. In the dataset folder,  it cotains multiple files

- mutants: The mutation flag from the original data
- proteins: The lasso sequence use in the embedding
- sequence_dict: The sequence dictionary use in the lasso sequence embedding
- regression: The experimental values of lasso enrichment 
- ssts: The secondary topology labels used in topology embedding
- topology_dict: The seconday dictionary use in the lasso topology embedding

## Descriptions of folders and files in the deeplasso

* **inference** folder contains the inference example with raw data file and the preprocess scripts.
* **layers** folder contains the multi-head attention layers use in the deeplasso model
* **output** folder contains the training log and the output of test set in the training epoch
* **params_trained** folder contains the pretrained params which can be used by the inference.py
* **secondary_structures_version** folder contains the model architectures embedding the secondary topology of the lassp peptide (ring, loop, tail)


## Step-by-step running:

## Training for provided data
- First, cd deeplasso, to get in the folder. 
  `python preprocess.py`
  Running preprocess script convert the data into word embedding

- Second, run train.py using
  `python train.py`
   Both training and testing result for the dataset provide will display on the screen. And you can find your training and testing log under the output
- If you want to setup your own hyperparameters, go to the train.main() to set the hyperparams

## Running the inference
   `python inference.py > inference.output`

The preprocessing "preprocess.py" and trainer "Trainer.train" code are obtained from DLKcat 

@article{li2022deep,
  title={Deep learning-based k cat prediction enables improved enzyme-constrained model reconstruction},
    author={Li, Feiran and Yuan, Le and Lu, Hongzhong and Li, Gang and Chen, Yu and Engqvist, Martin KM and Kerkhoven, Eduard J and Nielsen, Jens},
    journal={Nature Catalysis},
    volume={5},
    number={8},
    pages={662--672},
    year={2022},
    publisher={Nature Publishing Group UK London}
}
