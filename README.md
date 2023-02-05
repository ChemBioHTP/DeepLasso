# DeepLasso
## About the project
In collaboration with the Link lab at Princeton, we developed a deep learning model to predict  antimicrobial activity from the sequence of lasso peptide ubonodin. Please cite: "A High-Throughput Screen Reveals the structure-Activity Relationship of the Antimicrobial Lasso Peptide Ubonodin". 

## Getting started 
### Prerequistites
Install [Pytorch =1.8.1], [scikit-learn=1.2.], [numpy], [pandas]

## Dataset

The dataset for training is under the folder /input_31th. The dataset cotains:

- mutants: The mutation annotation of the original data
- proteins: The lasso peptide sequences used in the embedding
- sequence_dict: The sequence dictionary used in the lasso peptide sequence embedding
- regression: The experimentally measured enrichment values for ubonodin 
- ssts: The secondary topology labels used in topology embedding
- topology_dict: The secondary dictionary use in the lasso peptide topology embedding

## Descriptions of folders and files in DeepLasso

* **inference** contains the inference example with raw data files and the data-preprocessing scripts.
* **layers** contains the multi-head attention layers used in the model
* **output** contains the training log files and the output of test set files in each training epoch
* **params_trained** contains the pre-trained parameters that can be used by the inference.py
* **secondary_structures_version** contains the model architectures embedding the secondary topology of ubonodin (ring, loop, tail)

## Step-by-step operation:

## Training
- First, run `cd deeplasso`, to get inside the folder, and run `python preprocess.py` to convert the data into word embedding.

- Second, run `python train.py`. This will display training and testing results on the screen. You can also find your training and testing log file under the /output folder.

- If you want to set up hyperparameters for your model, go to the train.main() to set the hyperparams

## Running the inference
   Run `python inference.py > inference.output`

Notably, the preprocessing "preprocess.py" and trainer "Trainer.train" code were modified from DLKcat (https://github.com/SysBioChalmers/DLKcat). Please cite: Li, F., Yuan, L., Lu, H. et al. Deep learning-based kcat prediction enables improved enzyme-constrained model reconstruction. Nat Catal 5, 662–672 (2022). https://doi.org/10.1038/s41929-022-00798-z

## Developer of the DeepLasso code
Xinchun (Shone) Ran
Graduate Research Assistant
Yang Lab
Department of Chemistry
Vanderbilt University
