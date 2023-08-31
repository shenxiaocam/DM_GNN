Domain-adaptive Message Passing Graph Neural Network (DM_GNN)
====
This repository contains the author's implementation in Tensorflow for the paper "Domain-adaptive Message Passing Graph Neural Network".

Environment Requirement
===
The code has been tested running under the required packages as follows:

python == 3.7.11

tensorflow == 1.13.2

numpy == 1.21.5

scipy == 1.7.3

sklearn == 0.23.2


Datasets
===
input/ contains the 7 datasets used in our paper.

Each ".mat" file stores a network dataset, where

the variable "network" represents an adjacency matrix, 

the variable "attrb" represents a node attribute matrix,

the variable "group" represents a node label matrix. 

Code
===
"DM_GNN_model.py" is the implementation of the DM_GNN model.

"test_DM_GNN_Blog.py" is an example case of the cross-network node classification task from Blog2 to Blog1 networks.

"test_DM_GNN_citation.py" is an example case of the cross-network node classification task from citationv1 to dblpv7 networks.

"test_DM_GNN_squri.py" is an example case of the cross-network node classification task from squirrel1 to squirrel2 networks.

Please cite our paper as:
===
Shen, Xiao, et al. "Domain-adaptive message passing graph neural network." Neural Networks 164 (2023): 439-454. 

Paper: 
===
"NN-DM-GNN.pdf" is the PDF version of DM_GNN paper.


