Domain-adaptive Message Passing Graph Neural Network (DM_GNN)
====
This repository contains the author's implementation in Tensorflow for the paper "Domain-adaptive Message Passing Graph Neural Network".

Environment Requirement
===
The code has been tested running under Python 3.7.11. The required packages are as follows:

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

Plese cite our paper as:
===
Xiao Shen, Shirui Pan, Kup-Sze Choi, Xi Zhou. Domain-adaptive Message Passing Graph Neural Network. Neural Networks, 2023.

Pytorch Implementation of DM_GNN can be found at:
===
https://github.com/shenxiaocam/DM_GNN
