## Objective
### The project is about visualizing features that contribute to a target/decision of ML/DL architecture (Ex: Classification/Regression), which is the main goal in explainable AI field (xAI).
    
## Methods
### Using the Graph Neural Network as architecture in the classification task. For explanation and visualization, we use layer-wise relevance propagation which computes the contribution of features to the final layer or between layers.

## Quickstart
### 1: Run File main.ipynb - The result is at directory data/explanations/explanations.tex
### 2: Dataset Link: https://github.com/Franck-Dernoncourt/pubmed-rct
### 3: Pretrained model fasttext: https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
### 4: Setup SciSpacy language model
```
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_sm-0.2.0.tar.gz
```
