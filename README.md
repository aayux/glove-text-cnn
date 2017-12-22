# Text Classification with Sentence Level Convolutional Neural Networks 

### About Model

A Deep Convolutional Neural Network architecture based on CNN for Text Classification<sup>[1]</sup> with pretrained GloVe embeddings.

### How to run
- Download the [IMdB Movie Reviews](http://ai.stanford.edu/~amaas/data/sentiment/) and [GloVe](https://nlp.stanford.edu/projects/glove/) datasets.
- Generate embeddings using: 

`python3 embeddings.py -d data/glove.42B.300d.txt --npy_output data/embeddings.npy --dict_output data/vocab.pckl --dict_whitelist data/aclImdb/imdb.vocab`

- Start training with `python3 train.py`

### Sources

[1]  TextCNN: [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)

[2]  Data Helpers: [rampage644/qrnn](https://github.com/rampage644/qrnn)

