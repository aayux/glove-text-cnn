# Text Classification with Sentence Level Convolutional Neural Networks 

### About Model

A Deep Convolutional Neural Network architecture based on CNN for Text Classification<sup>[1]</sup> with pretrained GloVe embeddings.

### How to run
- Download the [IMdB Movie Reviews](http://ai.stanford.edu/~amaas/data/sentiment/) and [GloVe](https://nlp.stanford.edu/projects/glove/) datasets.
- Generate embeddings using: 

`python3 generate_embeddings.py -d data/glove.42B.300d.txt --npy_output data/embeddings.npy --dict_output data/vocab.pckl --dict_whitelist data/aclImdb/imdb.vocab`

- Start training with `python3 train.py`

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 300)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 128)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 500)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 1000)
  --num_checkpoints NUM_CHECKPOINTS
                        Number of checkpoints to store (default: 3)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement
  ```

### Sources

[1]  TextCNN: [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)

[2]  Data Helpers: [rampage644/qrnn](https://github.com/rampage644/qrnn)

