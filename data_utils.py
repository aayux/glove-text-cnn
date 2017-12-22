import sys
import os
import pickle

import numpy as np

"""
Utility functions for handling dataset, embeddings and batches
"""

def convert_file(filepath, word_dict):
    with open(filepath) as ifile:
        return [word_dict.get(w, 0) for w in ifile.read().split(' ')]


def discover_dataset(path, wdict):
    dataset = []
    for root, _, files in os.walk(path):
        for sfile in [f for f in files if '.txt' in f]:
            filepath = os.path.join(root, sfile)
            dataset.append(convert_file(filepath, wdict))
    return dataset


def pad_dataset(dataset, maxlen):
    return np.array(
        [np.pad(r, (0, maxlen-len(r)), mode='constant') if len(r) < maxlen else np.array(r[:maxlen])
         for r in dataset])


# Class for dataset related operations
class IMDBDataset():
    def __init__(self, path, dict_path, maxlen=128):
        pos_path = os.path.join(path, 'pos')
        neg_path = os.path.join(path, 'neg')

        with open(dict_path, 'rb') as dfile:
            wdict = pickle.load(dfile)

        self.pos_dataset = pad_dataset(discover_dataset(pos_path, wdict), maxlen).astype('i')
        self.neg_dataset = pad_dataset(discover_dataset(neg_path, wdict), maxlen).astype('i')

    def __len__(self):
        return len(self.pos_dataset) + len(self.neg_dataset)

    def get_example(self, i):
        is_neg = i >= len(self.pos_dataset)
        dataset = self.neg_dataset if is_neg else self.pos_dataset
        idx = i - len(self.pos_dataset) if is_neg else i
        label = [0, 1] if is_neg else [1, 0]
        
        print (type(dataset[idx]))
        return (dataset[idx], np.array(label, dtype=np.int32))
    
    def load(self):
        
        dataset = np.concatenate((self.pos_dataset, self.neg_dataset))
        labels = []
        
        for idx in range (0, len(self.pos_dataset)):
            labels.append([1, 0])
        
        for idx in range (0, len(self.neg_dataset)):
            labels.append([0, 1])
        
        return dataset, np.array(labels, dtype=np.int32)


# Function for handling word embeddings
def load_embeddings(path, size, dimensions):
    
    embedding_matrix = np.zeros((size, dimensions), dtype=np.float32)

    # As embedding matrix could be quite big we 'stream' it into output file
    # chunk by chunk. One chunk shape could be [size // 10, dimensions].
    # So to load whole matrix we read the file until it's exhausted.
    size = os.stat(path).st_size
    with open(path, 'rb') as ifile:
        pos = 0
        idx = 0
        while pos < size:
            chunk = np.load(ifile)
            chunk_size = chunk.shape[0]
            embedding_matrix[idx:idx+chunk_size, :] = chunk
            idx += chunk_size
            pos = ifile.tell()
    return embedding_matrix

# Function for creating batches
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    print ("Generating batch iterator ...")
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

