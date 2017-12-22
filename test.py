import os
import time
import datetime

import tensorflow as tf
import numpy as np
import data_utils as utils

from tensorflow.contrib import learn
from text_cnn import TextCNN
from data_utils import IMDBDataset

print ("Intialising test parameters ...")

batch_size = 64
# Checkpoint directory from training run
checkpoint_dir = "./runs/.../checkpoints"
# Evaluate on all training data
eval_train = False

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

print ("Loading test data ...")
dataset = IMDBDataset('./data/aclImdb/test', './data/vocab.pckl')
# NOTE: Fetch raw text
x_test, y_test = dataset.load()
print ("Dataset loaded. Starting evaluation ...")

# Evaulation
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = utils.batch_iter(list(x_test), batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    y_test = [col[1] for col in y_test]
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))