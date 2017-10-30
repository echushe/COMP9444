"""
You are encouraged to edit this file during development, however your final
model must be trained using the original version of this file. This file
trains the model defined in implementation.py, performs tensorboard logging,
and saves the model to disk every 10000 iterations. It also prints loss
values to stdout every 50 iterations.
"""


import numpy as np
import tensorflow as tf
from random import randint
import datetime
import os

import implementation as imp

batch_size = imp.batch_size
iterations = 100200
seq_length = 40  # Maximum length of sentence

checkpoints_dir = "./checkpoints"

def getTrainBatch(training_data):
    labels = []
    arr = np.zeros([batch_size, seq_length], dtype=np.int32)
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, len(training_data) // 2 - 2001)
            labels.append([1, 0])
        else:
            num = randint(len(training_data) // 2, len(training_data) - 2001)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels
    
def getTestBatch(training_data):
    labels = []
    arr = np.zeros([batch_size, seq_length], dtype=np.int32)
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(len(training_data) // 2 - 2000, len(training_data) // 2 - 1)
            labels.append([1, 0])
        else:
            num = randint(len(training_data) - 2000, len(training_data) - 1)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels

# Call implementation
glove_array, glove_dict = imp.load_glove_embeddings()
training_data = imp.load_data(glove_dict)

print('Size of training data: ', len(training_data))

input_data, labels, dropout_keep_prob, optimizer, accuracy, loss = imp.define_graph(glove_array)

# tensorboard
train_accuracy_op = tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

# saver
all_saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

str_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_logdir = "tensorboard/" + str_time + "-train/"
test_logdir = "tensorboard/" + str_time + "-test/"

train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
test_writer = tf.summary.FileWriter(test_logdir, sess.graph)

train_index = 0
while train_index < iterations:      
    # get training batch 
    train_batch_data, train_batch_labels = getTrainBatch(training_data)
    # The training
    sess.run(optimizer, {input_data: train_batch_data, labels: train_batch_labels, dropout_keep_prob: 0.75})

    # Print the accuracy of both training set and test set
    if (train_index % 50 == 0):
        # Calculate and add summary of trainng batch to tensorboard
        train_loss_value, train_accuracy_value, train_summary = sess.run(
            [loss, accuracy, summary_op],
            {input_data: train_batch_data,
             labels: train_batch_labels,
             dropout_keep_prob: 0.75})            
        train_writer.add_summary(train_summary, train_index)
        
        # get the test batch
        test_batch_data, test_batch_labels = getTestBatch(training_data)
        # Calculate and add summary of trainng batch to tensorboard
        test_loss_value, test_accuracy_value, test_summary = sess.run(
                [loss, accuracy, summary_op],
                {input_data: test_batch_data,
                 labels: test_batch_labels})
        test_writer.add_summary(test_summary, train_index)
        
        print("Iteration: {0}\t Train acc: {1}\t Test acc: {2}\t".format(train_index, train_accuracy_value, test_accuracy_value))
    
    if (train_index % 10000 == 0 and train_index != 0):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        save_path = all_saver.save(sess, checkpoints_dir +
                                   "/trained_model.ckpt",
                                   global_step=train_index)
        print("Saved model to %s" % save_path)

        
    train_index += 1

sess.close()
