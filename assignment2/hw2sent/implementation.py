import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string
import collections
from tensorflow.contrib import rnn
import heapq
from operator import itemgetter
import math

# Batch size of movie reviews
batch_size = 32
# Only the "top" 40 words in each movie review is available
review_length = 40
# The words in each movie review will be fed into RNN in chronological order
time_steps = review_length
# Nodes of the RNN hidden layer
num_hidden = 256
# The learning_rate
learning_rate = 0.0001

def least_common(array, to_find=None):
    counter = collections.Counter(array)
    if to_find is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))
    
    
def check_file(filename, expected_bytes):
    '''Download a file if not present, and make sure it's the right size.'''
    if not os.path.exists(filename):
        print("please make sure {0} exists in the current directory".format(filename))
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "File {0} didn't have the expected size. Please ensure you have downloaded the assignment files correctly".format(filename))
    return filename


# Read the data into a list of strings.
def extract_data(filename):
    '''Extract data from tarball and store as list of strings'''
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'review_data/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tarball, os.path.join(dir,"review_data/"))
    return

# Read all movie reivews from extracted data
def read_reviews():
    print("READING DATA")
    data = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir, 'review_data/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir, 'review_data/neg/*')))
    
    print("Parsing %s files" % len(file_list))   
    
    reviews = list()
    for f in file_list:
        with open(f, "r", encoding = 'utf-8') as openf:
            review = openf.read()
            # print(review)
            reviews.append(review)

    return reviews


def build_data_row(data_row, review, glove_dict):

    chs = []
    for ch in review:
        if ch in string.punctuation:
            chs.append(' ')
        else:
            chs.append(ch)
    
    l_review = ''.join(chs)
    l_review = l_review.lower()
    words = l_review.split()
    '''
    for word in words:
        print('{0} '.format(word), end='')
    print()
    '''
    i = 0
    for word in words:
        if i >= review_length:
            break
        if word in {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',\
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', \
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', \
        'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', \
        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', \
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', \
        'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', \
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', \
        'between', 'into', 'through', 'during', 'before', 'after', 'above', \
        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', \
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', \
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', \
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', \
        'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', \
        'don', 'should', 'now'}:
            continue

        # If the word exsits in the dictionary, assign index of this word
        # otherwise, assign index of 'UNK'
        data_row[i] = glove_dict.get( word, glove_dict.get('UNK') )
        i += 1

    while i < review_length:
        data_row[i] = glove_dict.get('UNK')
        i += 1
    
    # print(data_row)

    
def build_dataset(reviews, glove_dict):
    '''Process raw inputs into a dataset.'''
    
    data = np.zeros( [len(reviews), review_length], dtype=np.int32 )
    for i in range(len(reviews)):
        build_data_row(data[i], reviews[i], glove_dict) 
       
    return data


def load_data(glove_dict):
    '''
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form'''
    
    if os.path.exists(os.path.join(os.path.dirname(__file__), "data.npy")):
        print("loading saved parsed data, to reparse, delete 'data.npy'")
        data = np.load("data.npy")
        
    else:
        filename = check_file('reviews.tar.gz', 14839260)
        extract_data(filename) # unzip
        
        # Read reviews one by one
        reviews = read_reviews()
        print('Data size', len(reviews))
        
        # 
        data = build_dataset(reviews, glove_dict)

        np.save("data", data)

        del reviews  # Hint to reduce memory.
    
    return data


def load_glove_embeddings():
    '''
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    '''
    
    word_index_dict = dict()
    
    if os.path.exists(os.path.join(os.path.dirname(__file__), "embeddings.npy")):
        print("loading saved parsed embedding dictionary, to reparse, delete 'embeddings.npy'")
        embeddings = np.load("embeddings.npy")
        
    else:
        #data = open("glove.6B.50d.txt",'r',encoding="utf-8")
        #if you are running on the CSE machines, you can load the glove data from here
        #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
        list_embeddings = list()
        
        with open("glove.6B.50d.txt", "r", encoding = 'utf-8') as myfile:
            lines = myfile.readlines()
            
            # The first entry is a vector of zeros (UNK)
            list_embeddings.append([0.0 for i in range(len(lines[0].split()) - 1)])
            
            # First word in the dictionary is 'UNK'
            word_index_dict['UNK'] = 0
            
            for i in range(0, len(lines)):
                values = lines[i].split()
                # First value of each line is the word itself
                # Assign index i to this word in the dictionary
                word_index_dict[values[0]] = i + 1
                
                # Add vector values (float values)
                value_vector = list()
                for j in range(1, len(values)):
                    value_vector.append(float(values[j]))
                
                # append the new row
                list_embeddings.append(value_vector)
        
        # assign embeddings to a 2D numpy array
        embeddings = np.zeros( [ len(list_embeddings), len(list_embeddings[0]) ], dtype=np.float32 )
        
        for i in range(len(list_embeddings)):
            for j in range(len(list_embeddings[i])):
                embeddings[i][j] = list_embeddings[i][j]
                
        np.save("embeddings", embeddings)
    
    return embeddings, word_index_dict


def lstm_rnn_definition(glove_embeddings_arr, dropout_keep_prob):

    input_data = tf.placeholder(dtype=tf.int32, shape=[batch_size, review_length], name="input_data")
    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name="labels")
    
    weights = tf.Variable(tf.random_normal([num_hidden, 2]))
    biases = tf.Variable(tf.random_normal([2])) 
    
    # Get size of dictionary and each word vector
    (dictionary_size, word_vector_size) = glove_embeddings_arr.shape
    
    print('Dictionary size: {0} word_vector_size: {1}'.format(dictionary_size, word_vector_size))
   
    # Lookup the embedding for each batch input
    # Shape of embeds_of_each_batch is
    # [batch_size, review_length, word_vector_size]
    embeds_of_each_batch = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)
    
    # convert into a word sequence of shape [batch_size, word_vector_size]
    # split_inputs = tf.unstack(embeds_of_each_batch, time_steps, axis = 1)

    # 2-layer LSTM with num_hidden units.
    rnn_cell1 = rnn.LSTMCell(num_hidden)
    rnn_cell2 = rnn.LSTMCell(num_hidden)
    
    rnn_cell_list = [rnn_cell1, rnn_cell2]
    
    multi_layer_rnn_cell = rnn.MultiRNNCell(rnn_cell_list)
    multi_layer_rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=multi_layer_rnn_cell, output_keep_prob=dropout_keep_prob)    
    
    # generate prediction
    (outputs, states) = tf.nn.dynamic_rnn(multi_layer_rnn_cell, embeds_of_each_batch, dtype=tf.float32)
    
    output = tf.reduce_mean(outputs, axis=1)

    # there are num_input outputs but
    # we only want the last output
    return (input_data, labels, tf.matmul(output, weights) + biases)


def get_accuracy_definition(preds_op, labels):
    correct_preds_op = tf.equal(tf.argmax(preds_op, 1), tf.argmax(labels, 1))
    # the tf.cast sets True to 1.0, and False to 0.0. With N predictions, of
    # which M are correct, the mean will be M/N, i.e. the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32), name="accuracy")
    return accuracy
    
def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
    
    (input_data, labels, logits) = lstm_rnn_definition(glove_embeddings_arr, dropout_keep_prob)
    
    # Calculate RNN loss here
    loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits), name="loss")
    
    
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)    
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
   
    # Evaluate model
    pred_op = tf.nn.softmax(logits)   
    accuracy = get_accuracy_definition(pred_op, labels)
    
    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
