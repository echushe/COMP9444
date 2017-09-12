"""
All tensorflow objects, if not otherwise specified, should be explicity
created with tf.float32 datatypes. Not specifying this datatype for variables and
placeholders will cause your code to fail some tests.

For the specified functionality in this assignment, there are generally high
level Tensorflow library calls that can be used. As we are assessing tensorflow,
functionality that is technically correct but implemented manually, using a
library such as numpy, will fail tests. If you find yourself writing 50+ line
methods, it may be a good idea to look for a simpler solution.

Along with the provided functional prototypes, there is another file,
"train.py" which calls the functions listed in this file. It trains the
specified network on the MNIST dataset, and then optimizes the loss using a
standard gradient decent optimizer. You can run this code to check the models
you create.

"""

import tensorflow as tf

def input_placeholder():
    """
    This placeholder serves as the input to the model, and will be populated
    with the raw images, flattened into single row vectors of length 784.

    The number of images to be stored in the placeholder for each minibatch,
    i.e. the minibatch size, may vary during training and testing, so your
    placeholder must allow for a varying number of rows.

    :return: A tensorflow placeholder of type float32 and correct shape
    """
    return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")

def target_placeholder():
    """
    This placeholder serves as the output for the model, and will be
    populated with targets for training, and testing. Each output will
    be a single one-hot row vector, of length equal to the number of
    classes to be classified (hint: there's one class for each digit)

    The number of target rows to be stored in the placeholder for each
    minibatch, i.e. the minibatch size, may vary during training and
    testing, so your placeholder must allow for a varying number of
    rows.

    :return: A tensorflow placeholder of type float32 and correct shape
    """
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")

def onelayer(X, Y, layersize=10):
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    
    # Initialize weights and biases
    w = tf.Variable(tf.zeros([784, layersize]))
    b = tf.Variable(tf.zeros([layersize]))
    
    # Dot product (linear regression)
    the_logits = tf.matmul(X, w) + b
    # Softmax activation function
    the_preds = tf.nn.softmax(the_logits)
    
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = the_logits)
    batch_loss = tf.reduce_mean(batch_xentropy)
    
    return w, b, the_logits, the_preds, batch_xentropy, batch_loss

def twolayer(X, Y, hiddensize=30, outputsize=10):
    """
    Create a Tensorflow model for a Neural Network with one hidden layer

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    
    # Initialize weights and biases
    w1 = tf.Variable(tf.truncated_normal([784, hiddensize]))
    b1 = tf.Variable(tf.truncated_normal([hiddensize]))

    # Dot product
    the_logits1 = tf.matmul(X, w1) + b1
    # relu is the activation function of the first hidden layer
    the_preds1 = tf.nn.relu(the_logits1)
    
    # Weights and biases of the second layer
    w2 = tf.Variable(tf.zeros([hiddensize, outputsize]))
    b2 = tf.Variable(tf.zeros([outputsize]))
    
    # Dot product of the second layer
    # Result of the first layer as input of the seconde layer
    the_logits2 = tf.matmul(the_preds1, w2) + b2
    the_preds2 = tf.nn.softmax(the_logits2)
    
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = the_logits2)
    batch_loss = tf.reduce_mean(batch_xentropy)
    
    return w1, b1, w2, b2, the_logits2, the_preds2, batch_xentropy, batch_loss
    

def convnet(X, Y, convlayer_sizes=[10, 10], \
        filter_shape=[3, 3], outputsize=10, padding="same"):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    hints:
    1) consider tf.layer.conv2d
    2) the final layer is very similar to the onelayer network. Only the input
    will be from the conv2 layer. If you reshape the conv2 output using tf.reshape,
    you should be able to call onelayer() to get the final layer of your network
    """
    
    # The first layer of conv
    conv_shape1 = [filter_shape[0], filter_shape[1], 1, convlayer_sizes[0]]
    W_conv1 = tf.Variable(tf.truncated_normal(conv_shape1, stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[convlayer_sizes[0]]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    
    # The second layer of conv
    conv_shape2 = [filter_shape[0], filter_shape[1], convlayer_sizes[0], convlayer_sizes[1]]
    W_conv2 = tf.Variable(tf.truncated_normal(conv_shape2, stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[convlayer_sizes[1]]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    
    # print(h_conv2.get_shape())
    # Flatten output data of the second layer
    X_flat = tf.reshape(h_conv2, [-1, 28 * 28 * convlayer_sizes[1]])
    
    # Initialize weights and biases for the last layer
    w = tf.Variable(tf.zeros([28 * 28 * convlayer_sizes[1], outputsize]))
    b = tf.Variable(tf.zeros([outputsize]))
    
    # Dot product
    the_logits = tf.matmul(X_flat, w) + b
    # Softmax activation function
    the_preds = tf.nn.softmax(the_logits)
    
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = the_logits)
    batch_loss = tf.reduce_mean(batch_xentropy)
    
    
    return h_conv1, h_conv2, w, b, the_logits, the_preds, batch_xentropy, batch_loss
    

def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.

    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary
