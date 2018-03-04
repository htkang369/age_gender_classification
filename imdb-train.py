
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import os
import pprint
from six.moves import cPickle as pickle

image_size = 128
num_channels = 1
num_classes = 2
num_labels = 4


# In[2]:

def load_dataset():
    pickle_file = 'imdb.pkl'
    try:
        f = open(os.getcwd()+"/pkl_folder/"+pickle_file, 'rb')
        dataset = pickle.load(f)
        train_dataset = dataset['train_dataset']
        train_labels = dataset['train_labels']
        valid_dataset = dataset['valid_dataset']
        valid_labels = dataset['valid_labels']
        test_dataset = dataset['test_dataset']
        test_labels = dataset['test_labels']

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
    #     pprint.pprint(dataset)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_dataset()
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# In[3]:

train_labels[0]


# In[4]:

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


# * conv2d+ReLU - 64x64x16
# * maxpooling - 32x32x16
# * conv2d+ReLU - 32x32x16
# * maxpooling - 16x16x16
# * FC_age + ReLU - 4096x64
# * FC_age +softmax - 64x4
# * FC_gender + ReLU - 4096x64
# * FC_gender +softmax - 64x4
# 

# In[14]:

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_label = 4

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
    # depth: so filter
    # 64x64x16

    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    # 32x32x16

    layer3_age_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    # 16x16x16x64
    # 4096x64
    
    layer3_gender_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_gender_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    # 16x16x16x64
    # 4096x64

    layer4_age_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_label], stddev=0.1))
    layer4_age_biases = tf.Variable(tf.constant(1.0, shape=[num_label]))
    # 64x4
    
    layer4_gender_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_label], stddev=0.1))
    layer4_gender_biases = tf.Variable(tf.constant(1.0, shape=[num_label]))
    # 64x4

    # Model.
    def model(data):
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        bias1 = tf.nn.relu(conv1 + layer1_biases)
        pool1 = tf.nn.max_pool(bias1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        bias2 = tf.nn.relu(conv2 + layer2_biases)
        pool2 = tf.nn.max_pool(bias2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shape = pool2.get_shape().as_list()

        reshape_age = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden_age = tf.nn.relu(tf.matmul(reshape_age, layer3_age_weights) + layer3_biases)
        age_logits = tf.matmul(hidden_age, layer4_age_weights) + layer4_age_biases

        reshape_gender = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden_gender = tf.nn.relu(tf.matmul(reshape_gender, layer3_gender_weights) + layer3_biases)
        gender_logits = tf.matmul(hidden_gender, layer4_gender_weights) + layer4_gender_biases

        return age_logits, gender_logits

    # Training computation.
    age_logits, gender_logits = model(tf_train_dataset)
    
    age_labels = tf.reshape(
        tf.slice(tf_train_labels, [0,0,0], [batch_size, 1, 4]), 
        [batch_size, 4])
    loss_age = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=age_logits, labels=age_labels))

    gender_labels = tf.reshape(
        tf.slice(tf_train_labels, [0,1,0], [batch_size, 1, 4]),
        [batch_size, 4])
    loss_gender = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=gender_logits, labels=gender_labels))
                             
    joint_loss = loss_age + loss_gender

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(joint_loss)

    # Predictions for the training, validation, and test data.
    train_prediction_age = tf.nn.softmax(age_logits)
    train_prediction_gender = tf.nn.softmax(gender_logits)
    
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :, :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction_age, train_prediction_gender], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))

    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# In[ ]:



