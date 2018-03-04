# this util should contains memory model
# should contain env wapper# this util should contains memory model
# should contain env wapper
import numpy as np
import copy
import tensorflow as tf

def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p))
    return loss

def sigmoid_reward(x):

    return  1/(1+np.exp(-x))