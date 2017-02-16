#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/06/02 15:03:24

@author: Changzhi Sun
"""

import sklearn as sk
import tensorflow as tf
import numpy as np
from text_cnn import TextCNN


class CNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,
                 lengths,
                 num_classes,
                 word_vocab_size,
                 postag_vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input0 = tf.placeholder(tf.int32, [None, lengths[0]], name="input0")
        self.input1 = tf.placeholder(tf.int32, [None, lengths[1]], name="input1")
        self.input2 = tf.placeholder(tf.int32, [None, lengths[2]], name="input2")
        self.input3 = tf.placeholder(tf.int32, [None, lengths[3]], name="input3")
        self.input4 = tf.placeholder(tf.int32, [None, lengths[4]], name="input4")
        self.input5 = tf.placeholder(tf.int32, [None, lengths[5]], name="input5")
        self.input_y = tf.placeholder("float", [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.inputs = [self.input0, self.input1, self.input2,
                       self.input3, self.input4, self.input5]

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        self.embedded_expanded = []
        # Embedding layer
        with tf.device('/cpu:0'):
            W1 = tf.Variable(
                tf.random_uniform([word_vocab_size, embedding_size], -1.0, 1.0),
                name="W1")
            W2 = tf.Variable(
                tf.random_uniform([postag_vocab_size, embedding_size], -1.0, 1.0),
                name="W2")
            for i in range(0, len(lengths), 2):
                with tf.variable_scope("embedding-%d" % i):
                    word_embedded = tf.nn.embedding_lookup(W1, self.inputs[i])
                    postag_embedded = tf.nn.embedding_lookup(W2, self.inputs[i+1])
                    word_embedded_expanded = tf.expand_dims(word_embedded, -1)
                    postag_embedded_expanded = tf.expand_dims(postag_embedded, -1)
                    self.embedded_expanded.append(tf.concat(2,
                        [word_embedded_expanded, postag_embedded_expanded]))


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        W = []
        b = []
        conv = []
        h = []
        pooled = []
        for k in range(len(self.embedded_expanded)):
            pooled_outputs.append([])
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("phrase-conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, 2*embedding_size, 1, num_filters]
                for k in range(len(self.embedded_expanded)):
                    W.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W-%d" % k))
                    b.append(tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b-%d" % k))
                    conv = tf.nn.conv2d(
                        self.embedded_expanded[k],
                        W[k],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv-%d" % k)
                    # Apply nonlinearity
                    h.append(tf.nn.relu(tf.nn.bias_add(conv, b[k]), name="relu-%d" % k))
                    # Maxpooling over the outputs
                    pooled.append(tf.nn.max_pool(
                        h[k],
                        ksize=[1, lengths[2*k] - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool-%d" % k))
                    pooled_outputs[k].append(pooled[k])

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = []
        self.h_pool_flat = []
        for i in range(len(pooled_outputs)):
            self.h_pool.append(tf.concat(3, pooled_outputs[i]))
            self.h_pool_flat.append(tf.reshape(self.h_pool[i], [-1, num_filters_total]))

        self.joint_feature = tf.concat(1, self.h_pool_flat)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.joint_feature, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[3*num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

            self.scores_proba = tf.reshape(tf.slice(tf.nn.softmax(self.scores), [0, 1], [-1, 1]), [-1])
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.pred_ = tf.cast(tf.greater(self.scores_proba, 0.8), tf.int64)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # pred
        with tf.name_scope("pred"):
            self.pred = self.predictions

        # Accuracy
        with tf.name_scope("accuracy"):
            self.y = tf.argmax(self.input_y, 1)
            self.TP = tf.reduce_sum(tf.cast(tf.cast(self.y, "bool") & tf.cast(self.predictions, "bool"), "float"))
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Precision
        with tf.name_scope("precision"):
            self.precision = self.TP / tf.cast(tf.reduce_sum(self.predictions), 'float')

        # Recall
        with tf.name_scope("recall"):
            self.recall = self.TP / tf.cast(tf.reduce_sum(self.y), 'float')

        # F1 score
        with tf.name_scope("f1_score"):
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
