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
    def __init__(
        self, phrase_length, context_length, num_classes, word_vocab_size,
        postag_vocab_size, embedding_size, filter_sizes, num_filters,
        l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, phrase_length, 2], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, context_length, 2], name="input_x1")
        self.input_y = tf.placeholder("float", [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'):
            W1 = tf.Variable(
                tf.random_uniform([word_vocab_size, embedding_size], -1.0, 1.0),
                name="W1")
            W2 = tf.Variable(
                tf.random_uniform([postag_vocab_size, embedding_size], -1.0, 1.0),
                name="W2")
            with tf.name_scope("phrase-embedding"):
                word_x = tf.slice(self.input_x1, [0, 0, 0], [-1, -1, 1])
                word_x = tf.reshape(word_x, [-1, phrase_length])
                postag_x = tf.slice(self.input_x1, [0, 0, 1], [-1, -1, 1])
                postag_x = tf.reshape(postag_x, [-1, phrase_length])
                self.word_embedded1 = tf.nn.embedding_lookup(W1, word_x)
                self.postag_embedded1 = tf.nn.embedding_lookup(W2, postag_x)
                self.word_embedded_expanded1 = tf.expand_dims(self.word_embedded1, -1)
                self.postag_embedded_expanded1 = tf.expand_dims(self.postag_embedded1, -1)
                self.embedded_expanded1 = tf.concat(3, [self.word_embedded_expanded1, self.postag_embedded_expanded1])

            with tf.name_scope("context-embedding"):
                word_x = tf.slice(self.input_x2, [0, 0, 0], [-1, -1, 1])
                word_x = tf.reshape(word_x, [-1, context_length])
                postag_x = tf.slice(self.input_x2, [0, 0, 1], [-1, -1, 1])
                postag_x = tf.reshape(postag_x, [-1, context_length])
                self.word_embedded2 = tf.nn.embedding_lookup(W1, word_x)
                self.postag_embedded2 = tf.nn.embedding_lookup(W2, postag_x)
                self.word_embedded_expanded2 = tf.expand_dims(self.word_embedded2, -1)
                self.postag_embedded_expanded2 = tf.expand_dims(self.postag_embedded2, -1)
                self.embedded_expanded2 = tf.concat(3, [self.word_embedded_expanded2, self.postag_embedded_expanded2])

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs1 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("phrase-conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 2, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_expanded1,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, phrase_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs1.append(pooled)

        pooled_outputs2 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("context-conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 2, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_expanded2,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, context_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs2.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool1 = tf.concat(3, pooled_outputs1)
        self.h_pool_flat1 = tf.reshape(self.h_pool1, [-1, num_filters_total])
        self.h_pool2 = tf.concat(3, pooled_outputs2)
        self.h_pool_flat2 = tf.reshape(self.h_pool2, [-1, num_filters_total])

        self.joint_feature = tf.concat(1, [self.h_pool_flat1, self.h_pool_flat2])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.joint_feature, self.dropout_keep_prob)
            #  self.h_drop = tf.nn.dropout(self.h_pool_flat1, self.dropout_keep_prob)
            #  self.h_drop = tf.nn.dropout(self.h_pool_flat2, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[2*num_filters_total, num_classes],
                #  shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

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
