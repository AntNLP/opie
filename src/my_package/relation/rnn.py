#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/07/08 16:44:07

@author: Changzhi Sun
"""

import sklearn as sk
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell


class RNN(object):
    """
    A RNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,
                 lengths,
                 num_classes,
                 word_vocab_size,
                 postag_vocab_size,
                 embedding_size,
                 n_hidden,
                 l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input0 = tf.placeholder(tf.int32, [None, lengths[0]], name="input_x0")
        self.input1 = tf.placeholder(tf.int32, [None, lengths[1]], name="input_x1")
        self.input2 = tf.placeholder(tf.int32, [None, lengths[2]], name="input_x2")
        self.input3 = tf.placeholder(tf.int32, [None, lengths[3]], name="input_x3")
        self.input4 = tf.placeholder(tf.int32, [None, lengths[4]], name="input_x4")
        self.input5 = tf.placeholder(tf.int32, [None, lengths[5]], name="input_x5")
        self.input6 = tf.placeholder(tf.int32, [None, lengths[6]], name="input_x6")
        self.input7 = tf.placeholder(tf.int32, [None, lengths[7]], name="input_x7")
        self.input8 = tf.placeholder(tf.int32, [None, lengths[8]], name="input_x8")
        self.input9 = tf.placeholder(tf.int32, [None, lengths[9]], name="input_x9")
        self.input10 = tf.placeholder(tf.int32, [None, lengths[10]], name="input_x10")
        self.input11 = tf.placeholder(tf.int32, [None, lengths[11]], name="input_x11")
        self.input12 = tf.placeholder(tf.int32, [None, lengths[12]], name="input_x12")
        self.input13 = tf.placeholder(tf.int32, [None, lengths[13]], name="input_x13")

        self.inputs = [self.input0, self.input1, self.input2, self.input3,
                       self.input4, self.input5, self.input6, self.input7,
                       self.input8, self.input9, self.input10, self.input11,
                       self.input12]

        self.input_y = tf.placeholder("float", [None, num_classes], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.n_hidden = n_hidden

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'):
            W0 = tf.Variable(
                tf.random_uniform([2, embedding_size], -1.0, 1.0),
                name="W0")
            #  W1 = tf.Variable(tf.constant(0.0, shape=[word_vocab_size, embedding_size]),
                             #  trainable=True, name="W1")
            #  word_embedding = W1.assign(self.embedding_placeholder)
            self.word_embedding = tf.Variable(tf.random_uniform([word_vocab_size, embedding_size], -1.0, 1.0), name="W1")
            #  W1 = tf.Variable(
                #  tf.random_uniform([word_vocab_size, embedding_size], -1.0, 1.0),
                #  name="W1")
            W2 = tf.Variable(
                tf.random_uniform([postag_vocab_size, embedding_size], -1.0, 1.0),
                name="W2")
            self.embeded_position = tf.nn.embedding_lookup(W0, self.inputs[0])

            i, k = 1, 1
            self.output = []
            while i < len(lengths) - 1:
                with tf.variable_scope("feature-%d" % k):
                    embeded_position_tile = tf.tile(self.embeded_position,
                        [1, lengths[i], 1])
                    embeded = tf.concat(2, [tf.nn.embedding_lookup(self.word_embedding, self.inputs[i]),
                                            tf.nn.embedding_lookup(W2, self.inputs[i+1]),
                                            embeded_position_tile])
                    seq_len = tf.slice(self.input13, [0, i], [-1, 1])
                    seq_len = tf.reshape(seq_len, [-1])
                    self.output.append(self.BiLSTM(embeded, lengths[i], embedding_size*3, seq_len))
                    #  if k == 1:
                        #  self.embed = seq_len
                i += 2
                k += 1

        # attention
        with tf.name_scope("attention"):
            trans_feature = tf.pack(self.output)
            trans_feature = tf.transpose(trans_feature, [1, 2, 0])
            att_feature_raw = tf.reshape(trans_feature, [-1, 6])
            W_att = tf.Variable(
                tf.random_uniform([6, 1], -1.0, 1.0),
                name="W_att")
            att_feature = tf.matmul(att_feature_raw, W_att)
            att_feature = tf.reshape(att_feature, [-1, 2*self.n_hidden])

        self.joint_feature = att_feature
        #  self.joint_feature = tf.concat(1, self.output)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.joint_feature, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.n_hidden * 2, num_classes],
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

        # pred
        with tf.name_scope("pred"):
            self.pred = self.predictions

        # Accuracy
        with tf.name_scope("accuracy"):
            self.y_ = tf.argmax(self.input_y, 1)
            self.TP = tf.reduce_sum(tf.cast(tf.cast(self.y_, "bool") & tf.cast(self.predictions, "bool"), "float"))
            correct_predictions = tf.equal(self.predictions, self.y_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Precision
        with tf.name_scope("precision"):
            self.precision = self.TP / tf.cast(tf.reduce_sum(self.predictions), 'float')

        # Recall
        with tf.name_scope("recall"):
            self.recall = self.TP / tf.cast(tf.reduce_sum(self.y_), 'float')

        # F1 score
        with tf.name_scope("f1_score"):
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)

    def BiLSTM(self, x, n_steps, n_input, seq_len):
        x =  tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(0, n_steps, x)
        lstm_fw_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        lstm_bw_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                            dtype=tf.float32, sequence_length=seq_len)

        #  lstm_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        #  outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        #  return outputs[-1]
        return self.last_relevant(outputs, seq_len)

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
