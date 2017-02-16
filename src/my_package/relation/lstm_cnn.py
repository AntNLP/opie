#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/06/23 12:16:24

@author: Changzhi Sun
"""
import sklearn as sk
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell


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
                 n_hidden,
                 vocab_embeddings,
                 threshold=0.5,
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
        self.input14 = tf.placeholder(tf.int32, [None, lengths[14]], name="input_x14")
        self.n_hidden = n_hidden

        self.inputs = [self.input0, self.input1, self.input2, self.input3,
                       self.input4, self.input5, self.input6, self.input7,
                       self.input8, self.input9, self.input10, self.input11,
                       self.input12]

        self.static_embedding = tf.Variable(vocab_embeddings, trainable=False, name="static-embedding", dtype=tf.float32)

        self.input_y = tf.placeholder("float", [None, num_classes], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.embedded_expanded = []
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)


        # sentence tokens
        self.sentence_tokens = tf.concat(1, [self.input1, self.input3, self.input5, self.input7, self.input9])
        self.sentence_postag = tf.concat(1, [self.input2, self.input4, self.input6, self.input8, self.input10])

        # Embedding layer
        with tf.device('/cpu:0'):
            W0 = tf.Variable(
                tf.random_uniform([2, embedding_size], -1.0, 1.0),
                name="W0")
            W1 = tf.Variable(
                tf.random_uniform([3, embedding_size], -1.0, 1.0),
                name="W1")
            self.word_embedding = tf.Variable(
                    tf.random_uniform([word_vocab_size, embedding_size], -1.0, 1.0), name="word-embedding")
            W2 = tf.Variable(
                tf.random_uniform([postag_vocab_size, embedding_size], -1.0, 1.0),
                name="W2")
            self.embeded_position = tf.nn.embedding_lookup(W0, self.inputs[0])
            self.embeded_position_expanded = tf.expand_dims(self.embeded_position, -1)

            self.deptype_embeded = tf.nn.embedding_lookup(W1, self.input13)
            self.deptype_embeded_expanded = tf.expand_dims(self.deptype_embeded, -1)

            sentence_word_embed = tf.nn.embedding_lookup(self.word_embedding, self.sentence_tokens)
            sentence_postag_embed = tf.nn.embedding_lookup(W2, self.sentence_postag)
            self.sentence_embed = tf.concat(2, [sentence_word_embed, sentence_postag_embed])

            # sentence bi-lstm
            with tf.variable_scope("sentence-BiLSTM"):
                self.sentence_outputs = self.BiLSTM(
                    self.sentence_embed,
                    lengths[1]+lengths[3]+lengths[5]+lengths[7]+lengths[9],
                    embedding_size*2)

            i, k = 1, 1
            lengths[0] = 0
            while i < len(self.inputs):
                with tf.variable_scope("embedding-%d" % k):
                    static_word_embedded = tf.nn.embedding_lookup(self.static_embedding, self.inputs[i])
                    word_embedded = tf.nn.embedding_lookup(self.word_embedding, self.inputs[i])
                    postag_embedded = tf.nn.embedding_lookup(W2, self.inputs[i+1])
                    postag_embedded_expanded = tf.expand_dims(postag_embedded, -1)
                    embeded_position_tile = tf.tile(self.embeded_position_expanded,
                        [1, lengths[i], 1, 1])
                    word_embedded_expanded = tf.expand_dims(word_embedded, -1)
                    static_word_embedded_expanded = tf.expand_dims(static_word_embedded, -1)
                    if i <= 10:
                        embeded_sentence = tf.slice(self.sentence_outputs,
                                                    [0, lengths[i-1], 0],
                                                    [-1, lengths[i], 2*self.n_hidden])
                        embeded_sentence_expanded = tf.expand_dims(embeded_sentence, -1)
                        self.embedded_expanded.append(
                            tf.concat(2, [word_embedded_expanded,
                                        #  static_word_embedded_expanded,
                                          postag_embedded_expanded,
                                          embeded_sentence_expanded,
                                          embeded_position_tile]))
                    else:
                        self.embedded_expanded.append(
                            tf.concat(2, [ word_embedded_expanded,
                                           #  static_word_embedded_expanded,
                                           postag_embedded_expanded,
                                           self.deptype_embeded_expanded]))
                k += 1
                i += 2

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for k in range(len(self.embedded_expanded)):
            pooled_outputs.append([])
        W = []
        b = []
        conv = []
        h = []
        pooled = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("phrase-conv-maxpool-%s" % filter_size):
                # Convolution Layer
                for k in range(len(self.embedded_expanded)):
                    if k == len(self.embedded_expanded) - 1:
                        filter_shape = [filter_size, 3*embedding_size, 1, num_filters]
                    else:
                        filter_shape = [filter_size, 3*embedding_size+2*self.n_hidden, 1, num_filters]

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
                        ksize=[1, lengths[2*(k+1)] - filter_size + 1, 1, 1],
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

        #  num_part = len(pooled_outputs)
        # attention
        with tf.name_scope("attention"):
            # LSTM+OBT
            num_part = 5
            trans_feature = tf.pack(self.h_pool_flat[0:5])
            #  trans_feature = tf.pack(self.h_pool_flat)
            trans_feature = tf.transpose(trans_feature, [1, 2, 0])
            att_feature_raw = tf.reshape(trans_feature, [-1, num_part])
            W_att = tf.Variable(
                tf.random_uniform([num_part, 1], -1.0, 1.0),
                name="W_att")
            att_feature = tf.matmul(att_feature_raw, W_att)
            att_feature = tf.reshape(att_feature, [-1, num_filters_total])

        self.joint_feature = att_feature
        #  self.joint_feature = tf.concat(1, self.h_pool_flat)
        #  self.joint_feature = self.h_pool_flat[-1]

        #LSTM+B
        #  self.joint_feature = self.h_pool_flat[2]

        # lstm
        #  with tf.name_scope("lstm"):
            #  seq_len = tf.slice(self.input13, [0, 11], [-1, 1])
            #  seq_len = tf.reshape(seq_len, [-1])
            #  self.outputs = self.last_relevant(
                #  self.BiLSTM(self.dep_embedded, lengths[11], embedding_size*3, seq_len), seq_len)

        #  self.joint_feature = tf.concat(1, [self.joint_feature, self.outputs])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.joint_feature, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                #  shape=[num_filters_total + 2*self.n_hidden, num_classes],
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.scores_proba = tf.reshape(tf.slice(tf.nn.softmax(self.scores), [0, 1], [-1, 1]), [-1])
            #  self.predictions = tf.cast(tf.greater(self.scores_proba, threshold), tf.int64)
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

    def BiLSTM(self, x, n_steps, n_input, seq_len=None):
        x =  tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(0, n_steps, x)
        lstm_fw_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        lstm_bw_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                            dtype=tf.float32, sequence_length=seq_len)
        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
