#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/05/24 20:44:03

@author: Changzhi Sun
"""
import sklearn as sk
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import input_data
from cnn import CNN

from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
#  tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
#  tf.flags.DEFINE_integer("num_filters", 50, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 200)")
#  tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")

#  domain = "reviews_Movies_and_TV"
#  domain = "reviews_Pet_Supplies"
domain = "reviews_Cell_Phones_and_Accessories"
#  domain = "reviews_Grocery_and_Gourmet_Food"

domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data/domains", domain)
pickle_dir = os.path.join(domain_dir, "pickles")
complex_dir = os.path.join(domain_dir, "complex")


#  data = input_data.load_datas(domain_dir, pickle_dir, 1, 2)
#  save_pickle_file("data.pickle", data)
#  X, y, vocabulary, x_string, r = load_pickle_file("data.pickle")
X, y, vocabulary, x_string, r = input_data.load_datas(domain_dir, pickle_dir, 1, 2)

#  record = load_pickle_file(os.path.join(domain_dir, "complex", "test", "record.pickle"))
# Split train/test set
# TODO: This is very crude, should use cross-validation
X_train, X_dev = X[:r], X[r:]
y_train, y_dev = y[:r], y[r:]
# Randomly shuffle data
#  np.random.seed(10)
#  shuffle_indices = np.random.permutation(np.arange(len(y_train)))
#  x_train_phrase = x_train_phrase[shuffle_indices]
#  x_train_context = x_train_context[shuffle_indices]
#  y_train = y_train[shuffle_indices]

#  x_train, x_dev = x[:r], x[r:]
#  y_train, y_dev = y[:r], y[r:]
lengths = [len(X_train[0][0]), len(X_train[0][1]), len(X_train[0][2]),
           len(X_train[0][3]), len(X_train[0][4]), len(X_train[0][5])]
print("Word Vocabulary Size: {:d}".format(len(vocabulary["word"])))
print("Postag Vocabulary Size: {:d}".format(len(vocabulary["postag"])))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print("L context: ", lengths[0])
print("complex : ", lengths[2])
print("R context: ", lengths[4])


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN(
            lengths=lengths,
            num_classes=2,
            word_vocab_size=len(vocabulary["word"]),
            postag_vocab_size = len(vocabulary["postag"]),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(X_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input0: np.array([e[0] for e in X_batch]),
              cnn.input1: np.array([e[1] for e in X_batch]),
              cnn.input2: np.array([e[2] for e in X_batch]),
              cnn.input3: np.array([e[3] for e in X_batch]),
              cnn.input4: np.array([e[4] for e in X_batch]),
              cnn.input5: np.array([e[5] for e in X_batch]),
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            #  embed = sess.run(cnn.joint_feature, feed_dict)
            #  print(embed.shape)
            _, step, summaries, loss, f1_score = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, f1 score {:g}".format(time_str, step, loss, f1_score))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(X_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input0: np.array([e[0] for e in X_batch]),
              cnn.input1: np.array([e[1] for e in X_batch]),
              cnn.input2: np.array([e[2] for e in X_batch]),
              cnn.input3: np.array([e[3] for e in X_batch]),
              cnn.input4: np.array([e[4] for e in X_batch]),
              cnn.input5: np.array([e[5] for e in X_batch]),
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            scores = sess.run(cnn.scores_proba, feed_dict)
            print(scores)
            step, summaries, loss, precision, recall, f1_score, pred, pred_ = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.precision, cnn.recall, cnn.f1_score, cnn.pred, cnn.pred_],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, f1 score {:g}".format(time_str, step, loss, f1_score))
            print("Precision", precision)
            print("Recall", recall)
            print("f1_score", f1_score)
            if writer:
                writer.add_summary(summaries, step)
            return pred

        # Generate batches
        batches = input_data.batch_iter(
            list(zip(X_train, y_train)),
            FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            X_batch, y_batch = zip(*batch)
            train_step(X_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
               pred = dev_step(X_dev, y_dev, writer=dev_summary_writer)
                #  with open("pred.dump", "w", encoding="utf8") as out:
                    #  for i in range(len(pred)):
                        #  print(record[i][0], file=out)
                        #  print("{0}\t{1}\t{2}".format(pred[i], record[i][1], record[i][2]), file=out)
                        #  print(file=out)
                #  print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
