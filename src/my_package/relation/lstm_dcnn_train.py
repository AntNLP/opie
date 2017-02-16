#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/06/23 12:18:05

@author: Changzhi Sun
"""
import sklearn as sk
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from lstm_cnn import CNN
from normalization import handle_normalize
import sys
import getopt

from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.scripts import load_bin_vec

def usage():
    '''print help information'''
    print("lstm_dcnn_train.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("--num_filters")
    print("--n_hidden")
    print("--drop_prob")
    print("--l2")
    print("--batch_size")
    print("--num_epochs")
    print("--complex_op: 0 no complex   1 complex-0.5   2 complex-0.8")

try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:",
                                ["help",
                                "domain=",
                                "num_filters=",
                                "n_hidden=",
                                "drop_prob=",
                                "l2=",
                                "batch_size=",
                                "num_epochs=",
                                "complex_op="
                                ])
except getopt.GetoptError:
    print("命令行参数输入错误！")
    usage()
    sys.exit(1)
for op, value in opts:
    if op in ("-h", "--help"):
        usage()
        sys.exit()
    if op in ("-d", "--domain"):
        domain = value
    if op in ("--num_filters"):
        num_filters = int(value)
    if op in ("--n_hidden"):
        n_hidden = int(value)
    if op in ("--drop_prob"):
        dropout_keep_prob = value
    if op in ("--l2"):
        l2 = value
    if op in ("--batch_size"):
        batch_size = int(value)
    if op in ("--num_epochs"):
        num_epochs = int(value)
    if op in ("--complex_op"):
        complex_op = int(value)
dropout_keep_prob = float(dropout_keep_prob)
l2 = float(l2)

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", num_filters, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("n_hidden", n_hidden, "Number of hidden of LSTM (default: 64)")
tf.flags.DEFINE_float("dropout_keep_prob", dropout_keep_prob, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", l2, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", batch_size, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", num_epochs, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 20, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

Max = {
        "NN-P": 0,
        "NN-R": 0,
        "NN-F": 0,
        "P": 0,
        "R":0,
        "F": 0,
        "scores":[]
      }

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data/domains", domain)
relation_dir = os.path.join(domain_dir, "relation")
test_dir = os.path.join(relation_dir, "test")
word2vect_path = os.path.join(os.getenv("OPIE_DIR"), "tools", "GoogleNews-vectors-negative300.bin")
data = data_helper.load_datas(domain_dir, 1, 2, complex_op=complex_op)
save_pickle_file(os.path.join(relation_dir, "data.pickle"), data)
data = load_pickle_file(os.path.join(relation_dir, "data.pickle"))
X, y, vocabulary, r = data
ann_dict = data_helper.get_ann(os.path.join(test_dir, "ann"))
sentences = load_pickle_file(os.path.join(test_dir, "sentences.candidate.pickle"))
vocab_embeddings = load_bin_vec(word2vect_path, vocabulary["word"])
save_pickle_file(os.path.join(relation_dir, "vocab_embeddings.pickle"), vocab_embeddings)
vocab_embeddings = load_pickle_file(os.path.join(relation_dir, "vocab_embeddings.pickle"))

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

lengths = [len(X_train[0][0]), len(X_train[0][1]), len(X_train[0][2]),
           len(X_train[0][3]), len(X_train[0][4]), len(X_train[0][5]),
           len(X_train[0][6]), len(X_train[0][7]), len(X_train[0][8]),
           len(X_train[0][9]), len(X_train[0][10]), len(X_train[0][11]),
           len(X_train[0][12]), len(X_train[0][13])]
print("Word Vocabulary Size: {:d}".format(len(vocabulary["word"])))
print("Postag Vocabulary Size: {:d}".format(len(vocabulary["postag"])))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print("left length: ", lengths[1])
print("min length: ", lengths[3])
print("middle length: ", lengths[5])
print("max length: ", lengths[7])
print("right length: ", lengths[9])
print("dep length: ", lengths[11])

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
            n_hidden=FLAGS.n_hidden,
            vocab_embeddings=np.array(vocab_embeddings),
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

        vocab_embeddings = np.array(vocab_embeddings)
        sess.run(cnn.word_embedding.assign(vocab_embeddings))

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
              cnn.input6: np.array([e[6] for e in X_batch]),
              cnn.input7: np.array([e[7] for e in X_batch]),
              cnn.input8: np.array([e[8] for e in X_batch]),
              cnn.input9: np.array([e[9] for e in X_batch]),
              cnn.input10: np.array([e[10] for e in X_batch]),
              cnn.input11: np.array([e[11] for e in X_batch]),
              cnn.input12: np.array([e[12] for e in X_batch]),
              cnn.input13: np.array([e[13] for e in X_batch]),
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            #  for e in X_batch[0]:
                #  print(e)
            #  embed = sess.run(cnn.embed, feed_dict)
            #  print(embed.shape)
            _, step, summaries, loss, f1_score = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.f1_score],
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
              cnn.input6: np.array([e[6] for e in X_batch]),
              cnn.input7: np.array([e[7] for e in X_batch]),
              cnn.input8: np.array([e[8] for e in X_batch]),
              cnn.input9: np.array([e[9] for e in X_batch]),
              cnn.input10: np.array([e[10] for e in X_batch]),
              cnn.input11: np.array([e[11] for e in X_batch]),
              cnn.input12: np.array([e[12] for e in X_batch]),
              cnn.input13: np.array([e[13] for e in X_batch]),
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, pred, scores = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.pred, cnn.scores_proba],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            filename = os.path.join(test_dir, "test.dump")
            data_helper.dump(filename, sentences, pred)
            handle_normalize(filename)

            data_helper.combine_result(os.path.join(test_dir, 'relation.pattern.normalize'),
                           filename+".normalize",
                           os.path.join(test_dir, "test.dump.combine"))
            handle_normalize(os.path.join(test_dir, "test.dump.combine"))
            prec, rec, f1 = data_helper.calcu_PRF(filename+".normalize", ann_dict)
            print("{}: step {}, loss {:g}, f1 score {:g}".format(time_str, step, loss, f1))
            print("Precision", prec)
            print("Recall", rec)
            print("f1_score", f1)
            precision, recall, f1_score = data_helper.calcu_PRF(filename+".combine.normalize", ann_dict)
            print("combine...")
            print("{}: step {}, loss {:g}, f1 score {:g}".format(time_str, step, loss, f1_score))
            print("Precision", precision)
            print("Recall", recall)
            print("f1_score", f1_score)
            if f1_score > Max["F"]:
                Max["NN-P"] = prec
                Max["NN-R"] = rec
                Max["NN-F"] = f1
                Max["P"] = precision
                Max["R"] = recall
                Max["F"] = f1_score
                Max["scores"] = scores
                print("Max result")
                for key, value in Max.items():
                    print(key, value)
                print()

            if writer:
                writer.add_summary(summaries, step)
            return pred

        # Generate batches
        batches = data_helper.batch_iter(
            list(zip(X_train, y_train)),
            FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            X_batch, y_batch = zip(*batch)
            train_step(X_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                pred = dev_step(X_dev, y_dev,
                                writer=dev_summary_writer)
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

print("Max result")
print(domain)
for key, value in Max.items():
    print(key, value)
print()
i = 1
path = os.path.join(relation_dir, "result_%d.pickle" % i)
while os.path.exists(path+".bz2"):
    i += 1
    path = os.path.join(relation_dir, "result_%d.pickle" % i)
print(path)
Max["sentences"] = sentences
save_pickle_file(path, Max)
