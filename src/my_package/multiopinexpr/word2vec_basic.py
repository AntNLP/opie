# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import re
import random
import zipfile
import sys
import getopt

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from my_package.sentence import Sentence
from my_package.static import Static
from my_package.scripts import save_pickle_file

data_index = 0
# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = f.read(f.namelist()[0]).split()
  return [e.decode("utf8") for e in data]


def generate_data():
    """Extract the first file enclosed in a zip file as a list of words"""

    f = open(os.path.join(multi_opin_expr_dir, "clean_replace_text"),
             "w", encoding="utf8")
    g = open(os.path.join(multi_opin_expr_dir, "replace_text"),
             "r", encoding="utf8")
    for line in g:
        text, _ = line.split('\t')
        for token in text.split(' '):
            if token in Static.stopwords:
                continue
            if re.search(r"^\W*$", token):
                continue
            print(token, end=" ", file=f)
    g.close()
    f.close()
    filename = os.path.join(multi_opin_expr_dir, "text.zip")
    with zipfile.ZipFile(filename, "w") as myzip:
        myzip.write(os.path.join(multi_opin_expr_dir, "clean_replace_text"))
    return filename


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
    dictionary = {}
    for word, wc in count:
        if wc >= min_count:
            dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')

    plt.savefig(filename)



def usage():
    '''print help information'''
    print("word2vec_basic.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hd:",
            ["help", "domain="])
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
    domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data", "domains", domain)
    multi_opin_expr_dir = os.path.join(domain_dir, "multiopinexpr")

    # Step 1: Generate the data.
    filename = generate_data()
    words = read_data(filename)
    print(words[:10])
    print('Data size', len(words))

    # Step 2: Build the dictionary and replace rare words with UNK token.
    min_count = 3
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    save_pickle_file(os.path.join(multi_opin_expr_dir, "dictionary.pickle"),
                     dictionary)
    vocabulary_size = len(dictionary)
    save_pickle_file(
        os.path.join(multi_opin_expr_dir, "reverse_dictionary.pickle"),
        reverse_dictionary)
    del words  # Hint to reduce memory.
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    # Step 3: Function to generate a training batch for the skip-gram model.
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],
            '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    # Step 4: Build and train a skip-gram model.
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64    # Number of negative examples to sample.

    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                                num_sampled, vocabulary_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            # Add variable initializer.
            init = tf.initialize_all_variables()

    # Step 5: Begin training.
    #  num_steps = 1
    num_steps = 2000001

    with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
        init.run()
        print("Initialized")

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()

    save_pickle_file(
        os.path.join(multi_opin_expr_dir, "final_embeddings.pickle"),
        final_embeddings)
    #  # Step 6: Visualize the embeddings.
    #  try:
        #  from sklearn.manifold import TSNE
        #  import matplotlib.pyplot as plt

        #  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        #  plot_only = 500
        #  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
        #  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        #  plot_with_labels(low_dim_embs, labels)

    #  except ImportError:
        #  print("Please install sklearn and matplotlib to visualize embeddings.")
