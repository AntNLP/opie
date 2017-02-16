#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/05/24 20:48:05

@author: Changzhi Sun
"""
import numpy as np
import re
import os
import itertools
from collections import Counter

from my_package.static import Static
from my_package.scripts import load_pickle_file


def filter_complex(sentence, phrase):
    signal = set([",", ".", "!"])
    for i in phrase:
        if sentence.tokens[i] in signal:
            return True
    return False


def pseudo_data_generator():
    domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data/domains", "reviews_Cell_Phones_and_Accessories")
    btsp_dir = os.path.join(domain_dir, "pickles", "bootstrapping")
    near_dir = os.path.join(domain_dir, "near")
    f = open(os.path.join(near_dir, "pseudo_train"), "w", encoding="utf8")
    i = 1
    filename = os.path.join(btsp_dir, "bootstrapping_sentences_%d.pickle" % i)
    x_text = []
    y = []
    mark = set()
    positive_count = 0
    negative_count = 0
    while os.path.exists(filename + ".bz2"):
        sentences = load_pickle_file(filename)
        for sentence in sentences:
            for e1, e2 in sentence.relation:
                min_i = min(itertools.chain(e1, e2))
                max_i = max(itertools.chain(e1, e2))
                if len(e2) > 1 or (max_i - min_i + 1 == len(e1) + len(e2) and max_i - min_i < 9):
                    if len(e2) > 1:
                        phrase = list(e2)
                    else:
                        phrase = list(range(min_i, max_i+1))
                    t = []
                    for j in phrase:
                        t.append((sentence.tokens[j].lower(), sentence.pos_tag[j]))
                    if tuple(t) not in mark and not filter_complex(sentence, phrase):
                        x_text.append(t)
                        print("1\t{0}\t{1}".format(" ".join([sentence.tokens[e].lower() for e in phrase]),
                                                   " ".join([sentence.pos_tag[e] for e in phrase])), file=f)
                        y.append([0, 1])
                        mark.add(tuple(t))
                        positive_count += 1
                else:
                    phrase = list(itertools.chain(e1, e2))
                    t = []
                    for j in phrase:
                        t.append((sentence.tokens[j].lower(), sentence.pos_tag[j]))
                    if tuple(t) not in mark:
                        x_text.append(t)
                        print("0\t{0}\t{1}".format(" ".join([sentence.tokens[e].lower() for e in phrase]),
                                                   " ".join([sentence.pos_tag[e] for e in phrase])), file=f)
                        y.append([1, 0])
                        mark.add(tuple(t))
                        negative_count += 1
        i += 1
        filename = os.path.join(btsp_dir, "bootstrapping_sentences_%d.pickle" % i)
    f.close()
    return [x_text, y]


def load_data_and_labels(remove_seed=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data/domains", "reviews_Cell_Phones_and_Accessories")
    filename = os.path.join(domain_dir, "near", "complex_data.ann")
    f = open(filename, "r", encoding="utf8")
    y = []
    x_text = []
    for line in f:
        label, phrase, postag = line.strip().split('\t')
        have_seed = False
        for word in phrase.split(' '):
            if word in Static.opinwd:
                have_seed = True
                break
        if remove_seed and have_seed:
            continue
        x_text.append(list(zip(phrase.split(" "), postag.split(" "))))
        if label == "1":
            y.append([0, 1])
        else:
            y.append([1, 0])
    f.close()
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence
        for j in range(num_padding):
            new_sentence.append((padding_word, padding_word))
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter()
    postag_counts = Counter()
    for sentence in sentences:
        for word, postag in sentence:
            word_counts[word] += 1
            postag_counts[postag] += 1
    # Mapping from index to word
    vocabulary_inv = {}
    vocabulary_inv["word"] = [x[0] for x in word_counts.most_common()]
    vocabulary_inv["word"] = list(sorted(vocabulary_inv["word"]))
    vocabulary_inv["postag"] = [x[0] for x in postag_counts.most_common()]
    vocabulary_inv["postag"] = list(sorted(vocabulary_inv["postag"]))
    # Mapping from word to index
    vocabulary = {}
    vocabulary["word"] = {x: i for i, x in enumerate(vocabulary_inv["word"])}
    vocabulary["postag"] = {x: i for i, x in enumerate(vocabulary_inv["postag"])}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = []
    for sentence in sentences:
        t = []
        for word, postag in sentence:
            t.append([vocabulary["word"][word], vocabulary["postag"][postag]])
        x.append(t)
    y = np.array(labels)
    return [np.array(x), y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = pseudo_data_generator()
    #  print(labels.count([1, 0]))
    #  print(labels.count([0, 1]))
    #  print(len(labels))
    r = len(labels)
    ann_sentences, ann_labels = load_data_and_labels()
    sentences.extend(ann_sentences)
    labels.extend(ann_labels)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, r]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

x, y, vocabulary, vocabulary_inv, r = load_data()
